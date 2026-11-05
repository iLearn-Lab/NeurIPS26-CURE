import argparse
from ast import alias
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, CenterCrop, RandomResizedCrop, RandomHorizontalFlip
from torchvision.transforms import InterpolationMode, RandomErasing, ColorJitter, RandomGrayscale, RandomRotation
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch
import numpy as np
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.runtime import prepare_output
from cfg import *
from tools import *
from datasets import *
from datasets.ssl_utils import build_ssl_dataloaders, FixMatchDataLoader
from methods.vp import PaddingVR

class RandAugment:
    """RandAugment implementation for strong augmentation"""
    def __init__(self, n=2, m=10):
        self.n = n  # number of augmentation transformations to apply
        self.m = m  # magnitude for all the transformations
        
    def __call__(self, img):
        ops = [
            lambda img, mag: ColorJitter(brightness=mag/30)(img),
            lambda img, mag: ColorJitter(contrast=mag/30)(img), 
            lambda img, mag: ColorJitter(saturation=mag/30)(img),
            lambda img, mag: RandomRotation(degrees=mag*3)(img),
        ]
        
        for _ in range(self.n):
            op = np.random.choice(ops)
            img = op(img, self.m)
        return img

class FlexMatchCPL:
    """Curriculum Pseudo Labeling for FlexMatch"""
    def __init__(self, num_classes, initial_threshold=0.95, beta=0.95):
        self.num_classes = num_classes
        # self.initial_threshold = initial_threshold
        self.initial_threshold = 1 / num_classes
        self.beta = beta  # momentum for updating thresholds
        
        # Initialize per-class thresholds
        self.class_thresholds = torch.ones(num_classes) * self.initial_threshold
        
        # Track learning status for each class
        self.class_learning_status = torch.zeros(num_classes)
        self.class_counts = torch.zeros(num_classes)
        
    def update_threshold(self, predictions, targets, device):
        """Update per-class thresholds based on model's learning status"""
        self.class_thresholds = self.class_thresholds.to(device)
        self.class_learning_status = self.class_learning_status.to(device)
        self.class_counts = self.class_counts.to(device)
        
        with torch.no_grad():
            # Get predicted classes and their confidence
            max_probs, pred_classes = torch.max(F.softmax(predictions, dim=1), dim=1)
            
            # Update learning status for each class
            for c in range(self.num_classes):
                class_mask = (targets == c)
                if class_mask.sum() > 0:
                    # Calculate accuracy for this class
                    class_acc = (pred_classes[class_mask] == targets[class_mask]).float().mean()
                    
                    # Update learning status with momentum
                    self.class_learning_status[c] = (self.beta * self.class_learning_status[c] + 
                                                    (1 - self.beta) * class_acc)
                    self.class_counts[c] += class_mask.sum()
                    
                    # Adjust threshold based on learning status
                    # If class is learned well, lower threshold to include more pseudo-labels
                    # If class is difficult, keep higher threshold
                    if self.class_counts[c] >= 10:  # Only adjust after sufficient samples
                        learning_rate = self.class_learning_status[c]
                        # Dynamic threshold: well-learned classes get lower thresholds
                        target_threshold = self.initial_threshold * (1 - 0.3 * learning_rate)
                        target_threshold = max(0.5, min(0.95, target_threshold))  # Clamp between 0.5 and 0.95
                        
                        # Smooth update
                        self.class_thresholds[c] = (0.9 * self.class_thresholds[c] + 
                                                   0.1 * target_threshold)
    
    def get_mask(self, predictions, device):
        """Get pseudo-label mask based on per-class thresholds"""
        self.class_thresholds = self.class_thresholds.to(device)
        
        probs = F.softmax(predictions, dim=1)
        max_probs, pred_classes = torch.max(probs, dim=1)
        
        # Create mask based on per-class thresholds
        mask = torch.zeros_like(max_probs, dtype=torch.bool)
        for c in range(self.num_classes):
            class_mask = (pred_classes == c)
            threshold_mask = max_probs >= self.class_thresholds[c]
            mask = mask | (class_mask & threshold_mask)
            
        return mask, pred_classes
    
    def get_threshold_info(self):
        """Get current threshold information for logging"""
        return {
            'mean_threshold': self.class_thresholds.mean().item(),
            'min_threshold': self.class_thresholds.min().item(),
            'max_threshold': self.class_thresholds.max().item(),
            'mean_learning_status': self.class_learning_status.mean().item()
        }

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=['caltech101' ,'dtd' ,'eurosat' ,'fgvc' ,'food101',
                                         'oxford_flowers' ,'oxford_pets','stanford_cars' ,'sun397' ,'ucf101', 'resisc45',
                                         'imagenet'], default='caltech101')
    p.add_argument('--alpha', type=float, default=0.5) # alpha: the balance hyperparameter lambda
    p.add_argument('--num_attr', type=int, default=20) # number of attributes
    p.add_argument('--k', type=int, default=3) # k for knn attribute selection
    p.add_argument('--epoch', type=int, default=200)   # total epochs
    p.add_argument('--lr', type=float, default=40) # the initial learning rate
    p.add_argument('--input_size', type=int, default=192) # 224*224 images with VR pattern frame size=16
    p.add_argument('--shot', type=int, default=10) # few-shot training set
    p.add_argument('--bs', type=int, default=4)
    
    # FlexMatch specific parameters
    p.add_argument('--initial_threshold', type=float, default=0.95) # initial confidence threshold
    p.add_argument('--lambda_u', type=float, default=1.0) # weight for unlabeled loss
    p.add_argument('--unlabeled_ratio', type=int, default=7) # ratio of unlabeled to labeled batch size
    p.add_argument('--beta', type=float, default=0.95) # momentum for CPL threshold updates
    
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--backbone-weights", type=str, default=None)
    args = p.parse_args()
    print(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    exp = "results/fs_ar_flexmatch"
    save_path = os.path.join(exp, args.dataset + '_' + str(args.shot) + 'k' + str(args.k) + 'a' + str(args.alpha) + 
                            'tau' + str(args.initial_threshold) + 'lu' + str(args.lambda_u) + 's' + str(args.seed))
    if args.output is not None:
        save_path = str(args.output.resolve())
    save_path = prepare_output(save_path, args)

    # Load pretrained CLIP
    model, _ = clip.load(args.backbone_weights or "ViT-B/16", device=device)
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    model = model.to(device)

    # Data augmentation - Weak augmentation (standard)
    weak_process = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(p=0.5),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
    ])
    
    # Strong augmentation (RandAugment + Cutout)
    strong_process = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(p=0.5),
        RandAugment(n=2, m=10),  # RandAugment
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
        RandomErasing(p=1, scale=(0.01, 0.01), ratio=(1, 1), value=127),  # Cutout
    ])

    test_process = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(args.input_size),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
    ])

    # Loading dataset with semi-supervised setup
    labeled_loader, unlabeled_loader, testloader, classes = build_ssl_dataloaders(
        args.dataset, DOWNSTREAM_PATH, weak_process, weak_process, strong_process, test_process, 
        batch_size=args.bs, shot=args.shot, mu=args.unlabeled_ratio,
        seed=args.seed
    )
    
    # Create combined data loader for FlexMatch
    flexmatch_loader = FixMatchDataLoader(labeled_loader, unlabeled_loader)

    # Loading Attributes and Text Embeddings
    model_for_text = model.module if isinstance(model, torch.nn.DataParallel) else model
    if torch.cuda.device_count() > 1:
        model = model.module
    TEMPLATES = [basic_template]
    txt_emb = clip_classifier(classes, TEMPLATES, model)



    # Initialize FlexMatch CPL
    num_classes = len(classes)
    cpl = FlexMatchCPL(num_classes, args.initial_threshold, args.beta)

    # Repurposing CLIP for Downstream Classification
    def network(x):
        x_emb = model.encode_image(x)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * x_emb @ txt_emb
        return logits

        
    # Visual Reprogramming
    visual_reprogram = PaddingVR(224, args.input_size).to(device)

    # Optimizer
    # optimizer = torch.optim.SGD(visual_reprogram.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # t_max = args.epoch * len(labeled_loader)
    # scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    optimizer = torch.optim.SGD(visual_reprogram.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    warm_epoch = 5
    warmup_steps = max(1, warm_epoch * len(flexmatch_loader))
    cosine_steps = max(1, (args.epoch - warm_epoch) * len(flexmatch_loader))

    # 1) 线性 warmup：从 1e-8 * lr 线性涨到 1.0 * lr
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6, 
        end_factor=1.0,
        total_iters=warmup_steps
    )

    # 2) 余弦退火：从 lr 衰减到 eta_min
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=1e-4
    )

    # 串联两个调度器
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    # Make Dir
    os.makedirs(save_path, exist_ok=True)

    # Train with FlexMatch
    # visual_reprogram.train()
    best_acc = 0.
    
    progress_bar = tqdm(total=args.epoch, desc='Training FlexMatch', leave=True)

    for epoch in range(args.epoch):
        total_num = 0
        true_num = 0
        loss_sum = 0
        supervised_loss_sum = 0
        unsupervised_loss_sum = 0
        mask_prob = 0

        length = len(flexmatch_loader)
        visual_reprogram.train()
        
        for i, (labeled_batch, unlabeled_batch) in enumerate(flexmatch_loader):
            # Process labeled data
            labeled_x, labeled_y = labeled_batch
            labeled_x, labeled_y = labeled_x.to(device), labeled_y.to(device)
                
            # Process unlabeled data
            unlabeled_x_weak, unlabeled_x_strong, _ = unlabeled_batch
            unlabeled_x_weak = unlabeled_x_weak.to(device)
            unlabeled_x_strong = unlabeled_x_strong.to(device)
            
            optimizer.zero_grad()
            
            # Supervised loss on labeled data
            labeled_logits = network(visual_reprogram(labeled_x))
            supervised_loss = F.cross_entropy(labeled_logits, labeled_y, reduction='mean')
            
            # Update CPL thresholds based on labeled data performance
            cpl.update_threshold(labeled_logits, labeled_y, device)
            
            # Generate pseudo-labels on weakly augmented unlabeled data
            weak_logits = network(visual_reprogram(unlabeled_x_weak))
            
            # Get dynamic mask and pseudo-labels using CPL
            mask, pseudo_labels = cpl.get_mask(weak_logits, device)
            
            # Unsupervised loss on strongly augmented unlabeled data
            if mask.sum() > 0:
                strong_logits = network(visual_reprogram(unlabeled_x_strong))
                unsupervised_loss = F.cross_entropy(
                    strong_logits[mask], pseudo_labels[mask], reduction='mean'
                )
            else:
                unsupervised_loss = torch.tensor(0).to(device)
                attr_consistency_loss = torch.tensor(0).to(device)

            # Total loss
            total_loss = supervised_loss + args.lambda_u * unsupervised_loss
            
            total_loss.backward()
            optimizer.step()

            # Clamp logit scale
            model_to_use = model.module if isinstance(model, torch.nn.DataParallel) else model
            model_to_use.logit_scale.data = torch.clamp(model_to_use.logit_scale.data, 0, 4.6052)
            
            # Statistics
            total_num += labeled_y.shape[0]
            true_num += torch.argmax(labeled_logits, 1).eq(labeled_y).float().sum().item()

            loss_sum += total_loss.item()
            supervised_loss_sum += supervised_loss.item()
            unsupervised_loss_sum += unsupervised_loss.item()
            mask_prob += mask.float().mean().item()
            
            scheduler.step()
            
        # Calculate training metrics
        train_acc = true_num / total_num if total_num > 0 else 0.0
        avg_loss = loss_sum / len(flexmatch_loader) if len(flexmatch_loader) > 0 else 0.0
        avg_sup_loss = supervised_loss_sum / len(flexmatch_loader) if len(flexmatch_loader) > 0 else 0.0
        avg_unsup_loss = unsupervised_loss_sum / len(flexmatch_loader) if len(flexmatch_loader) > 0 else 0.0
        avg_mask_prob = mask_prob / len(flexmatch_loader) if len(flexmatch_loader) > 0 else 0.0
        
        # Get CPL threshold information
        threshold_info = cpl.get_threshold_info()
        
        progress_bar.set_postfix({
            'Epoch': epoch + 1, 
            'Train Acc': f'{train_acc:.4f}',
            'Loss': f'{avg_loss:.4f}',
            'Mask Prob': f'{avg_mask_prob:.4f}',
            'Avg Thresh': f'{threshold_info["mean_threshold"]:.3f}'
        })

        # Evaluation
        visual_reprogram.eval()
        test_total_num = 0
        test_true_num = 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                fx = network(visual_reprogram(x))
                test_total_num += y.size(0)
                test_true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        
        test_acc = test_true_num / test_total_num
        progress_bar.set_postfix({
            'Epoch': epoch + 1, 
            'Train Acc': f'{train_acc:.4f}',
            'Test Acc': f'{test_acc:.4f}', 
            'Best Acc': f'{best_acc:.4f}',
            'Mask Prob': f'{avg_mask_prob:.4f}',
            'Avg Thresh': f'{threshold_info["mean_threshold"]:.3f}'
        }, refresh=False)

        # Save model
        state_dict = {
            "visual_prompt_dict": visual_reprogram.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "avg_loss": avg_loss,
            "avg_sup_loss": avg_sup_loss,
            "avg_unsup_loss": avg_unsup_loss,
            "avg_mask_prob": avg_mask_prob,
            "threshold_info": threshold_info
        }

        if test_acc > best_acc:
            best_acc = test_acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
            
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(state_dict, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))
            
        progress_bar.update(1)
        visual_reprogram.train()

    progress_bar.close()
    print(f"\nTraining completed!")
    print(f"Final best test accuracy: {best_acc:.4f}")
    print(f"Results saved to: {save_path}")
    
    # Print final CPL statistics
    final_threshold_info = cpl.get_threshold_info()
    print(f"\nFinal CPL Statistics:")
    print(f"Mean threshold: {final_threshold_info['mean_threshold']:.4f}")
    print(f"Min threshold: {final_threshold_info['min_threshold']:.4f}")
    print(f"Max threshold: {final_threshold_info['max_threshold']:.4f}")
    print(f"Mean learning status: {final_threshold_info['mean_learning_status']:.4f}")
