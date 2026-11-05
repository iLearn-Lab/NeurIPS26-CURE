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
from datasets.ssl_utils import build_ssl_dataloaders
from methods.vp import PaddingVR

class MixMatchDataLoader:
    """
    Custom data loader for MixMatch that handles labeled and unlabeled data synchronization
    """
    def __init__(self, labeled_loader, unlabeled_loader):
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.labeled_iter = iter(labeled_loader)
        self.unlabeled_iter = iter(unlabeled_loader)
        
    def __iter__(self):
        # Reset iterators at the beginning of each epoch
        self.labeled_iter = iter(self.labeled_loader)
        self.unlabeled_iter = iter(self.unlabeled_loader)
        return self
    
    def __next__(self):
        try:
            labeled_batch = next(self.labeled_iter)
        except StopIteration:
            self.labeled_iter = iter(self.labeled_loader)
            labeled_batch = next(self.labeled_iter)
            
        try:
            unlabeled_batch = next(self.unlabeled_iter)
        except StopIteration:
            raise StopIteration
        
        return labeled_batch, unlabeled_batch
    
    def __len__(self):
        # Return the length of unlabeled_loader so iteration stops when unlabeled data is exhausted
        return len(self.unlabeled_loader)

def mixup(x, y, alpha=0.75):
    """
    Apply MixUp augmentation to data and labels
    
    Args:
        x: Input data tensor
        y: Labels (can be one-hot or class indices)
        alpha: Beta distribution parameter for lambda sampling
    
    Returns:
        Mixed data and labels
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    if len(y.shape) == 1:  # Class indices
        # Convert to one-hot for mixing
        num_classes = y.max().item() + 1
        y_onehot = torch.zeros(batch_size, num_classes).to(y.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
        mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index, :]
    else:  # Already one-hot or soft labels
        mixed_y = lam * y + (1 - lam) * y[index, :]
    
    return mixed_x, mixed_y

def sharpen(p, T=0.5):
    """
    Sharpen probability distribution
    
    Args:
        p: Probability distribution
        T: Temperature parameter (lower = sharper)
    
    Returns:
        Sharpened probability distribution
    """
    return (p ** (1/T)) / (p ** (1/T)).sum(dim=1, keepdim=True)

def guess_labels(model, unlabeled_data, K=2):
    """
    Generate pseudo-labels for unlabeled data using label guessing
    
    Args:
        model: Trained model
        unlabeled_data: Unlabeled input data
        K: Number of augmentations to average over
    
    Returns:
        Guessed labels (sharpened average predictions)
    """
    model.eval()
    with torch.no_grad():
        # Average predictions over K augmentations
        avg_predictions = None
        for k in range(K):
            predictions = F.softmax(model(unlabeled_data), dim=1)
            if avg_predictions is None:
                avg_predictions = predictions
            else:
                avg_predictions += predictions
        
        avg_predictions /= K
        
        # Sharpen the averaged predictions
        sharpened_labels = sharpen(avg_predictions)
    
    model.train()
    return sharpened_labels

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
    
    # MixMatch specific parameters
    p.add_argument('--lambda_u', type=float, default=75.0) # weight for unlabeled loss
    p.add_argument('--unlabeled_ratio', type=int, default=7) # ratio of unlabeled to labeled batch size
    p.add_argument('--mixup_alpha', type=float, default=0.75) # MixUp alpha parameter
    p.add_argument('--temperature', type=float, default=0.5) # Temperature for label sharpening
    p.add_argument('--K', type=int, default=2) # Number of augmentations for label guessing
    
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--backbone-weights", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    exp = "results/fs_attrvr_mixmatch"
    save_path = os.path.join(exp, args.dataset + '_' + str(args.shot) + 'k' + str(args.k) + 'a' + str(args.alpha) + 
                            'T' + str(args.temperature) + 'lu' + str(args.lambda_u) + 's' + str(args.seed))
    if args.output is not None:
        save_path = str(args.output.resolve())
    save_path = prepare_output(save_path, args)

    # Load CLIP model
    model, _ = clip.load(args.backbone_weights or "ViT-B/16", device=device)
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    model = model.to(device)

    # Data augmentation transforms
    weak_process = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(p=0.5),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
    ])
    
    # For MixMatch, we use the same weak augmentation for both labeled and unlabeled data
    # The strong augmentation comes from MixUp rather than visual augmentations
    strong_process = weak_process

    test_process = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(args.input_size),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
    ])

    # Build SSL data loaders
    labeled_loader, unlabeled_loader, testloader, classes = build_ssl_dataloaders(
        args.dataset, DOWNSTREAM_PATH, weak_process, weak_process, strong_process, test_process, 
        batch_size=args.bs, shot=args.shot, mu=args.unlabeled_ratio,
        seed=args.seed
    )
    
    # Create MixMatch data loader
    mixmatch_loader = MixMatchDataLoader(labeled_loader, unlabeled_loader)

    # Build attribute classifiers
    model_for_text = model.module if isinstance(model, torch.nn.DataParallel) else model
    txt_emb_desattr = clip_attr_classifier(classes, model_for_text, 'attributes/gpt3/' + args.dataset + '_des.json', num_attr=args.num_attr)
    txt_emb_distattr = clip_attr_classifier(classes, model_for_text, 'attributes/gpt3/' + args.dataset + '_dist.json', num_attr=args.num_attr)

    # Define network function
    def network(x):
        img_emb = model.encode_image(x)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        exp = model.logit_scale.exp()
        
        desattr_logits = getLogitsCLIP(args.num_attr, exp, img_emb, txt_emb_desattr, k=args.k)
        distattr_logits = getLogitsCLIP(args.num_attr, exp, img_emb, txt_emb_distattr, k=args.k)
        
        logits = args.alpha * desattr_logits + (1 - args.alpha) * distattr_logits
        return logits

    # Initialize visual reprogramming
    visual_reprogram = PaddingVR(224, args.input_size).to(device)

    # Setup optimizer and scheduler
    optimizer = torch.optim.SGD(visual_reprogram.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    t_max = args.epoch * len(labeled_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    # Create save directory
    os.makedirs(save_path, exist_ok=True)

    # Training loop
    visual_reprogram.train()
    best_acc = 0.
    progress_bar = tqdm(total=args.epoch, desc='Training MixMatch', leave=True)

    for epoch in range(args.epoch):
        total_num = 0
        true_num = 0
        loss_sum = 0
        supervised_loss_sum = 0
        unsupervised_loss_sum = 0

        length = len(mixmatch_loader)
        
        for i, (labeled_batch, unlabeled_batch) in enumerate(mixmatch_loader):
            # Get labeled data
            labeled_x, labeled_y = labeled_batch
            labeled_x, labeled_y = labeled_x.to(device), labeled_y.to(device)
                
            # Get unlabeled data (both weak and strong are the same for MixMatch)
            unlabeled_x_weak, unlabeled_x_strong, _ = unlabeled_batch
            unlabeled_x = unlabeled_x_weak.to(device)  # Use weak augmentation
            
            optimizer.zero_grad()
            
            # Step 1: Label Guessing - Generate pseudo-labels for unlabeled data
            with torch.no_grad():
                # Average predictions over K augmentations
                avg_predictions = None
                for k in range(args.K):
                    unlabeled_logits = network(visual_reprogram(unlabeled_x))
                    predictions = F.softmax(unlabeled_logits, dim=1)
                    if avg_predictions is None:
                        avg_predictions = predictions
                    else:
                        avg_predictions += predictions
                
                avg_predictions /= args.K
                
                # Sharpen the averaged predictions
                pseudo_labels = sharpen(avg_predictions, T=args.temperature)
            
            # Step 2: MixUp - Apply MixUp to both labeled and unlabeled data
            # Convert labeled_y to one-hot
            num_classes = len(classes)
            labeled_y_onehot = torch.zeros(labeled_y.size(0), num_classes).to(device)
            labeled_y_onehot.scatter_(1, labeled_y.unsqueeze(1), 1)
            
            # Combine labeled and unlabeled data
            all_x = torch.cat([labeled_x, unlabeled_x], dim=0)
            all_y = torch.cat([labeled_y_onehot, pseudo_labels], dim=0)
            
            # Apply MixUp
            mixed_x, mixed_y = mixup(all_x, all_y, alpha=args.mixup_alpha)
            
            # Step 3: Forward pass and loss computation
            mixed_logits = network(visual_reprogram(mixed_x))
            
            # Split back into labeled and unlabeled portions
            labeled_size = labeled_x.size(0)
            mixed_labeled_logits = mixed_logits[:labeled_size]
            mixed_unlabeled_logits = mixed_logits[labeled_size:]
            
            mixed_labeled_y = mixed_y[:labeled_size]
            mixed_unlabeled_y = mixed_y[labeled_size:]
            
            # Compute losses
            supervised_loss = -torch.mean(torch.sum(mixed_labeled_y * F.log_softmax(mixed_labeled_logits, dim=1), dim=1))
            unsupervised_loss = -torch.mean(torch.sum(mixed_unlabeled_y * F.log_softmax(mixed_unlabeled_logits, dim=1), dim=1))
            
            # Total loss
            total_loss = supervised_loss + args.lambda_u * unsupervised_loss
            
            total_loss.backward()
            optimizer.step()

            # Clamp logit scale
            model_to_use = model.module if isinstance(model, torch.nn.DataParallel) else model
            model_to_use.logit_scale.data = torch.clamp(model_to_use.logit_scale.data, 0, 4.6052)
            
            # Compute accuracy on original labeled data (for monitoring)
            with torch.no_grad():
                original_labeled_logits = network(visual_reprogram(labeled_x))
                total_num += labeled_y.shape[0]
                true_num += torch.argmax(original_labeled_logits, 1).eq(labeled_y).float().sum().item()

            loss_sum += total_loss.item()
            supervised_loss_sum += supervised_loss.item()
            unsupervised_loss_sum += unsupervised_loss.item()
            
            scheduler.step()
            
        # Calculate epoch metrics
        train_acc = true_num / total_num if total_num > 0 else 0.0
        avg_loss = loss_sum / len(mixmatch_loader) if len(mixmatch_loader) > 0 else 0.0
        avg_sup_loss = supervised_loss_sum / len(mixmatch_loader) if len(mixmatch_loader) > 0 else 0.0
        avg_unsup_loss = unsupervised_loss_sum / len(mixmatch_loader) if len(mixmatch_loader) > 0 else 0.0
        
        progress_bar.set_postfix({
            'Epoch': epoch + 1, 
            'Train Acc': f'{train_acc:.4f}'
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
            'Best Acc': f'{best_acc:.4f}'
        }, refresh=False)

        # Save checkpoint
        state_dict = {
            "visual_prompt_dict": visual_reprogram.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "avg_loss": avg_loss,
            "avg_sup_loss": avg_sup_loss,
            "avg_unsup_loss": avg_unsup_loss
        }

        if test_acc > best_acc:
            best_acc = test_acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
            
        # Save periodic checkpoints
        if (epoch + 1) % 50 == 0:
            torch.save(state_dict, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))
            
        progress_bar.update(1)
        visual_reprogram.train()

    progress_bar.close()
    print(f"\nTraining completed!")
    print(f"Final best test accuracy: {best_acc:.4f}")
    print(f"Results saved to: {save_path}")