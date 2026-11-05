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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=['caltech101' ,'dtd' ,'eurosat' ,'fgvc' ,'food101',
                                         'oxford_flowers' ,'oxford_pets','stanford_cars' ,'sun397' ,'ucf101', 'resisc45',
                                         'imagenet'], default='caltech101')
    # p.add_argument('--dataset', choices=['caltech101'])
    p.add_argument('--alpha', type=float, default=0.5) # alpha: the balance hyperparameter lambda
    p.add_argument('--num_attr', type=int, default=20) # number of attributes
    p.add_argument('--k', type=int, default=3) # k for knn attribute selection
    p.add_argument('--epoch', type=int, default=200)   # total epochs
    p.add_argument('--lr', type=float, default=40) # the initial learning rate
    p.add_argument('--input_size', type=int, default=192) # 224*224 images with VR pattern frame size=16
    p.add_argument('--shot', type=int, default=10) # few-shot training set
    p.add_argument('--bs', type=int, default=4)
    
    # FixMatch specific parameters
    p.add_argument('--confidence_threshold', type=float, default=0.95) # confidence threshold for pseudo-labels
    p.add_argument('--lambda_u', type=float, default=1.0) # weight for unlabeled loss
    p.add_argument('--unlabeled_ratio', type=int, default=7) # ratio of unlabeled to labeled batch size
    p.add_argument('--ema_decay', type=float, default=0.999) # EMA decay for model averaging
    
    p.add_argument('--tua', type=float, default=0.5) ## the temperature of sharpening the pseudo-label distribution
    
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--backbone-weights", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    exp = "results/fs_attrvr_refixmatch"
    save_path = os.path.join(exp, args.dataset + '_' + str(args.shot) + 'k' + str(args.k) + 'a' + str(args.alpha) + 
                            'tau' + str(args.confidence_threshold) + 'lu' + str(args.lambda_u) + 's' + str(args.seed))
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
        # RandAugment(n=2, m=10),  # RandAugment
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
    # if args.shot == 1: bs = 16
    # else: bs = 64
    
    labeled_loader, unlabeled_loader, testloader, classes = build_ssl_dataloaders(
        args.dataset, DOWNSTREAM_PATH, weak_process, weak_process, strong_process, test_process, 
        batch_size=args.bs, shot=args.shot, mu=args.unlabeled_ratio,
        seed=args.seed
    )
    
    # Create combined data loader for FixMatch
    fixmatch_loader = FixMatchDataLoader(labeled_loader, unlabeled_loader)

    # Loading Attributes and Text Embeddings
    # Handle DataParallel wrapper for text embedding generation
    model_for_text = model.module if isinstance(model, torch.nn.DataParallel) else model
    txt_emb_desattr = clip_attr_classifier(classes, model_for_text, 'attributes/gpt3/' + args.dataset + '_des.json', num_attr=args.num_attr)
    txt_emb_distattr = clip_attr_classifier(classes, model_for_text, 'attributes/gpt3/' + args.dataset + '_dist.json', num_attr=args.num_attr)

    # Repurposing CLIP for Downstream Classification
    def network(x):
        # Handle DataParallel wrapper
        model_to_use = model.module if isinstance(model, torch.nn.DataParallel) else model
        x_emb = model_to_use.encode_image(x)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        exp = model_to_use.logit_scale.exp()
        desattr_logits = getLogitsCLIP(args.num_attr, exp, x_emb, txt_emb_desattr, k=args.k)
        distattr_logits = getLogitsCLIP(args.num_attr, exp, x_emb, txt_emb_distattr, k=args.k)
        return desattr_logits * args.alpha + distattr_logits * (1 - args.alpha)

    # Visual Reprogramming
    visual_reprogram = PaddingVR(224, args.input_size).to(device)

    # Optimizer
    optimizer = torch.optim.SGD(visual_reprogram.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    t_max = args.epoch * len(labeled_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    # Make Dir
    os.makedirs(save_path, exist_ok=True)

    # Train with FixMatch
    visual_reprogram.train()
    best_acc = 0.
    progress_bar = tqdm(total=args.epoch, desc='Training FixMatch', leave=True)

    for epoch in range(args.epoch):
        total_num = 0
        true_num = 0
        loss_sum = 0
        supervised_loss_sum = 0
        unsupervised_loss_sum = 0
        mask_prob = 0

        length = len(fixmatch_loader)
        
        for i, (labeled_batch, unlabeled_batch) in enumerate(fixmatch_loader):
            # Unpack labeled data
            labeled_x, labeled_y = labeled_batch
            labeled_x, labeled_y = labeled_x.to(device), labeled_y.to(device)
                
            # Unpack unlabeled data
            unlabeled_x_weak, unlabeled_x_strong, _ = unlabeled_batch
            unlabeled_x_weak = unlabeled_x_weak.to(device)
            unlabeled_x_strong = unlabeled_x_strong.to(device)
            
            optimizer.zero_grad()
            
            labeled_logits = network(visual_reprogram(labeled_x))
            supervised_loss = F.cross_entropy(labeled_logits, labeled_y, reduction='mean')
            
            # Generate pseudo-labels for unlabeled data
            # 生成伪标签
            with torch.no_grad():  # 修正语法错误
                weak_logits = network(visual_reprogram(unlabeled_x_weak))
                weak_probs = F.softmax(weak_logits, dim=1)
                max_probs, pseudo_labels = torch.max(weak_probs, dim=1)
                mask = max_probs >= args.confidence_threshold

            # 强增强数据的前向传播
            strong_logits = network(visual_reprogram(unlabeled_x_strong))

            # 高置信度样本的伪标签损失
            if mask.sum() > 0:
                pseudo_loss = F.cross_entropy(
                    strong_logits[mask], pseudo_labels[mask], reduction='mean'
                )
            else:
                pseudo_loss = torch.tensor(0.0).to(device)

            # 低置信度样本的一致性损失
            if (~mask).sum() > 0:
                weak_logits_sharpen = sharpen(weak_logits[~mask], args.tua)
                kl_loss = F.kl_div(
                    F.log_softmax(strong_logits[~mask], dim=1),
                    F.softmax(weak_logits_sharpen, dim=1),
                    reduction='batchmean'
                )
            else:
                kl_loss = torch.tensor(0.0).to(device)

            # 总损失
            unsupervised_loss = pseudo_loss + kl_loss
            total_loss = supervised_loss + args.lambda_u * unsupervised_loss
            
            # Total loss
            total_loss = supervised_loss + args.lambda_u * unsupervised_loss
            
            total_loss.backward()
            optimizer.step()

            # Handle DataParallel wrapper
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
            
        # Calculate averages
        train_acc = true_num / total_num if total_num > 0 else 0.0
        avg_loss = loss_sum / len(fixmatch_loader) if len(fixmatch_loader) > 0 else 0.0
        avg_sup_loss = supervised_loss_sum / len(fixmatch_loader) if len(fixmatch_loader) > 0 else 0.0
        avg_unsup_loss = unsupervised_loss_sum / len(fixmatch_loader) if len(fixmatch_loader) > 0 else 0.0
        avg_mask_prob = mask_prob / len(fixmatch_loader) if len(fixmatch_loader) > 0 else 0.0
        
        progress_bar.set_postfix({
            'Epoch': epoch + 1, 
            'Train Acc': f'{train_acc:.4f}',
            'Mask Prob': f'{avg_mask_prob:.4f}'
        })

        # Test
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
            'Mask Prob': f'{avg_mask_prob:.4f}'
        }, refresh=False)

        # Save
        state_dict = {
            "visual_prompt_dict": visual_reprogram.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "avg_loss": avg_loss,
            "avg_sup_loss": avg_sup_loss,
            "avg_unsup_loss": avg_unsup_loss,
            "avg_mask_prob": avg_mask_prob
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