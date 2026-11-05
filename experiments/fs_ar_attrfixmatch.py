import argparse
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, CenterCrop, RandomResizedCrop, RandomHorizontalFlip
from torchvision.transforms import InterpolationMode, RandomErasing, ColorJitter, RandomGrayscale, RandomRotation
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.runtime import prepare_output
from cfg import *
from tools import *
from datasets import *
from methods.vp import PaddingVR
from datasets.ssl_utils import build_ssl_dataloaders, FixMatchDataLoader


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=['caltech101' ,'dtd' ,'eurosat' ,'fgvc' ,'food101',
                                         'oxford_flowers' ,'oxford_pets','stanford_cars' ,'sun397' ,'ucf101', 'resisc45',
                                         'imagenet'], default='dtd')
    p.add_argument('--epoch', type=int, default=200)   # total epochs
    p.add_argument('--alpha', type=float, default=0.5) # alpha: the balance hyperparameter lambda
    p.add_argument('--num_attr', type=int, default=20) # number of attributes
    p.add_argument('--confidence_threshold', type=float, default=0.95) # the confidence threshold for pseudo-labeling

    p.add_argument('--k', type=int, default=3) # the number of knn 
    p.add_argument('--lr', type=float, default=40) # the initial learning rate
    p.add_argument('--input_size', type=int, default=192) # 224*224 images with VR pattern frame size=16
    p.add_argument('--shot', type=int, default=16) # few-shot training set
    p.add_argument('--bs', type=int, default=10) # batch-size
    p.add_argument('--unlabeled_ratio', type=int, default=7) # the ratio of unlabel and label data.
    p.add_argument('--lambda_u', type=float, default=1) # the weight of unsupervised loss

    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--backbone-weights", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    exp = "results/fs_ar_attrfixmatch"
    save_path = os.path.join(exp, str(args.shot) + args.dataset + 's' + str(args.seed))
    if args.output is not None:
        save_path = str(args.output.resolve())
    save_path = prepare_output(save_path, args)

    # Load pretrained CLIP
    model, _ = clip.load(args.backbone_weights or "ViT-B/16", device=device)
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)
    model = model.to(device)

    # Data augmentation
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
        RandomErasing(p=1, scale=(0.02, 0.4), ratio=(1, 1), value=127),  # Cutout
    ])

    test_process = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(args.input_size),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
    ])

    # trainloader, testloader, classes = build_loader(args.dataset, DOWNSTREAM_PATH, train_process, test_process, batch_size=args.bs, shot=args.shot, seed=args.seed)
    labeled_loader, unlabeled_loader, testloader, classes = build_ssl_dataloaders(
        args.dataset, DOWNSTREAM_PATH, weak_process, weak_process, strong_process, test_process, 
        batch_size=args.bs, shot=args.shot, mu=args.unlabeled_ratio,
        seed=args.seed
    )

    fixmatch_dataloader = FixMatchDataLoader(
        labeled_loader, unlabeled_loader)


    # Loading Labels and Text Embeddings
    TEMPLATES = [basic_template]
    txt_emb = clip_classifier(classes, TEMPLATES, model)

    txt_emb_distattr = clip_attr_classifier(classes, model, 'attributes/gpt3/' + args.dataset + '_dist.json', num_attr=args.num_attr)

    # Repurposing CLIP for Downstream Classification
    def network(x):
        x_emb = model.encode_image(x)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * x_emb @ txt_emb
        disattr_logits = getLogitsCLIP(args.num_attr, model.logit_scale.exp(), x_emb, txt_emb_distattr, args.k)
        return logits, disattr_logits

    # Visual Reprogramming
    visual_reprogram = PaddingVR(224, args.input_size).to(device)

    # Optimizer
    optimizer = torch.optim.SGD(visual_reprogram.parameters(), lr=args.lr, momentum=0.9)
    t_max = args.epoch * len(fixmatch_dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    # Make Dir
    os.makedirs(save_path, exist_ok=True)

    # # Train
    # visual_reprogram.train()
    best_acc = 0.
    progress_bar = tqdm(total=args.epoch, desc='Training', leave=True)

    for epoch in range(args.epoch):
        total_num = 0
        true_num = 0
        loss_sum = 0
        pseudo_num = 0
        pseudo_total_num = 0

        visual_reprogram.train()
        for i, (labeled_batch, unlabeled_batch) in enumerate(fixmatch_dataloader):
            labeled_x, labeled_y = labeled_batch
            unlabeled_x_weak, unlabeled_x_strong, unlabeled_idx = unlabeled_batch

            labeled_x, labeled_y = labeled_x.to(device), labeled_y.to(device)
            unlabeled_x_weak, unlabeled_x_strong = unlabeled_x_weak.to(device), unlabeled_x_strong.to(device)

            optimizer.zero_grad()
            with autocast():
                labeled_probs,_ = network(visual_reprogram(labeled_x))
                supervised_loss = F.cross_entropy(labeled_probs, labeled_y, reduction='mean')

                unlabeled_probs_weak, unlabeled_disattr_probs_weak = network(visual_reprogram(unlabeled_x_weak))
                unlabeled_probs_strong, unlabeled_disattr_probs_strong = network(visual_reprogram(unlabeled_x_strong))
                
                mask_prob, pseudo_label  = unlabeled_disattr_probs_weak.softmax(dim=-1).max(dim=1)
                mask = mask_prob.ge(args.confidence_threshold)

                if mask.sum() > 0:
                    pseudo_label = pseudo_label[mask]
                    pseduo_unlabeled_probs_strong = unlabeled_probs_strong[mask]
                    unsupervised_loss = F.cross_entropy(pseduo_unlabeled_probs_strong, pseudo_label, reduction='mean')
                else:
                    unsupervised_loss = torch.tensor(0.0).to(device)
                    pseduo_unlabeled_probs_strong = unlabeled_probs_strong
                
                unsupervised_loss += F.kl_div(F.log_softmax(unlabeled_probs_weak, dim=1), F.softmax(unlabeled_disattr_probs_weak, dim=1), reduction='batchmean')
                unsupervised_loss += F.kl_div(F.log_softmax(unlabeled_probs_strong, dim=1), F.softmax(unlabeled_disattr_probs_strong, dim=1), reduction='batchmean')

            loss = supervised_loss + args.lambda_u * unsupervised_loss

            loss.backward()
            optimizer.step()

            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
            true_num += torch.argmax(labeled_probs, 1).eq(labeled_y).float().sum().item() 
            total_num += labeled_y.size(0)
            # print(f"Shape pseduo: {pseduo_unlabeled_probs_strong.shape}, label: {pseudo_label.shape}, labeled_shape:{labeled_x.shape}")
            pseudo_num += torch.argmax(pseduo_unlabeled_probs_strong, 1).eq(pseudo_label).float().sum().item()
            pseudo_total_num += unlabeled_x_strong.size(0)


            loss_sum += loss.item() * labeled_y.size(0)

            scheduler.step()
        acc = true_num / total_num
        progress_bar.set_postfix({'Epoch': epoch + 1, 'Train Label Acc': acc, 'Train Pseudo Acc': pseudo_num / pseudo_total_num, 'Loss': loss_sum / total_num})


        # Test
        visual_reprogram.eval()
        total_num = 0
        true_num = 0
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx,_ = network(visual_reprogram(x))
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        acc = true_num / total_num
        progress_bar.set_postfix({'Epoch': epoch + 1, 'Test Acc': acc, 'Best Acc': best_acc}, refresh=False)


        # Save
        state_dict = {
            "visual_prompt_dict": visual_reprogram.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }
        if acc > best_acc:
            best_acc = acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
        progress_bar.update(1)

    progress_bar.close()