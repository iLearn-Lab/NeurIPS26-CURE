import argparse
from torch.nn import functional as F
from torch.amp import autocast
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, CenterCrop, RandomResizedCrop, RandomHorizontalFlip, Normalize, RandomAdjustSharpness, RandomRotation
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.runtime import prepare_output
from cfg import *
from tools import *
from datasets import *
from methods.vp import PaddingVR

from methods.backbones import get_model

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=['caltech101' ,'dtd' ,'eurosat' ,'fgvc' ,'food101',
                                         'oxford_flowers' ,'oxford_pets','stanford_cars' ,'sun397' ,'ucf101', 'resisc45',
                                         'imagenet'], default='dtd')
    p.add_argument('--alpha', type=float, default=0.5) # alpha: the balance hyperparameter lambda
    p.add_argument('--num_attr', type=int, default=20) # number of attributes
    p.add_argument('--k', type=int, default=3) # k for knn attribute selection
    p.add_argument('--epoch', type=int, default=200)   # total epochs
    p.add_argument('--lr', type=float, default=40) # the initial learning rate
    p.add_argument('--input_size', type=int, default=192) # 224*224 images with VR pattern frame size=16
    p.add_argument('--shot', type=int, default=16) # few-shot training set
    p.add_argument('--bs', type=int, default=128) # batch size
    p.add_argument('--backbone', choices=['ViT-B/16', 'ViT-L/14'], default='ViT-B/16')
    p.add_argument('--pre_model', type=str, default='clip')

    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--backbone-weights", type=str, default=None)
    p.add_argument("--model-name", type=str, default=None)
    args = p.parse_args()

    print(args)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    exp = "results/fs_attrvr"
    save_path = os.path.join(exp, str(args.pre_model) + str(args.shot) + args.dataset  + 'k' + str(args.k) + 'a' + str(args.alpha) + 's' + str(args.seed))
    if args.output is not None:
        save_path = str(args.output.resolve())
    save_path = prepare_output(save_path, args)

    model, processor = get_model(args, device)

    # Data augmentation
    if args.pre_model == 'clip':
        train_process = Compose([
            Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            ToTensor(),
            # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        train_process = Compose([
            Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),

        # Resize((224, 224)),
        # RandomRotation(90),
        # RandomAdjustSharpness(2),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # Normalize(mean=processor.image_processor.image_mean, std=processor.image_processor.image_std)
    ])

    test_process = Compose([
        Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(args.input_size),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # Normalize(mean=processor.image_processor.image_mean, std=processor.image_processor.image_std) if args.pre_model == 'siglip' or args.pre_model == 'siglip2' else \
        #     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    trainloader, testloader, classes = build_loader(args.dataset, DOWNSTREAM_PATH, train_process, test_process, batch_size=args.bs, shot=args.shot, seed=args.seed)

    # Loading Attributes and Text Embeddings
    txt_emb_desattr = clip_attr_classifier(classes, model, 'attributes/gpt3/' + args.dataset + '_des.json', num_attr=args.num_attr, processor=processor)
    txt_emb_distattr = clip_attr_classifier(classes, model, 'attributes/gpt3/' + args.dataset + '_dist.json', num_attr=args.num_attr, processor=processor)

    txt_emb_desattr = txt_emb_desattr / txt_emb_desattr.norm(dim=-1, keepdim=True)
    txt_emb_distattr = txt_emb_distattr / txt_emb_distattr.norm(dim=-1, keepdim=True)

    # Visual Reprogramming
    visual_reprogram = PaddingVR(224, args.input_size, model=args.pre_model).to(device)
    def network(x):
        if args.pre_model == 'clip':
            x_emb = model.encode_image(x)
            x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
            exp = model.logit_scale.exp()
            desattr_logits = getLogitsCLIP(args.num_attr, exp, x_emb, txt_emb_desattr, k=args.k)
            distattr_logits = getLogitsCLIP(args.num_attr, exp, x_emb, txt_emb_distattr, k=args.k)
        else:
            x_emb = model.get_image_features(pixel_values=x)
            x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
            exp, bias = model.logit_scale.to(device).exp(), model.logit_bias.to(device)
            desattr_logits = getLogitsSigLIP(args.num_attr, exp, bias, x_emb, txt_emb_desattr, k=args.k)
            distattr_logits = getLogitsSigLIP(args.num_attr, exp, bias, x_emb, txt_emb_distattr, k=args.k)
        return desattr_logits * args.alpha + distattr_logits * (1 - args.alpha)
        
    # Optimizer
    optimizer = torch.optim.SGD(visual_reprogram.parameters(), lr=args.lr, momentum=0.9)
    t_max = args.epoch * len(trainloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    # Make Dir
    os.makedirs(save_path, exist_ok=True)

    # Train
    visual_reprogram.train()
    best_acc = 0.
    progress_bar = tqdm(total=args.epoch, desc='Training', leave=True)

    for epoch in range(args.epoch):
        total_num = 0
        true_num = 0
        loss_sum = 0
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                fx = network(visual_reprogram(x))
                if args.pre_model == 'clip':
                    loss = F.cross_entropy(fx, y, reduction='mean')
                elif args.pre_model == 'siglip' or args.pre_model == 'siglip2':
                    # loss = F.cross_entropy(fx, y, reduction='mean')
                    y_onehot = torch.zeros_like(fx).scatter_(1, y.view(-1, 1), 1.0)
                    m1 = -torch.ones_like(fx) + 2 * y_onehot
                    loglik = torch.nn.functional.logsigmoid(m1 * fx)
                    nll = -torch.sum(loglik, dim=-1)
                    loss = nll.mean()
            loss.backward()
            optimizer.step()

            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            scheduler.step()
        acc = true_num / total_num
        progress_bar.set_postfix({'Epoch': epoch + 1, 'Train Loss': loss_sum / total_num, 'Train Acc': acc})

        # Test
        visual_reprogram.eval()
        total_num = 0
        true_num = 0
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx = network(visual_reprogram(x))
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
