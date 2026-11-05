import argparse
from torch.nn import functional as F
from torch.amp import autocast
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, CenterCrop, RandomResizedCrop, RandomHorizontalFlip, Normalize, RandomAdjustSharpness, RandomRotation, Normalize
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cfg import *
from tools import *
from datasets import *
from methods.vp import PaddingVR

from methods.backbones import get_model
from methods.runtime import prepare_output

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=['caltech101' ,'dtd' ,'eurosat' ,'fgvc' ,'food101',
                                         'oxford_flowers' ,'oxford_pets','stanford_cars' ,'sun397' ,'ucf101', 'resisc45',
                                         'imagenet'], default='dtd')

    p.add_argument('--epoch', type=int, default=0)   # total epochs
    p.add_argument('--input_size', type=int, default=192) # 224*224 images with VR pattern frame size=16
    p.add_argument('--shot', type=int, default=16) # few-shot training set
    p.add_argument('--bs', type=int, default=128) # batch size
    p.add_argument('--pre_model', choices=['clip', 'siglip', 'siglip2'], default='clip')
    p.add_argument('--backbone', choices=['ViT-B/16', 'ViT-L/14'], default='ViT-B/16')
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--backbone-weights", type=str, default=None)
    p.add_argument("--model-name", type=str, default=None)
    args = p.parse_args()
    if args.output is not None:
        prepare_output(args.output, args)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)

    # Load pretrained CLIP
    model, processor = get_model(args, device)

    # Data augmentation
    # train_process = Compose([
    #     Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
    #     RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
    #     RandomHorizontalFlip(),
    #     Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
    #     ToTensor(),

    # ])

    # test_process = Compose([
    #     Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
    #     CenterCrop(args.input_size),
    #     Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
    #     ToTensor(),
    # ])

    if args.pre_model == 'clip':
        train_process = Compose([
            Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        train_process = Compose([
            Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            ToTensor(),
            Normalize(mean=processor.image_processor.image_mean, std=processor.image_processor.image_std)
    ])

    # test_process = Compose([
    #     Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
    #     CenterCrop(args.input_size),
    #     Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
    #     ToTensor(),
    #     Normalize(mean=processor.image_processor.image_mean, std=processor.image_processor.image_std) if args.pre_model == 'siglip' or args.pre_model == 'siglip2' else \
    #         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    # ])

    if args.pre_model == 'clip':
        test_process = Compose([
            Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(args.input_size),
            Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        def test_process(img):
            out = processor.image_processor(images=img, return_tensors="pt")
            return out["pixel_values"].squeeze(0)

    trainloader, testloader, classes = build_loader(args.dataset, DOWNSTREAM_PATH, train_process, test_process, batch_size=args.bs, shot=args.shot, seed=args.seed)

    # Loading Attributes and Text Embeddings
    txt_emb_classes = clip_classifier(classes, [basic_template], model, processor)

    # Repurposing CLIP for Downstream Classification
    def network(x):
        if args.pre_model == 'clip':
            x_emb = model.encode_image(x)
            x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
            exp = model.logit_scale.exp()
            logits = exp * x_emb @ txt_emb_classes
        elif args.pre_model == 'siglip' or args.pre_model == 'siglip2':
            x_emb = model.get_image_features(pixel_values=x)
            x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
            exp, bias = model.logit_scale.exp(), model.logit_bias
            logits = exp * x_emb @ txt_emb_classes + bias
        return logits

    total_num = 0
    true_num = 0

    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            fx = network(x)
        total_num += y.size(0)
        true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
    acc = true_num / total_num

    print("Zero Shot Accuracy: ", acc)
    if args.output is not None:
        (args.output / "evaluation.json").write_text(json.dumps({
            "dataset": args.dataset, "seed": args.seed,
            "accuracy": acc, "test_size": total_num,
        }, indent=2) + "\n")
