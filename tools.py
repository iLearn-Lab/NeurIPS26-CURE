import sys
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Add current directory to path to import local datasets module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import from local datasets module
import datasets as local_datasets
build_dataset = local_datasets.build_dataset
build_data_loader = local_datasets.build_data_loader
import torch
import random
import numpy as np
import clip
import json
from pathlib import Path

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def clip_classifier(classnames, template, model, processor=None):
    '''
    Text encoder for label-based classification

    params:
    classnames: class name of the label space
    template: text prompts
    model: the pretrained model
    processor: siglip model processor
    '''

    device = next(model.parameters()).device
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            if processor is not None:
                texts = processor(text=texts, padding="max_length", return_tensors="pt").to(device)
                class_embeddings = model.get_text_features(**texts)
            else:
                texts = clip.tokenize(texts).to(device)
                class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights

def clip_attr_classifier(classnames, model, dir, num_attr, processor=None):
    '''
    Text encoder for attribute-based classification.

    params:
    classnames: class name of the label space
    model: the pretrained model
    dir: attribute path
    num_attr: 'm' in the paper, the attribute number
    processor: siglip model processor
    '''

    attribute_file = Path(dir)
    if not attribute_file.is_absolute():
        attribute_file = Path(__file__).resolve().parent / attribute_file
    with attribute_file.open(encoding="utf-8") as stream:
        data = json.load(stream)
    device = next(model.parameters()).device
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize
            filtered_sentences = [sentence for sentence in data[classname] if len(sentence) >= 20]
            if len(filtered_sentences) == 0:
                print(classname)
                raise ValueError("No valid attributes have been generated")
            if num_attr < len(filtered_sentences):
                texts = random.sample(filtered_sentences, num_attr)
            else:
                texts = random.choices(filtered_sentences, k=num_attr)
            
            if processor is not None:
                texts = processor(text=texts, padding="max_length", return_tensors="pt").to(device)
                class_embeddings = model.get_text_features(**texts)
            else:
                texts = clip.tokenize(texts).to(device)
                class_embeddings = model.encode_text(texts)

            # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            clip_weights.append(class_embeddings)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights

def getLogitsCLIP(num_attr, exp, x_emb, t_emb, k=3):
    '''
    Return the similarity logits output

    params:
    num_attr: 'm' in the paper, the attribute number
    exp: CLIP logit_scale
    x_emb: image embeddings
    t_emb: text embeddings
    '''
    t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
    fea_dim = t_emb.shape[-1]
    t_emb = t_emb.permute(2, 1, 0).reshape(fea_dim, -1)
    res = exp * x_emb @ t_emb
    bs = res.shape[0]
    res = res.reshape(bs, -1, num_attr)
    res, _ = torch.sort(res, dim=-1, descending=True)
    logits = torch.mean(res[:, :, :k], dim=2)
    return logits

def getLogitsSigLIP(num_attr, exp, bias, x_emb, t_emb, k=3):
    '''
    Return the similarity logits output

    params:
    num_attr: 'm' in the paper, the attribute number
    exp: SigLIP logit_scale
    bias: SigLIP bias 
    x_emb: image embeddings
    t_emb: text embeddings
    '''
    t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
    fea_dim = t_emb.shape[-1]
    t_emb = t_emb.permute(2, 1, 0).reshape(fea_dim, -1)
    res = exp * x_emb @ t_emb + bias
    bs = res.shape[0]
    res = res.reshape(bs, -1, num_attr)
    res, _ = torch.sort(res, dim=-1, descending=True)
    logits = torch.mean(res[:, :, :k], dim=2)
    return logits

def build_loader(dataset_name, root_path, train_preprocess=None, test_preprocess=None, batch_size=64, shot=16, seed=0):
    '''
    Retuen the loader of downstream tasks.

    params:
    dataset_name: downstream dataset name
    root_path: the path of dataset
    train_preprocess/test_preprocess: the data argumentation performed on samples
    batch_size: training batch size
    shot: the available number of samples per class
    seed: the random seed
    '''
    dataset = build_dataset(dataset_name, root_path, shot, seed)
    train_loader = build_data_loader(data_source=dataset.train_x, batch_size=batch_size, is_train=True, tfm=train_preprocess, shuffle=True)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=batch_size, is_train=False, tfm=test_preprocess, shuffle=False)
    return train_loader, test_loader, dataset.classnames
