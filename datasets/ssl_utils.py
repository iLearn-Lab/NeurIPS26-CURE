import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
from . import build_dataset
import random
from copy import deepcopy

class SSLDataset(Dataset):
    """Semi-supervised learning dataset wrapper"""
    def __init__(self, data_source, transform_weak=None, transform_strong=None, is_labeled=True):
        self.data_source = data_source
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
        self.is_labeled = is_labeled
        
    def __len__(self):
        return len(self.data_source)
    
    def __getitem__(self, idx):
        item = self.data_source[idx]
        from .utils import read_image
        img = read_image(item.impath)
        label = item.label if hasattr(item, 'label') else item[1]
        
        if self.is_labeled:
            # For labeled data, only apply weak augmentation
            if self.transform_weak:
                img = self.transform_weak(img)
            return img, label
        else:
            # For unlabeled data, return both weak and strong augmentations
            img_weak = self.transform_weak(img) if self.transform_weak else img
            img_strong = self.transform_strong(img) if self.transform_strong else img
            return img_weak, img_strong, label  # Return index for consistency

def build_ssl_dataloaders(dataset_name, root_path, transform_train, transform_weak, transform_strong, transform_test,
                         batch_size=64, shot=16, mu=7, seed=0, num_workers=2):
    """
    Build semi-supervised data loaders for FixMatch
    
    Args:
        dataset_name: Name of the dataset
        root_path: Root path to dataset
        transform_weak: Weak augmentation transform
        transform_strong: Strong augmentation transform  
        transform_test: Test transform
        batch_size: Batch size for labeled data
        shot: Number of shots per class (for few-shot setting) - controls labeled data size
        mu: Number of unlabeled samples per labeled sample
        seed: Random seed
        num_workers: Data-loading processes; use 0 for synchronous loading
        
    Note:
        - shot: 控制每个类别的标注样本数量（few-shot learning的核心参数）
        - unlabeled_ratio: 控制无标注数据的batch size相对于标注数据的倍数
        - 在固定shot的情况下，标注数据量已确定，无需额外的labeled_ratio参数
    """
    # Build original dataset with specified shot (labeled samples per class)
    dataset = build_dataset(dataset_name, root_path, shot, seed)
    
    # Use the few-shot train_x as labeled data (already controlled by shot parameter)
    labeled_data = dataset.train_x
    
    # Create unlabeled data from train_u set (removing labels for SSL training)
    unlabeled_data_raw = dataset.train_u
    
    # Convert unlabeled data to remove label information for training
    unlabeled_data = []
    for item in unlabeled_data_raw:
        if hasattr(item, 'impath'):
            # Create a new Datum-like object without label for unlabeled training
            unlabeled_item = type('UnlabeledDatum', (), {
                'impath': item.impath, 
                'label': item.label,  # Placeholder label
                'domain': getattr(item, 'domain', 0),
                'classname': 'unlabeled'
            })()
            unlabeled_data.append(unlabeled_item)
        else:
            # Handle other formats if needed
            unlabeled_data.append(item)
    
    # Create datasets
    labeled_dataset = SSLDataset(labeled_data, transform_train, None, is_labeled=True)
    unlabeled_dataset = SSLDataset(unlabeled_data, transform_weak, transform_strong, is_labeled=False)
    test_dataset = SSLDataset(dataset.test, transform_test, None, is_labeled=True)
    
    # Create data loaders
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # unlabeled_loader = DataLoader(
    #     unlabeled_dataset,
    #     batch_size=batch_size * mu,
    #     shuffle=True,
    #     num_workers=2,
    #     pin_memory=True
    # )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size * mu,
        sampler=RandomSampler(unlabeled_dataset, replacement=True, num_samples=batch_size * mu * len(labeled_loader)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return labeled_loader, unlabeled_loader, test_loader, dataset.classnames

class FixMatchDataLoader:
    """
    Custom data loader for FixMatch that handles labeled and unlabeled data synchronization
    """
    def __init__(self, labeled_loader, unlabeled_loader):
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        # self.labeled_iter = iter(labeled_loader)
        # self.unlabeled_iter = iter(unlabeled_loader)

        self.max_len = len(labeled_loader)
        # self.max_len = max(len(self.labeled_loader), len(self.unlabeled_loader))
        self.step = 0
        
    def __iter__(self):
        # Reset iterators at the beginning of each epoch
        self.labeled_iter = iter(self.labeled_loader)
        self.unlabeled_iter = iter(self.unlabeled_loader)
        self.step = 0
        return self
    
    def __next__(self):
        if self.step >= self.max_len:
            raise StopIteration
        self.step += 1


        try:
            labeled_batch = next(self.labeled_iter)
        except StopIteration:
            self.labeled_iter = iter(self.labeled_loader)  # 重置 labeled_loader
            labeled_batch = next(self.labeled_iter)         

        try:
            unlabeled_batch = next(self.unlabeled_iter)
        except StopIteration:
            self.unlabeled_iter = iter(self.unlabeled_loader)
            unlabeled_batch = next(self.unlabeled_iter)
        
        return labeled_batch, unlabeled_batch
    
    def __len__(self):
        return self.max_len
