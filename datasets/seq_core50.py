import json
import os 
from typing import Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from datasets.utils.validation import ValidationDataset
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path

from glob import glob
import yaml
import cv2
from tqdm import tqdm


class Core50(Dataset):
    INPUT_SIZE = (128, 128)
    
    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, subset: float = 1.) -> None:
        
        self.not_aug_transform = transforms.Compose([
            transforms.Resize(Core50.INPUT_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()])
        self.root = root 
        self.train = train
        self.phase = 'train' if train else 'test'
        self.transform = transform
        self.target_transform = target_transform
        assert 0 < subset <= 1.
        self.subset = subset 

        #read data yml file
        with open(os.path.join(root, f'{self.phase}_data.yml'),'r') as readfile:
            data = yaml.load(readfile, Loader=yaml.FullLoader)

        #read class json file
        with open(os.path.join(root, 'labels.json'),'r') as readfile:
            self.classes = json.load(readfile)
        
        self.data = []
        for img_path in tqdm(data['image_paths']):
            img = cv2.imread(os.path.join(self.root, img_path))
            img = np.ascontiguousarray(img[:,:,::-1])
            self.data.append(img)
    
        self.data = np.stack(np.array(self.data))
        self.targets = np.array(data['labels'])

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        img = transforms.ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        

class MyCore50(Core50):

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        img = transforms.ToPILImage()(img)
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img
        
        

class SequentialCore50(ContinualDataset):

    NAME = 'seq-core50'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    DATASET_FOLDER = 'core50_128x128'

    TRANSFORM = transforms.Compose([
        transforms.Resize(Core50.INPUT_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])

    def get_data_loaders(self):
        transforms = self.TRANSFORM
        test_transform = self.TRANSFORM

        train_dataset = MyCore50(base_path() + SequentialCore50.DATASET_FOLDER, train=True, transform=transforms)

        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, test_transform, self.NAME)
        else:
            test_dataset = Core50(base_path() + SequentialCore50.DATASET_FOLDER, train=False, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test
    
    @staticmethod
    def get_backbone() -> nn.Module:
        return resnet18(SequentialCore50.N_CLASSES_PER_TASK 
                            * SequentialCore50.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 10

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCore50.get_batch_size()