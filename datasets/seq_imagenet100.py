import os
from typing import Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18

from torch.utils.data import Dataset

from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path

from glob import glob
import yaml
import cv2
from tqdm import tqdm

IMAGENET_CLASS_PATH ={
    'primary' : 'imagenet100_primary.yml',
    'aux' : 'imagenet100_aux.yml'
}

class  Imagenet100(Dataset):
    
    INPUT_SIZE = (256, 256)

    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                target_transform: Optional[nn.Module] = None, subset:float = 1., primary:bool = True, in_memory=False) -> None:
         
        self.not_aug_transform = transforms.Compose([
            transforms.Resize(Imagenet100.INPUT_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()])
        self.root = root 
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        assert 0 < subset <= 1.
        self.subset = subset if train else 1.
        self.in_memory = in_memory

        dataset_class_file = IMAGENET_CLASS_PATH['primary'] if primary else IMAGENET_CLASS_PATH['aux']

        with open(dataset_class_file,'r') as readfile:
            self.all_classes = yaml.load(readfile, Loader=yaml.FullLoader)
        
        phase_str = 'train' if train else 'val'           
        
        self.data_path = []
        self.targets = []
        for cls_name in self.all_classes:
            img_list = glob(os.path.join(root, 'images', phase_str, cls_name, '*.JPEG'))
            if self.subset < 1. :
                img_list = img_list[:int(len(img_list)*self.subset)]
            self.data_path.extend(img_list)
            self.targets.extend([self.all_classes[cls_name]] * len(img_list))
        
        if self.in_memory:
            self.data = []
            print('loading dataset...')
            for img_path in tqdm(self.data_path):
                img = cv2.resize(cv2.imread(img_path), (Imagenet100.INPUT_SIZE))
                img = np.ascontiguousarray(img[:, :, ::-1])
                self.data.append(img)

            self.data = np.stack(np.array(self.data))
        else:
            self.data = np.array(self.data_path)

        self.targets = np.array(self.targets)
        assert(len(self.data) > 0)
        classes = {v:k for k, v in self.all_classes.items()}
        self.classes = []
        for i in range(len(classes)):
            self.classes.append(classes[i])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.in_memory:
            img, target = self.data[index], self.targets[index]
        else:
            img_path, target = self.data[index], self.targets[index]
            #img_path is only the path where img is located
            img = cv2.imread(img_path)
            img = np.ascontiguousarray(img[:, :, ::-1])

        img = transforms.ToPILImage()(img)
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]
        
        return img, target 
    

class MyImagenet100(Imagenet100):
   
    def __getitem__(self, index):
        if self.in_memory:
            img, target = self.data[index], self.targets[index]
        else:
            img_path, target = self.data[index], self.targets[index]
            #img is only the path
            img = cv2.imread(img_path)
            img = np.ascontiguousarray(img[:, :, ::-1])
        
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


class SequentialImagenet100(ContinualDataset):

    NAME = 'seq-img100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 20
    TRANSFORM = transforms.Compose(
        [transforms.Resize(Imagenet100.INPUT_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    
    def get_data_loaders(self, in_memory:bool=False):
        transform = self.TRANSFORM

        test_transform = self.TRANSFORM
        

        train_dataset = MyImagenet100(self.args.dataset_path + 'ImageNet100',
                                      train = True, transform=transform, subset=self.args.dataset_subset, in_memory=in_memory)
                            
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, test_transform, self.NAME)
        else:
            test_dataset = Imagenet100(self.args.dataset_path + 'ImageNet100',
                                      train = False, transform=test_transform, in_memory=in_memory)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_backbone() -> nn.Module:
        return resnet18(SequentialImagenet100.N_CLASSES_PER_TASK 
                            * SequentialImagenet100.N_TASKS)

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
        return 1

    @staticmethod
    def get_batch_size():
        return 8

    @staticmethod
    def get_minibatch_size():
        return SequentialImagenet100.get_batch_size()