from argparse import Namespace
from typing import Optional, Tuple
import torch.nn as nn

import torchvision.transforms as transforms
from tqdm import tqdm
from datasets.seq_core50 import MyCore50, SequentialCore50
from datasets.seq_imagenet100 import MyImagenet100
from datasets.seq_halftinyimagenet import HalfTinyImagenet

from datasets.utils.continual_dataset import store_previous_masked_loaders
from datasets.utils.validation import get_train_val

import numpy as np
import copy 
import os
from torchvision.datasets import CIFAR10, CIFAR100
from skimage.transform import resize as image_resize
from PIL import Image


class DreamDataset(MyCore50):

    DREAM_CLASS_LABELS = []
    DREAM_CLASS_NAMES = []

    def __init__(self, root_1: str, root_2: str, train: bool = True, transform: Optional[nn.Module] = None, target_transform: Optional[nn.Module] = None,
                    subset: float = 1, labels = list(range(SequentialCore50.N_CLASSES_PER_TASK)) + [None] * SequentialCore50.N_CLASSES_PER_TASK * (SequentialCore50.N_TASKS-1)) -> None:
            
        super().__init__(root_1, train, transform, target_transform, subset)
        
        #aux_data
        d2_name = os.path.split(root_2)[-1]
        if d2_name == 'AUXImageNet100':
            dataset_2 = MyImagenet100(root_2, train, transform, target_transform, subset, primary=False)
        elif d2_name == 'CIFAR10':
            dataset_2 = CIFAR10(root_2, train, transform, target_transform, download=True)
        elif d2_name == 'CIFAR100':
            dataset_2 = CIFAR100(root_2, train, transform, target_transform, download=True)
        elif d2_name == 'TINYIMG':
            dataset_2 = HalfTinyImagenet(root_2, train, transform, target_transform, download=False, firstHalf=False)
        else:
            raise ValueError(f'Dataset: {d2_name} not supported')

        num_classes_1 = len(self.classes)
        num_classes_2 = len(dataset_2.classes)

        assert num_classes_2 >= num_classes_1, "dataset2 num classes too small"

        if len(DreamDataset.DREAM_CLASS_LABELS) == 0:
            if num_classes_2 > num_classes_1:
                DreamDataset.DREAM_CLASS_LABELS = np.random.choice(num_classes_2, len(labels), replace=False)
            else:
                DreamDataset.DREAM_CLASS_LABELS = np.arange(num_classes_2)
            #save dream_classes names
            DreamDataset.DREAM_CLASS_NAMES = [dataset_2.classes[l] for l in DreamDataset.DREAM_CLASS_LABELS]

        #Preparing data from main dataset
        index_mask = [t in labels for t in self.targets]
        data_1 = self.data[index_mask]
        targets_1 = np.array(self.targets)[index_mask]

        #store original_indexes
        swap_dict = {}
        for l in labels:
            if l is not None and l != labels.index(l):
                swap_dict[l] = [targets_1 == l]
        #update original_indexes with used_indexes
        for k, v in swap_dict.items():
            targets_1[tuple(v)] = labels.index(k)

        #save new classes list
        classes_1 = [self.classes[l] if l is not None else None for l in labels]

        #fill dataset with dream data
        dream_labels = [DreamDataset.DREAM_CLASS_LABELS[i] if labels[i] == None else None for i in range(len(labels))]
        index_mask = [t in dream_labels for t in dataset_2.targets]
        data_2 = dataset_2.data[index_mask]
        targets_2 = np.array(dataset_2.targets)[index_mask]

        swap_dict_2 = {}
        for l in dream_labels:
            if l is not None: 
                swap_dict_2[l] = [targets_2 == l]
        for k, v in swap_dict_2.items():
            targets_2[tuple(v)] = dream_labels.index(k)

        if len(targets_2) > 0:
            if not data_1.shape[:-1] == data_2.shape[:-1]:
                assert data_1.shape[-1] == data_2.shape[-1]
                dsize = data_1.shape[1:-1]
                tmp_data = []
                print("resizing dream images")
                for img in tqdm(data_2):
                    rszd_img = image_resize(img, dsize)
                    tmp_data.append(np.rint(rszd_img * 255).astype(np.uint8))
                data_2 = np.array(tmp_data)

            self.data = np.concatenate([data_1, data_2]) 
            self.targets = np.concatenate([targets_1, targets_2])
        else:
            self.data = data_1
            self.targets = targets_1
        assert len(self.data) == len(self.targets)
        self.classes = [dataset_2.classes[x] if x is not None else classes_1[i] for i, x in enumerate(dream_labels)]
        self.class_to_idx = {x: i for i, x in enumerate(self.classes)}




class DreamImagenet(SequentialCore50):
    NAME = 'dream-seq-core50'

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

        
        #list of lists. keeps track of the original class subdivision according to the task
        self.classes_per_task = [[j for j in range(i, i+self.N_CLASSES_PER_TASK)] for i in range(0, self.N_CLASSES_PER_TASK * self.N_TASKS, self.N_CLASSES_PER_TASK)]

        '''
        dict contaninig the mapping between the label used by the model and the original class label
        key: new_label
        value: original_label
        '''
        self.pos_to_label = {k: None for k in range(self.N_TASKS * self.N_CLASSES_PER_TASK)}

        '''
        dict containing the mapping between the original class label and the label by the model
        key: original_label
        value: new_label
        '''
        self.label_to_pos = {k: None for k in range(self.N_TASKS * self.N_CLASSES_PER_TASK)}

    def set_pos_new_tasks(self, pos:list) -> None:
        assert len(pos) == self.N_CLASSES_PER_TASK
        task_number = self.i // self.N_CLASSES_PER_TASK
        classes = self.classes_per_task[task_number]
        for p, c in zip(pos, classes):
            self.pos_to_label[p] = c
            self.label_to_pos[c] = p

    def get_free_pos(self) -> list:
        free_pos = [k for k, v in self.pos_to_label.items() if v is None]
        return free_pos
    
    def get_data_loaders(self):
        
        transform = self.TRANSFORM
        test_transform = self.TRANSFORM

        train_dataset = DreamDataset(self.args.dataset_path + SequentialCore50.DATASET_FOLDER, self.args.dataset_path + self.args.dataset_2, train=True, transform=transform, labels = list(self.pos_to_label.values()), subset = self.args.dataset_subset)
        test_dataset = DreamDataset(self.args.dataset_path + SequentialCore50.DATASET_FOLDER, self.args.dataset_path + self.args.dataset_2, train=False, transform=test_transform, labels = list(self.pos_to_label.values()))
        

        val = None
        if self.args.validation:
            val_dataset = copy.deepcopy(train_dataset)
            train, val, buff, test = store_previous_masked_loaders(train_dataset, test_dataset, self, val_dataset)
        else:
            train, val, buff, test = store_previous_masked_loaders(train_dataset, test_dataset, self)

        return train, val, buff, test
        
    
    def get_current_labels(self):
        
        t = self.i // self.N_CLASSES_PER_TASK
        return list(self.label_to_pos.values())[(t-1)*self.N_CLASSES_PER_TASK : t*self.N_CLASSES_PER_TASK]
        
    def get_task_labels(self, t:int):
        
        return list(self.label_to_pos.values())[t*self.N_CLASSES_PER_TASK : (t+1)*self.N_CLASSES_PER_TASK]
