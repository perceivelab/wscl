# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from typing import Tuple

import numpy as np
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, Dataset
import copy

from torch.optim import SGD

class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME: str
    SETTING: str
    N_CLASSES_PER_TASK: int
    N_TASKS: int
    

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args
        self.val_size = self.args.val_dataset_size
        self.freezing_buff_size = self.args.freezing_buff_size

        if not all((self.NAME, self.SETTING, self.N_CLASSES_PER_TASK, self.N_TASKS)):
            raise NotImplementedError('The dataset must be initialized with all the required fields.')

    def get_data_loaders(self, in_memory:bool=False) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        raise NotImplementedError

    @staticmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_transform() -> nn.Module:
        """
        Returns the transform to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_loss() -> nn.Module:
        """
        Returns the loss to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_normalization_transform() -> nn.Module:
        """
        Returns the transform used for normalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_denormalization_transform() -> nn.Module:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_epochs():
        raise NotImplementedError

    @staticmethod
    def get_batch_size():
        raise NotImplementedError

    @staticmethod
    def get_minibatch_size():
        raise NotImplementedError
    
    @staticmethod
    def get_optimizer(parameters, args):
        return SGD(parameters, lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)


def store_masked_loaders(train_dataset: Dataset, test_dataset: Dataset,
                         setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
                                np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
                               np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader


def get_previous_train_loader(train_dataset: Dataset, batch_size: int,
                              setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
                                setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
                                < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def store_previous_masked_loaders(train_dataset: Dataset, test_dataset: Dataset,
                                  setting: ContinualDataset, validation_dataset: Dataset=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :param validation_dataset: val dataset 
    :return: train and test loaders
    """
    if setting.i > 0: # for the first task we want the whole dataset
        labels_to_remove = list(setting.label_to_pos.values())[:setting.i]
        #train_mask: samples of the current task + future classes
        mask_list = [np.array(train_dataset.targets) != label for label in labels_to_remove]

        if validation_dataset is not None:
            current_labels = list(setting.label_to_pos.values())[setting.i:setting.i+setting.N_CLASSES_PER_TASK]
            current_task_mask_list = [np.array(train_dataset.targets) != label for label in current_labels]                 

            for mask in current_task_mask_list:                 #mask sarebbe la maschera degli indici delle imgs con label = 2 (esempio)
                indices = np.where(mask == False)[0]            # indici delle immagini che NON sono della classe 2 
                mask_filter = indices[(setting.val_size+setting.freezing_buff_size)//setting.N_CLASSES_PER_TASK]
                mask[mask_filter:] = True                          # dal valore mask_filter in poi metti tutto a true
            
            past_labels_train_mask = np.stack(mask_list, axis=1)
            past_labels_train_mask = np.all(past_labels_train_mask, axis=1)

            current_labels_train_mask = np.stack(current_task_mask_list, axis=1)
            current_labels_train_mask = np.all(current_labels_train_mask, axis=1)

            train_mask = np.logical_and(past_labels_train_mask, current_labels_train_mask)

        else:
            train_mask = np.stack(mask_list, axis=1)
            train_mask = np.all(train_mask, axis=1)
        
        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    
    elif validation_dataset is not None:                                                #in questo pezzo di codice togliamo le prime 500 img per classe dal training set (verranno usate per il validation set)
        labels = list(setting.label_to_pos.values())[:setting.N_CLASSES_PER_TASK]

        mask_list = [np.array(train_dataset.targets) != label for label in labels]
        for mask in mask_list:
            indices = np.where(mask == False)[0]                                        #indici delle immagini che NON sono della classe 1 (esempio)
            mask_filter = indices[setting.val_size//setting.N_CLASSES_PER_TASK]
            mask[mask_filter:] = True                                                       

        train_mask = np.stack(mask_list, axis=1)
        train_mask = np.all(train_mask, axis=1)
        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    
    #test_mask: only samples of the current task
    next_task_labels = list(setting.label_to_pos.values())[setting.i : setting.i+setting.N_CLASSES_PER_TASK]
    test_mask = np.isin(test_dataset.targets, next_task_labels)

    test_dataset.data = test_dataset.data[test_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]


    val_loader = None
    buff_loader = None

    buffer_dataset = copy.deepcopy(validation_dataset)

    if validation_dataset is not None:
        #create the buffer loader                                                                    # le prime n immagini con label = 2 (esempio) vengono messe nel buffer
        mask_list = [np.array(validation_dataset.targets) == label for label in next_task_labels]    # mask_list contains a mask for each class of the next task
        for mask in mask_list:
            indices = np.where(mask == True)[0]                                                      # indices of images with a specific label (e.g. 2)
            mask_filter = indices[setting.freezing_buff_size//setting.N_CLASSES_PER_TASK]            # index of the (buff_size//class_per_task)-th image with a specific label 
            mask[mask_filter:] = False                                                               # keep in the buffer only the first 'buff_size//class_per_task' images with that label 
    
        buff_mask = np.any(np.stack(mask_list, axis=1), axis=1)                                         
        
        buffer_dataset.data = buffer_dataset.data[buff_mask]
        buffer_dataset.targets = np.array(buffer_dataset.targets)[buff_mask]
        buff_loader = DataLoader(buffer_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
        setting.buff_loader = buff_loader
        
        if setting.i > 0:                             #le prime 'buff_size//class_per_task' images vengono scartate, dopodichè le successive 'val_size//class_per_task' vengono usate per il validation set 
            #create the val_loader
            mask_list = [np.array(validation_dataset.targets) == label for label in next_task_labels]
            for mask in mask_list:
                indices = np.where(mask == True)[0]
                per_class_buff_size = setting.freezing_buff_size//setting.N_CLASSES_PER_TASK
                end_mask_filter = indices[per_class_buff_size]
                per_class_val_size = setting.val_size//setting.N_CLASSES_PER_TASK
                start_mask_filter = indices[per_class_buff_size+ per_class_val_size + 1]
                mask[:end_mask_filter+1] = False
                mask[start_mask_filter:] = False
        
            val_mask = np.any(np.stack(mask_list, axis=1), axis=1)
            
            validation_dataset.data = validation_dataset.data[val_mask]
            validation_dataset.targets = np.array(validation_dataset.targets)[val_mask]
            val_loader = DataLoader(validation_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=4)

    train_loader = DataLoader(train_dataset, batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK

    return train_loader, val_loader, buff_loader, test_loader



##### backup #####
# def store_previous_masked_loaders(train_dataset: Dataset, test_dataset: Dataset,
#                                   setting: ContinualDataset, validation_dataset: Dataset=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Divides the dataset into tasks.
#     :param train_dataset: train dataset
#     :param test_dataset: test dataset
#     :param setting: continual learning setting
#     :param validation_dataset: val dataset 
#     :return: train and test loaders
#     """
#     if setting.i > 0: # for the first task we want the whole dataset
#         labels_to_remove = list(setting.label_to_pos.values())[:setting.i]
#         #train_mask: samples of the current task + future classes
#         mask_list = [np.array(train_dataset.targets) != label for label in labels_to_remove]

#         if validation_dataset is not None:
#             current_labels = list(setting.label_to_pos.values())[setting.i:setting.i+setting.N_CLASSES_PER_TASK]
#             current_task_mask_list = [np.array(train_dataset.targets) != label for label in current_labels]

#             for mask in current_task_mask_list:
#                 m, = np.where(mask == False) 
#                 mask_filter= m[(setting.val_size+setting.freezing_buff_size)//setting.N_CLASSES_PER_TASK]
#                 mask[mask_filter:] = True
            
#             past_labels_train_mask = np.stack(mask_list, axis=1)
#             past_labels_train_mask = np.all(past_labels_train_mask, axis=1)

#             current_labels_train_mask = np.stack(current_task_mask_list, axis=1)
#             current_labels_train_mask = np.all(current_labels_train_mask, axis=1)

#             train_mask = np.logical_and(past_labels_train_mask, current_labels_train_mask)

#         else:
#             train_mask = np.stack(mask_list, axis=1)
#             train_mask = np.all(train_mask, axis=1)
        
#         train_dataset.data = train_dataset.data[train_mask]
#         train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    
#     elif validation_dataset is not None:
#         labels = list(setting.label_to_pos.values())[:setting.N_CLASSES_PER_TASK]

#         mask_list = [np.array(train_dataset.targets) != label for label in labels]
#         for mask in mask_list:
#             m, = np.where(mask == False) 
#             mask_filter= m[int(np.ceil(setting.args.buffer_size/setting.N_CLASSES_PER_TASK))]
#             mask[mask_filter:] = True

#         train_mask = np.stack(mask_list, axis=1)
#         train_mask = np.all(train_mask, axis=1)
#         train_dataset.data = train_dataset.data[train_mask]
#         train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    
#     #test_mask: only samples of the current task
#     next_task_labels = list(setting.label_to_pos.values())[setting.i : setting.i+setting.N_CLASSES_PER_TASK]
#     mask_list = [np.array(test_dataset.targets) == label for label in next_task_labels]
#     test_mask = np.stack(mask_list, axis=1)
#     test_mask = np.any(test_mask, axis=1)
#     test_dataset.data = test_dataset.data[test_mask]
#     test_dataset.targets = np.array(test_dataset.targets)[test_mask]


#     val_loader = None
#     buff_loader = None

#     buffer_dataset = copy.deepcopy(validation_dataset)

#     if validation_dataset is not None:
        
        
#         mask_list = [np.array(validation_dataset.targets) == label for label in next_task_labels]
#         for mask in mask_list:
#             m, = np.where(mask == True) 
#             mask_filter= m[int(np.ceil(setting.freezing_buff_size/setting.N_CLASSES_PER_TASK))]
#             mask[mask_filter:] = False
    
#         buff_mask = np.stack(mask_list, axis=1)
#         buff_mask = np.any(buff_mask, axis=1)
        
#         buffer_dataset.data = buffer_dataset.data[buff_mask]
#         buffer_dataset.targets = np.array(buffer_dataset.targets)[buff_mask]
#         buff_loader = DataLoader(buffer_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=1)
#         setting.buff_loader = buff_loader
        
#         if setting.i > 0:
#             mask_list = [np.array(validation_dataset.targets) == label for label in next_task_labels]
#             for mask in mask_list:
#                 m, = np.where(mask == True) 
#                 end_mask_filter = m[int(np.ceil(setting.freezing_buff_size/setting.N_CLASSES_PER_TASK))]
#                 start_mask_filter= m[(setting.val_size//setting.N_CLASSES_PER_TASK)+int(np.ceil(setting.freezing_buff_size/setting.N_CLASSES_PER_TASK))+1]
#                 mask[:end_mask_filter+1] = False
#                 mask[start_mask_filter:] = False
        
#             val_mask = np.stack(mask_list, axis=1)
#             val_mask = np.any(val_mask, axis=1)
            
#             validation_dataset.data = validation_dataset.data[val_mask]
#             validation_dataset.targets = np.array(validation_dataset.targets)[val_mask]
#             val_loader = DataLoader(validation_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=1)
#             setting.buff_loader = buff_loader

#     train_loader = DataLoader(train_dataset, batch_size=setting.args.batch_size, shuffle=True, num_workers=1)
#     test_loader = DataLoader(test_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=1)
#     setting.test_loaders.append(test_loader)
#     setting.train_loader = train_loader

#     setting.i += setting.N_CLASSES_PER_TASK

#     return train_loader, val_loader, buff_loader, test_loader