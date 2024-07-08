# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
from datasets import get_dataset, get_forward_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.status import ProgressBar


from utils.layers_freezing import freeze_layers
import re
import statistics
from torch.utils.data import DataLoader
from utils.freezing_eval import Buffer_dataset
from utils.freezing_eval import model_eval
from copy import deepcopy
from utils.buffer import Buffer

try:
    import wandb
except ImportError:
    wandb = None

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """

    #if 'dream-' in dataset.NAME: 
    current_task_labels = dataset.get_task_labels(k)
    
    mask = torch.full_like(outputs, 1, dtype=bool)
    mask[:, current_task_labels] = False
    outputs[mask] = -float('inf')
    
    # else:
    #     outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    #     outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
    #                dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        assert set(test_loader.dataset.targets) == set(dataset.get_task_labels(k)), "Something wrong in test dataset creation."
        for data in test_loader:
            with torch.no_grad():
                inputs, labels, _ = data
                inputs, labels= inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes

def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)
    freezed_layers = 0
    
    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        args.wandb_url = wandb.run.get_url()

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            if 'dream-' in dataset_copy.NAME:
                dataset_copy.set_pos_new_tasks(tuple(range(t * dataset_copy.N_CLASSES_PER_TASK, (t+1)*dataset_copy.N_CLASSES_PER_TASK)))
                _, _, _, _ = dataset_copy.get_data_loaders()
            else:
                _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)

    buffers_list = []

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()

        '''
        If we are using DreamDATASET as 'dataset', we need to choose which heads assign to the new task.
        '''
        if 'dream-' in dataset.NAME:
            if t == 0:
                #create forward dataset
                forward_dataset = get_forward_dataset(args)
                _, _ = forward_dataset.get_data_loaders() 
                dataset.set_pos_new_tasks(tuple(range(0, dataset.N_CLASSES_PER_TASK)))
            else:
                # given samples of the new task, the most activated heads become the next heads 
                next_pos_classes = get_next_pos_classes(model, forward_dataset, dataset.get_free_pos())
                dataset.set_pos_new_tasks(next_pos_classes)

        if args.validation:
            train_loader, val_loader, buff_loader, _ = dataset.get_data_loaders()
            buffers_list.append(buff_loader)
        elif 'dream-' in dataset.NAME:
            train_loader, *_ = dataset.get_data_loaders()
        else:
            train_loader, _ = dataset.get_data_loaders()

        current_task_labels = []
        if 'dream-' in dataset.NAME:
            current_task_labels = dataset.get_current_labels()
            print(train_loader.dataset.class_to_idx)
            

        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)
        print(f"Task n: {t} involves {len(train_loader.dataset)} samples.")

        for epoch in range(model.args.n_epochs):
            if args.model == 'joint':
                continue
            #Freezing mechanims
            if args.freezing_eval is not None and t > 0 and epoch == 0:

                models, modules_names_to_freeze = [deepcopy(model)], []
                last_frozen_module = None
                for module_name, module in model.net.named_modules():
                    if re.match(re.compile('layer..'), module_name) is not None and not any(word in module_name for word in ["conv1", "bn1", "conv2", "bn2", "4", "shortcut"]):
                        for _, parameters in module.named_parameters():
                            if parameters.requires_grad != False:
                                modules_names_to_freeze.append(module_name.replace(".", "[") + "]")
                                break
                            else:
                                last_frozen_module = str(module_name.replace(".", "[") + "]")

                print(f"Last frozen layer: {last_frozen_module}")
                if last_frozen_module is not None:
                    freeze_layers(models[0].net, eval(f'models[0].net.{last_frozen_module}'), torch.Tensor().new_ones((1, 3, 32, 32)).to(model.device))
                print(f"Decide whether to freeze: " + ', '.join(str(m) for m in modules_names_to_freeze))
                for i, module_name in enumerate(modules_names_to_freeze):
                    models.append(deepcopy(model))
                    freeze_layers(models[i+1].net, eval(f'models[{i+1}].net.{module_name}'), torch.Tensor().new_ones((1, 3, 32, 32)).to(model.device))
                
                correct, total, accs_val, losses = [], [], [], []
                for _ in models:
                    correct.append(0.0)
                    total.append(0.0)
                    accs_val.append(0.0)
                    losses.append([])

            
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)

                    not_aug_inputs = not_aug_inputs.to(model.device)

                    if args.freezing_eval is not None and t > 0 and epoch == 0:
                        for model_copy in models:
                            loss = model_copy.meta_observe(inputs, labels, not_aug_inputs, current_task_labels)
                            assert not math.isnan(loss)
                    else:
                        loss = model.meta_observe(inputs, labels, not_aug_inputs, current_task_labels)

                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)
            
            # Freezing evaluation
            if args.freezing_eval is not None and t > 0 and epoch == 0:
                
                if args.freezing_eval != "training": #evaluation over previous task
                    # buffer_dataset = Buffer_dataset(root="", buffer=eval_buffer)
                    # buffer_loader = DataLoader(buffer_dataset,
                    #                         batch_size=args.batch_size, shuffle=False, num_workers=4)
                
                    correct_buff, total_buff, losses_buff = [], [], []
                    for _ in models:
                        correct_buff.append(0.0)
                        total_buff.append(0.0)
                        losses_buff.append([])

                    for i, data in enumerate(buffers_list[t-1]):
                        inputs, labels, _ = data
                        inputs, labels = inputs.to(model.device), labels.to(model.device)
                        for i, model_copy in enumerate(models):
                            correct_batch, total_batch, loss = model_eval(model_copy, inputs, labels, dataset.get_loss())
                            correct_buff[i] += correct_batch
                            total_buff[i] += total_batch
                            losses_buff[i].append(loss)

                    accs_buff = []
                    for i, loss in enumerate(losses_buff):
                        accs_buff.append(correct_buff[i] / total_buff[i] * 100)
                        losses_buff[i] = statistics.mean(loss)

                # if args.freezing_eval != "training":
                #     eval_buffer = Buffer(args.buffer_size, model.device)

                for i, data in enumerate(val_loader):
                    if args.freezing_eval != "buffer":
                        inputs, labels, _ = data
                        inputs, labels = inputs.to(model.device), labels.to(model.device)
                        for i, model_copy in enumerate(models):
                            correct_batch, total_batch, loss = model_eval(model_copy, inputs, labels, dataset.get_loss())
                            correct[i] += correct_batch
                            total[i] += total_batch
                            losses[i].append(loss)
                    # elif args.freezing_eval != "training":
                    #     inputs, labels = data
                    #     inputs, labels = inputs.to(model.device), labels.to(model.device)
                    #     eval_buffer.add_data(examples=inputs, labels=labels)
                
                if args.freezing_eval != "buffer":
                    accs_val = []
                    for i, loss in enumerate(losses):
                        accs_val.append(correct[i] / total[i] * 100)
                        losses[i] = round(statistics.mean(loss), 2)

                    if args.freezing_eval == "training":
                        print()
                        for i, _ in enumerate(accs_val):
                            print(f"accs_{i} = {accs_val[i]:.4f} e loss_{i} = {losses[i]:.4f}")
                
                if args.freezing_eval == "buffer":
                    print()
                    for i, _ in enumerate(accs_buff):
                        print(f"accs_buff_{i} = {accs_buff[i]:.4f} e loss_buff_{i} = {losses_buff[i]:.4f}")
                        accs_val[i] = accs_buff[i] 
                        losses[i] = losses_buff[i] 

                if args.freezing_eval == "training_and_buff":
                    print()
                    for i, _ in enumerate(accs_buff):
                        accs_val[i] = round(statistics.mean([accs_val[i], accs_buff[i]]), 2)
                        losses[i] = round(statistics.mean([losses[i], losses_buff[i]]), 2)
                        print(f"accs_{i} = {accs_val[i]:.4f} e loss_{i} = {losses[i]:.4f}")
                    

                best_model = losses.index(min(losses))
                
                model.net.load_state_dict(models[best_model].net.state_dict())
                freezed_layers = freezed_layers + best_model

                if  best_model != 0:
                    print(f"It's better freezing up to {modules_names_to_freeze[best_model-1]}")
                    freeze_layers(model.net, eval('model.net.' + modules_names_to_freeze[best_model-1]), torch.Tensor().new_ones((1, 3, 32, 32)).to(model.device))
                else:
                    print(f"It's better not to freeze")
                    if last_frozen_module is not None:
                        freeze_layers(model.net, eval('model.net.' + last_frozen_module), torch.Tensor().new_ones((1, 3, 32, 32)).to(model.device))
        
            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if not args.disable_log:
            logger.log(mean_acc)
            logger.log_fullacc(accs)

        if not args.nowand:
            d2={'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1], 'STEP': t, 'num_freezed_layers': freezed_layers,
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])},
                }

            wandb.log(d2)


    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                    results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()



def get_next_pos_classes(model: ContinualModel, forward_dataset: ContinualDataset, free_heads: list):
    status = model.net.training
    forward_train_loader, _ = forward_dataset.get_data_loaders()
    model.net.eval()
    
    list_labels, list_pred = [], []
    
    #Compute outputs next task
    for data in forward_train_loader:
        inputs, labels, _ = data
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        with torch.no_grad():
            outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        
        list_pred.append(pred.cpu())
        list_labels.append(labels.cpu())
    
    pred = torch.cat(list_pred)
    labels = torch.cat(list_labels)
    
    #create predictions matrix
    predictions_matrix = torch.ones((forward_dataset.N_CLASSES_PER_TASK, forward_dataset.N_CLASSES_PER_TASK * forward_dataset.N_TASKS),dtype=torch.int) * -1
    #only columns in free_outputs are available
    predictions_matrix[:,free_heads] = 0
    
    next_labels = list(set(forward_train_loader.dataset.targets))
    print(f'{forward_dataset.NAME} next classes: {next_labels}')
    
    for i, label in enumerate (next_labels):
        l_mask = [labels == label]
        all_pred = pred[l_mask].unique().tolist()
        print(f'activated_heads: {all_pred}')
        # possible heads
        possible_outputs = list(set(all_pred) & set(free_heads))

        for l in possible_outputs:
            predictions_matrix[i,l] = (pred[l_mask] == l).sum(dim=0).item()
    print(predictions_matrix)
    
    #define the labels-heads mapping for the next task
    list_pos = [0] * len(next_labels)
    for row in range(predictions_matrix.shape[0]):
        index = (predictions_matrix == torch.max(predictions_matrix)).nonzero(as_tuple=False)[0]
        r = index[0].item()
        c = index[1].item()
        list_pos[r] = c
        predictions_matrix[r,:] = -1
        predictions_matrix[:, c] = -1
    
    print('next heads:', list_pos)
    
    model.net.train(status)
    return list_pos