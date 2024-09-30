<div align="center">
  
# Wake-Sleep Consolidated Learning
Amelia Sorrenti, Giovanni Bellitto, Federica Proietto Salanitri, Matteo Pennisi, Simone Palazzo, Concetto Spampinato 

<!--- [![Paper](http://img.shields.io/badge/paper-arxiv.2401.08623-B31B1B.svg)](https://arxiv.org/abs/2401.08623) -->
[![arXiv](https://img.shields.io/badge/arXiv-_-darkgreen?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2401.08623)
[![cvf](https://img.shields.io/badge/IEEE%20TNNLS_2024-_-darkgreen?style=flat-square)](https://ieeexplore.ieee.org/document/10695036)


</div>

# Overview
The official PyTorch implementation for paper: <b> "Wake-Sleep Consolidated Learning" </b>.

# Abstract 
We propose Wake-Sleep Consolidated Learning (WSCL), a learning strategy leveraging Complementary Learning System theory and the wake-sleep phases of the human brain to improve the performance of deep neural networks for visual classification tasks in continual learning settings. Our method learns continually via the synchronization between distinct wake and sleep phases. During the wake phase, the model is exposed to sensory input and adapts its representations, ensuring stability through a dynamic parameter freezing mechanism and storing episodic memories in a short-term temporary memory (similarly to what happens in the hippocampus). During the sleep phase, the training process is split into NREM and REM stages. In the NREM stage, the model's synaptic weights are consolidated using replayed samples from the short-term and long-term memory and the synaptic plasticity mechanism is activated, strengthening important connections and weakening unimportant ones. In the REM stage, the model is exposed to previously-unseen realistic visual sensory experience, and the dreaming process is activated, which enables the model to explore the potential feature space, thus preparing synapses to future knowledge. We evaluate the effectiveness of our approach on three benchmark datasets: CIFAR-10, Tiny-ImageNet and FG-ImageNet. In all cases, our method outperforms the baselines and prior work, yielding a significant performance gain on continual visual classification tasks. Furthermore, we demonstrate the usefulness of all processing stages and the importance of dreaming to enable positive forward transfer.

# Method
<p align = "center"><img src="img/wscl.png" width="600" style = "text-align:center"/></p>
 
## How to run

- CIFAR-10
```bash
python utils/main.py --dataset dream-seq-cifar10 --dataset_2 CIFAR100 --forward_dataset seq-cifar10 --buffer_size 200 --model er_ace --freezing_eval training_and_buff --validation 1 --load_best_args
```

- CIFAR-100
```bash
python utils/main.py --dataset dream-seq-cifar100 --dataset_2 AUXImageNet100 --forward_dataset seq-cifar100 --model er_ace --buffer_size 500 --freezing_eval training_and_buff --validation 1 --load_best_args
```

- Tiny-ImageNet
```bash
python utils/main.py --dataset dream-seq-halftinyimg --dataset_2 TINYIMG --forward_dataset seq-halftinyimg --buffer_size 500 --model er_ace --freezing_eval training_and_buff --validation 1 --load_best_args
```

- FG-ImageNet
```bash
python utils/main.py --model er_ace --dataset dream-seq-img100 --dataset_2 AUXImageNet100 --forward_dataset seq-img100 --buffer_size 1000 --freezing_eval training_and_buff --validation 1 --load_best_args
```
