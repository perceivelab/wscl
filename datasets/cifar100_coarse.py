from torchvision.datasets import CIFAR100
import numpy as np


#from https://github.com/ryanchankh/cifar100coarse
class CIFAR100Coarse(CIFAR100):
    '''
    Converts CIFAR100 PyTorch dataset from sparse labels to coarse labels based on superclass.
    '''
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        self.targets = coarse_labels[self.targets]

        # update classes
        '''
        self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
        '''
        
        self.classes = ['aquatic mammals',
                        'fish',
                        'flowers',
                        'food containers',
                        'fruit and vegetables',
                        'household electrical devices',
                        'household furniture',
                        'insects',
                        'large carnivores',
                        'large man-made outdoor things',
                        'large natural outdoor scenes',
                        'large omnivores and herbivores',
                        'medium-sized mammals',
                        'non-insect invertebrates',
                        'people',
                        'reptiles',
                        'small mammals',
                        'trees',
                        'vehicles 1',
                        'vehicles 2'
                        ]
        
        # remove these classes from the dataset as they are similar to those in CIFAR10
        labels_to_remove = [8,11,12,13,15,16,18,19] 
        mask_list = [np.array(self.targets) != label for label in labels_to_remove]
        dataset_mask = np.stack(mask_list, axis=1)
        dataset_mask = np.all(dataset_mask, axis=1)
        self.data = self.data[dataset_mask]
        self.targets = self.targets[dataset_mask]
        # update labels of remaining classes
        old_labels = list(set(self.targets))
        old_labels.sort()
        classes = []
        for i, l in enumerate(old_labels):
            self.targets[self.targets==l] = i
            classes.append(self.classes[l])
        self.classes = classes