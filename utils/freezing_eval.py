import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from copy import deepcopy

class Buffer_dataset(Dataset):
    """
    Defines Imagenette as for the others pytorch datasets.
    """
    def __init__(self, root: str, buffer, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2615))])
        self.target_transform = target_transform
        self.download = download
        
        self.buffer = deepcopy(buffer)
        
        self.buffer.examples = torch.permute(self.buffer.examples, (0, 2, 3, 1))
        self.data = self.buffer.examples.cpu().numpy()#.astype(np.uint8)
        
        self.targets = self.buffer.labels.cpu().numpy()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]

        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


def model_eval(model, inputs, labels, criterion):
    outputs = model(inputs)
    _, pred = torch.max(outputs.data, 1)
    correct = torch.sum(pred == labels).item()
    total = labels.shape[0]

    loss = criterion(outputs, labels)

    return correct, total, loss.item()
