import os
import torch
from torchvision.datasets import Food101 as PyTorchFood101

class Food101:
    def __init__(self,
                 train_preprocess,
                 val_preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16):

        self.train_dataset = PyTorchFood101(
            root=location, download=True, split='train', transform=train_preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
        )

        self.test_dataset = PyTorchFood101(
            root=location, download=True, split='test', transform=val_preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=min(batch_size*8, 1024), shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes