import os
import torch
import torchvision.datasets as datasets

class FashionMNIST:
    def __init__(self,
                 train_preprocess,
                 val_preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16):


        self.train_dataset = datasets.FashionMNIST(
            root=location,
            download=True,
            train=True,
            transform=train_preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        self.test_dataset = datasets.FashionMNIST(
            root=location,
            download=True,
            train=False,
            transform=val_preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=min(batch_size*8, 1024),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        self.classnames = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']