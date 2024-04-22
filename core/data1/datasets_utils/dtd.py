import os
import torch
import torchvision.datasets as datasets

class DTD:
    def __init__(self,
                 train_preprocess, val_preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16):


        self.train_dataset = datasets.DTD(
            root=location,
            download=True,
            split='train',
            transform=train_preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = datasets.DTD(
            root=location,
            download=True,
            split='test',
            transform=val_preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=min(batch_size*8, 1024),
            shuffle=False,
            num_workers=num_workers
        )

        self.val_dataset = datasets.DTD(
            root=location,
            download=True,
            split='val',
            transform=val_preprocess
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=min(batch_size*8, 1024),
            shuffle=False,
            num_workers=num_workers
        )

        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]