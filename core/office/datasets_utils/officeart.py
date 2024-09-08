import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from .OfficeHomeBase import BaseOfficeHomeDataset

class OfficeArt:
    def __init__(self, train_preprocess, val_preprocess, location=os.path.expanduser('~/data'), batch_size=128, num_workers=0):
        self.train_preprocess = train_preprocess
        self.val_preprocess = val_preprocess

        self.train_dataset = BaseOfficeHomeDataset(location, domain='OfficeArt', split='train', transform=train_preprocess)
        self.test_dataset = BaseOfficeHomeDataset(location, domain='OfficeArt', split='test', transform=val_preprocess)
        self.val_dataset = None

        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        self.val_loader = None

        class_to_idx = self.train_dataset.class_to_idx
        self.update_classnames()

    def update_classnames(self):
        self.classnames = [c.replace('_', ' ') for c in self.train_dataset.class_to_idx.keys()]


