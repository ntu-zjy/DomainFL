import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from .PACSBase import BasePACSDataset

class SketchD:
    def __init__(self, train_preprocess, val_preprocess, location=os.path.expanduser('~/data/PACS'), batch_size=128, num_workers=0):
        self.train_dataset = BasePACSDataset(location, domain='sketch', split='train', transform=train_preprocess)
        self.test_dataset = BasePACSDataset(location, domain='sketch', split='test', transform=val_preprocess)
        self.val_dataset = None

        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        self.val_loader = None

        class_to_idx = self.train_dataset.class_to_idx
        # self.classnames = [c.replace('_', ' ') for c in class_to_idx.keys()]

        self.update_classnames()

    def update_classnames(self):
        self.classnames = [c.replace('_', ' ') for c in self.train_dataset.class_to_idx.keys()]


