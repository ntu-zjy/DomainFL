import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from .OfficeHomeBase import BaseOfficeHomeDataset

class OfficeArt:
    def __init__(self, train_preprocess, val_preprocess, location=os.path.expanduser('~/data'), batch_size=128, num_workers=0):
        self.dataset = BaseOfficeHomeDataset(location, domain='OfficeArt', split='train', transform=train_preprocess)
        class_to_idx = self.dataset.class_to_idx

        # Extract labels from the dataset for stratified splitting
        labels = self.dataset.img_labels['label'].tolist()

        # Configure StratifiedShuffleSplit for stratified sampling, with a 70:30 train:test split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)

        # Use StratifiedShuffleSplit to get indices for train and test sets
        train_idx, test_idx = next(sss.split(X=[0]*len(labels), y=labels))

        train_labels = self.dataset.img_labels['label'].values[train_idx]
        test_labels = self.dataset.img_labels['label'].values[test_idx]

        # Create subsets for training and testing using the indices from stratified split
        self.train_dataset = CustomSubset(self.dataset, train_idx, class_to_idx, train_labels)
        self.test_dataset = CustomSubset(self.dataset, test_idx, class_to_idx, test_labels)
        self.val_dataset = None

        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        self.val_loader = None

        # self.classnames = [c.replace('_', ' ') for c in class_to_idx.keys()]

        self.update_classnames()

    def update_classnames(self):
        self.classnames = [c.replace('_', ' ') for c in self.train_dataset.class_to_idx.keys()]


class CustomSubset(Subset):
    def __init__(self, dataset, indices, class_to_idx, image_labels=None):
        super().__init__(dataset, indices)
        self.class_to_idx = class_to_idx
        self.image_labels = image_labels
