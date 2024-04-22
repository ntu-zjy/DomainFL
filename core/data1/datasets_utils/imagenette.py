import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

lbl_dict = dict(
    n01440764='tench',
    n02102040='English springer',
    n02979186='cassette player',
    n03000684='chain saw',
    n03028079='church',
    n03394916='French horn',
    n03417042='garbage truck',
    n03425413='gas pump',
    n03445777='golf ball',
    n03888257='parachute'
)

class PytorchImagenette(Dataset):
    def __init__(self, annotations_file, root_dir, split='train', transform=None):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            split (string): 'train' for training set and 'val' for validation set.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
        self.root_dir = os.path.join(root_dir, 'imagenette2')
        self.annotations_file = annotations_file
        self.split = split
        self.transform = transform
        if not os.path.exists(self.root_dir):
            print(f"The dataset was not found in {self.root_dir}. Downloading...")
            self.download_and_extract_dataset()

        annotations_path = os.path.join(self.root_dir, annotations_file)
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations file {annotations_path} does not exist.")

        # Updated path to annotations_file based on the split
        self.img_labels = pd.read_csv(annotations_path)

        # keep the lines that path has the split in it
        self.img_labels = self.img_labels[self.img_labels['path'].str.startswith(split)]

        self.lbl_dict = lbl_dict  # Removed the 'or {}' since lbl_dict is defined outside the class
        self.class_to_idx = {v: k for k, v in enumerate(self.lbl_dict.values())}

    def download_and_extract_dataset(self):
        download_and_extract_archive(self.dataset_url, download_root=os.path.dirname(self.root_dir), filename='imagenette2.tgz')

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Filename is now directly available without additional path components
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])
        # print(self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        # Assuming the label is directly available from the CSV
        class_id = self.img_labels.iloc[idx, 1]
        label_name = self.lbl_dict.get(class_id, "Unknown")
        label = self.class_to_idx[label_name]  # Convert class name to index
        if self.transform:
            image = self.transform(image)
        return image, label


class Imagenette:
    def __init__(self, train_preprocess, val_preprocess, location=os.path.expanduser('~/data'), batch_size=128, num_workers=0):
        self.train_dataset = PytorchImagenette('noisy_imagenette.csv', location, split='train', transform=train_preprocess)
        self.test_dataset = PytorchImagenette('noisy_imagenette.csv', location, split='val', transform=val_preprocess)

        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=min(batch_size*8, 1024), num_workers=num_workers, pin_memory=True)

        idx_to_class = {v: k for k, v in self.train_dataset.class_to_idx.items()}
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]


