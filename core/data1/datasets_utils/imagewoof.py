import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

lbl_dict = dict(
    n02093754='Australian terrier',
    n02089973='Border terrier',
    n02099601='Samoyed',
    n02087394='Beagle',
    n02105641='Shih-Tzu',
    n02096294='English foxhound',
    n02088364='Rhodesian ridgeback',
    n02115641='Dingo',
    n02111889='Golden retriever',
    n02086240='Old English sheepdog'
)

class PytorchImagewoof(Dataset):
    def __init__(self, annotations_file, root_dir, split='train', transform=None):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            split (string): 'train' for training set and 'val' for validation set.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Updated path to annotations_file based on the split
        self.img_labels = pd.read_csv(os.path.join(root_dir, 'imagewoof2', annotations_file))

        # keep the lines that path has the split in it
        self.img_labels = self.img_labels[self.img_labels['path'].str.startswith(split)]
        self.root_dir = os.path.join(root_dir, 'imagewoof2')  # Update to include split
        self.transform = transform
        self.lbl_dict = lbl_dict  # Removed the 'or {}' since lbl_dict is defined outside the class
        self.class_to_idx = {v: k for k, v in enumerate(self.lbl_dict.values())}

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


class Imagewoof:
    def __init__(self, preprocess, location=os.path.expanduser('~/data'), batch_size=32, num_workers=0):
        self.train_dataset = PytorchImagewoof('noisy_imagewoof.csv', location, split='train', transform=preprocess)
        self.test_dataset = PytorchImagewoof('noisy_imagewoof.csv', location, split='val', transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=min(batch_size*8, 1024), num_workers=num_workers)

        idx_to_class = {v: k for k, v in self.train_dataset.class_to_idx.items()}
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]


