import os
import numpy as np
from zipfile import ZipFile
from io import BytesIO
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
import pandas as pd
from PIL import Image


class BaseAdaptiopeDataset(Dataset):
    def __init__(self, root_dir, domain, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.domain = domain
        self.split = split

        # Build the list of image paths and labels
        self.img_labels = self._build_image_label_list()
        # Get the list of unique class names
        self.classes = sorted(set(self.img_labels['label']))
        # Create a mapping dictionary from class names to indices
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        # Update img_labels to use numeric labels instead of string names
        self.img_labels['label'] = self.img_labels['label'].map(self.class_to_idx)

    def _build_image_label_list(self):

        # Initialize lists to store image paths and labels
        img_paths = []
        labels = []
        # Traverse all category directories in the specified domain
        domain_path = os.path.join(self.root_dir, self.domain)
        for class_dir in os.listdir(domain_path):
            class_path = os.path.join(domain_path, class_dir)
            if os.path.isdir(class_path):
                # randomly split test and train data into 5:5
                np.random.seed(0)
                train_idx = np.random.choice(len(os.listdir(class_path)), int(len(os.listdir(class_path)) * 0.5), replace=False).tolist()  # Convert train_idx to a list
                if self.split == 'train':
                    # data from train_idx
                    class_path_list = [os.listdir(class_path)[i] for i in train_idx]  # Use train_idx as a list
                else:
                    # data not from train_idx
                    class_path_list = [os.listdir(class_path)[i] for i in range(len(os.listdir(class_path))) if i not in train_idx]
                # Traverse all image files in the directory
                for img_file in class_path_list:
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_paths.append(os.path.join(class_path, img_file))
                        labels.append(class_dir.replace('_', ' '))  # Remove underscores from class names
        # Create DataFrame
        data = pd.DataFrame({
            'image_path': img_paths,
            'label': labels
        })
        return data

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]  # Label is already a numeric index
        if self.transform:
            image = self.transform(image)
        return image, label