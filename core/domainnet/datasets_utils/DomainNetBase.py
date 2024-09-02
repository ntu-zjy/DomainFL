import os
import requests
from zipfile import ZipFile
from io import BytesIO
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
import pandas as pd
from PIL import Image


zip_url_dict = {
    'clipart': 'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip',
    'infograph': "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
    'painting': 'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip',
    'quickdraw': 'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
    'real': 'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
    'sketch': 'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip',
}

class BaseDomainNetDataset(Dataset):
    def __init__(self, root_dir, domain, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.domain = domain
        self.split = split
        self.annotations_file = f"{domain}_{split}.txt"
        self.data_zip_url = f"{zip_url_dict[domain]}"
        if domain == 'painting':
            self.annotations_url = f"http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/{self.annotations_file}"
        else:
            self.annotations_url = f"http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/{self.annotations_file}"

        # Check if the data and annotations exist; if not, download them
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        if not os.path.exists(os.path.join(self.root_dir, domain)):
            self.download_and_extract_data()

        annotations_path = os.path.join(self.root_dir, self.annotations_file)
        if not os.path.exists(annotations_path):
            self.download_annotations()

        self.img_labels = pd.read_csv(annotations_path, delimiter=' ', header=None, names=['image_path', 'label'])
        self.num_classes = self.img_labels['label'].max() + 1
        self.img_labels['class_name'] = self.img_labels['image_path'].apply(lambda x: x.split('/')[1])
        self.class_to_idx = self.img_labels[['class_name', 'label']].drop_duplicates().set_index('class_name').to_dict()['label']

    def download_and_extract_data(self):
        print(f"Downloading {self.domain} data from {self.data_zip_url}...")
        download_root = os.path.dirname(self.root_dir)  # The root directory where the data should be downloaded
        download_and_extract_archive(url=self.data_zip_url, download_root=f"{self.root_dir}", filename=f"{self.domain}.zip", md5=None)
        print(f"Data for {self.domain} domain has been downloaded and extracted in {download_root}.")

    def download_annotations(self):
        print(f"Downloading {self.domain} annotations from {self.annotations_url}...")
        response = requests.get(self.annotations_url)
        annotations_path = os.path.join(self.root_dir, self.annotations_file)
        with open(annotations_path, 'wb') as f:
            f.write(response.content)
        print(f"Annotations for {self.domain} domain have been downloaded.")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
