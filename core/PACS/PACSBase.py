import os
import requests
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


def download_github_folder(repo_url, folder_path, local_dir):
    api_url = repo_url.replace("github.com", "api.github.com/repos")
    api_url = api_url.replace("tree/master", "contents") + f"/{folder_path}"

    response = requests.get(api_url)
    if response.status_code == 200:
        contents = response.json()
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
            print(f'make{local_dir}')

        for content in contents:
            download_url = content['download_url']
            if download_url:  # 确保是文件而不是目录
                file_name = local_dir + f"/{content['name']}"
                # file_name = os.path.join(local_dir, content['name'])
                # print(f"Downloading {file_name}...")
                file_response = requests.get(download_url)
                with open(file_name, 'wb') as f:
                    f.write(file_response.content)
            else:
                # 如果是目录，递归下载
                new_folder_path = folder_path + f"/{content['name']}"
                # os.path.join(folder_path, content['name'])
                print(f"Downloading {new_folder_path}...")

                new_local_dir = local_dir + f"/{content['name']}"
                download_github_folder(repo_url, new_folder_path, new_local_dir)
        return True
    else:
        print(f"Failed to fetch contents from {api_url}. Status code: {response.status_code}")
        return False


PACS_url = 'https://github.com/MachineLearning2020/Homework3-PACS/contents'


class BasePACSDataset(Dataset):
    def __init__(self, root_dir, domain, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.domain = domain
        self.split = split

        # Assumes data is organized as /PACS/domain/class/image.jpg
        self.domain_dir = root_dir + f"/{domain}"
        self.img_labels = self._load_annotations()

    def _load_annotations(self):
        """
        Load image paths and labels.
        """
        img_labels = []
        class_names = sorted(os.listdir(self.domain_dir))  # Assuming class directories are sorted
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(self.domain_dir, class_name)
            if os.path.isdir(class_dir):
                for img_filename in os.listdir(class_dir):
                    img_path = os.path.join(class_name, img_filename)
                    label = self.class_to_idx[class_name]
                    img_labels.append((img_path, label))

        return pd.DataFrame(img_labels, columns=['image_path', 'label'])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.domain, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == 'main':
    # 示例用法
    root_dir = "'~/data/PACS"
    domain = "cartoon"  # 例如，选择一个域
    dataset = BasePACSDataset(root_dir=root_dir, domain=domain, transform=None)
