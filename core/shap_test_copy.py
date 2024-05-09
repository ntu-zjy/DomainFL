import json
import os
import copy
import time
import torch
import torchvision
import torchvision.transforms as transforms
from models.CLIP import *
from utils.get_data import data
from utils.get_data import get_data
from utils.data_utils import build_subset, split_train_and_val
from utils.json_utils import generate_json_config
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore")

torch.manual_seed(1)
torch.cuda.manual_seed(1) if torch.cuda.is_available() else None

dataset = globals()['data']

train_preprocess = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            transforms.Resize((224, 224))
                            ])
cds = []
for id, data_name in enumerate(dataset):
    cd = get_data(data_name, train_preprocess, train_preprocess, f'./data/{data_name}', 128, 12)
    cd = build_subset(cd, 3)
    cds.append(cd)
print(cds)
#build a resnet50 model with 10 classes
model = torchvision.models.resnet50(pretrained=False, num_classes=3)

model1 = copy.deepcopy(model)
model2 = copy.deepcopy(model)
model3 = copy.deepcopy(model)

def train(cd, model):
    model = model.to('cuda')
    print('Training model')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(2):
        print(f'Epoch {epoch}')
        for image, label in cd:
            image = image.to('cuda')
            label = label.to('cuda')
            out = model(image)
            loss = torch.nn.functional.cross_entropy(out, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    model = model.to('cpu')
    return model

model1 = train(cds[0].train_loader, model1)
model2 = train(cds[1].train_loader, model2)
model3 = train(cds[2].train_loader, model3)

from torch.utils.data import DataLoader, Subset, ConcatDataset

class CustomConcatDataset(ConcatDataset):
    def __init__(self, datasets, class_to_idx):
        super().__init__(datasets)
        self.class_to_idx = class_to_idx

def concat_datasets(dataObjects):

    # Concatenate train datasets
    train_datasets = [dataObject.train_dataset for dataObject in dataObjects]
    class_to_idx = train_datasets[0].class_to_idx
    train_dataset = CustomConcatDataset(train_datasets, class_to_idx)

    # Concatenate test datasets
    test_datasets = [dataObject.test_dataset for dataObject in dataObjects]
    test_dataset = CustomConcatDataset(test_datasets, class_to_idx)

    # Create a new DataLoader for the concatenated train dataset
    batch_size = dataObjects[0].train_loader.batch_size
    num_workers = dataObjects[0].train_loader.num_workers
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Create a new DataLoader for the concatenated test dataset
    batch_size = dataObjects[0].test_loader.batch_size
    num_workers = dataObjects[0].test_loader.num_workers
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Create a new DataObject with the concatenated datasets
    dataObject = copy.deepcopy(dataObjects[0])
    dataObject.train_dataset = train_dataset
    dataObject.test_dataset = test_dataset
    dataObject.train_loader = train_loader
    dataObject.test_loader = test_loader
    dataObject.update_classnames()

    return dataObject

cds = [cds[0], cds[1], cds[2]]
conds = concat_datasets(cds)

import torchvision.models as models
import torch.nn as nn

def replace_relu(model):
    """ Replace all inplace ReLU with out-of-place ReLU to avoid RuntimeError during SHAP value calculation. """
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, nn.ReLU(inplace=False))
        elif isinstance(module, nn.Sequential):
            replace_relu(module)

# Example usage with ResNet50
model_c = torchvision.models.mobilenet_v2(pretrained=False, progress=False)
replace_relu(model_c)  # Adjust the model to use non-in-place ReLU

# Continue with your setup and training as before
model_c = train(conds.train_loader, model_c)
model_c.eval()  # Set the model to evaluation mode

# Now try to compute SHAP values again
import shap

batch = next(iter(conds.train_loader))
images, _ = batch

background = images[:100]  # Ensure tensors are on the right device
test_images = images[100:103]

e = shap.DeepExplainer(model_c, background)
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy)
