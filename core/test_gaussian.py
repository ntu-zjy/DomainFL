import json
import os
import copy
import time
import torch
import argparse
from utils.get_data import data1, data2
from utils.get_data import get_data
from utils.data_utils import build_subset
from utils.server import Server
from utils.client import Client
from tqdm import tqdm
from utils.json_utils import generate_json_config
import open_clip
import warnings
import numpy as np
warnings.simplefilter("ignore")

torch.manual_seed(1)
torch.cuda.manual_seed(1) if torch.cuda.is_available() else None

def input_mapping(x, B):
    x_projected = torch.matmul(x, B)
    return torch.cat([torch.sin(x_projected), torch.cos(x_projected)], dim=-1)

def run():
    dataset = data2
    pretrained_model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    # initialize clients
    # client image encoder is the same as the global image encoder
    clients, cls_heads, cds = [], [], []
    for data_name in dataset:
        cd = get_data(data_name, train_preprocess, val_preprocess, f'./data2/{data_name}', 128, 12)
        cd = build_subset(cd, 100)
        cds.append(cd)

    pretrained_model.eval()
    pretrained_model = pretrained_model.to('cuda')
    test_data1 = cds[0].test_loader
    test_data2 = cds[1].test_loader

    with torch.no_grad():
        for (image1, label1), (image2, label2) in zip(test_data1, test_data2):
            image1 = image1.to('cuda')
            image2 = image2.to('cuda')
            f1 = pretrained_model.encode_image(image1)
            f2 = pretrained_model.encode_image(image2)
            print('label1:', label1)
            print('label2:', label2)
            # cal the cosine similarity
            sim = torch.nn.functional.cosine_similarity(f1, f2)
            print("normal similarity:", sim)

            B_gauss = 10 * torch.randn(512, 512).to('cuda')
            encoded_f1 = pretrained_model.encode_image(image1)
            encoded_f2 = pretrained_model.encode_image(image2)
            mapped_f1 = input_mapping(encoded_f1, B_gauss)
            mapped_f2 = input_mapping(encoded_f2, B_gauss)

            sim = torch.nn.functional.cosine_similarity(mapped_f1, mapped_f2)
            print("gaussian similarity:", sim)

            break

if __name__ == "__main__":
    run()