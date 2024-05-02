import json
import os
import copy
import time
import torch
import argparse
from models.CLIP import *
from utils.get_data import data1, data2, source, target
from utils.get_data import get_data
from utils.data_utils import build_subset, concat_test_datasets, generate_domain_label, split_dataloader_by_labels
from utils.server import Server
from utils.client import Client
from utils.json_utils import generate_json_config
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

import warnings
warnings.simplefilter("ignore")

torch.manual_seed(1)
torch.cuda.manual_seed(1) if torch.cuda.is_available() else None


def run(args):
    # initialize server
    server = Server(args)

    # set dataset
    dataset = globals()[args.dataset]

    # initialize clients
    # client image encoder is the same as the global image encoder
    clients = []
    for id, data_name in enumerate(dataset):
        init_image_encoder = copy.deepcopy(server.image_encoder)
        cd = get_data(data_name, server.train_preprocess, server.val_preprocess, f'./{args.dataset}/{data_name}', args.batch_size, args.num_workers)
        cd = build_subset(cd, args.subset_size)
        cls_head = server.generate_cls_head(cd, data_name)
        # by setting test_split=True, we can load the pretrained weight of the global adapter and local adapter
        client = Client(args, id, cd.train_dataset, cd.test_dataset, cd.train_loader, cd.test_loader, cd.classnames, init_image_encoder, cls_head, data_name, test_split=True)
        clients.append(client)
        del cd
    # print('last layer of adapter in clients[0].model:', clients[0].model.base.adapter.state_dict()['fc.2.weight'])
    # build target test data
    for target_name in target:
        target_data = get_data(target_name, server.train_preprocess, server.val_preprocess, f'./target/{target_name}', args.batch_size, args.num_workers)
        target_data = build_subset(target_data, args.subset_size)

    # test clients
    start_time = time.time()

    for id, client in enumerate(clients):
        acc = client.test_on_target(target_data.test_loader)
    # with open(f'./results/{args.algorithm}_split/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'a+') as f:
    #     json.dump({'round':0, 'own_acc': client_own_domain_acc, 'other_acc': client_other_domain_acc, 'domain_acc': acc}, f)
    #     f.write('\n')

    test_time = time.time() - start_time
    print(f'total test time cost: {test_time:.2f}s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DomainFL')
    parser.add_argument('-d','--dataset', type=str, default='source', help='Dataset name')
    parser.add_argument('-ss','--subset_size', type=int, default=100, help='Subset size')
    parser.add_argument('-m','--model', type=str, default='CLIP', help='Model name')
    parser.add_argument('-ien','--image_encoder_name', type=str, default='ViT-B-32', help='Image encoder name')
    parser.add_argument('-optim','--optimizer', type=str, default='AdamW', help='Optimizer name')
    parser.add_argument('-lr','--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-clip','--clip', type=float, default=5, help='Gradient clip')
    parser.add_argument('-bs','--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('-le','--local_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-warm_up','--warm_up', type=int, default=5, help='Warm up epochs')
    parser.add_argument('-gr','--global_rounds', type=int, default=50, help='Number of global rounds')
    parser.add_argument('-device','--device', type=str, default='cuda', help='Device')
    parser.add_argument('-num_workers','--num_workers', type=int, default=12, help='Number of workers')
    parser.add_argument('-eval','--eval_interval', type=int, default=1, help='Log interval')
    parser.add_argument('-did','--device_id', type=str, default=0, help='Device ID')
    parser.add_argument('-seed','--seed', type=int, default=1, help='Seed')

    # parser only for auto test split
    parser.add_argument('-algo','--algorithm', type=str, default='fedts', help='Algorithm name')
    # parser.add_argument()
    args = parser.parse_args()

    if args.device == 'cuda':
        args.device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    # os.makedirs(f'./results/{args.algorithm}_split/', exist_ok=True)
    # with open(f'./results/{args.algorithm}_split/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'w+') as f:
    #     json.dump(generate_json_config(args), f)
    #     f.write('\n')

    run(args)
