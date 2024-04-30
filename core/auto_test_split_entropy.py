import json
import os
import copy
import time
import torch
import argparse
from models.CLIP import *
from utils.get_data import data1
from utils.get_data import data2
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

def max_entropy_classify(model, dataloader, args):
    pred_domain_labels = []
    for input, target in dataloader:
        input = input.to(args.device)
        target = target.to(args.device)
        image_features = model.base.model.encode_image(input)
        global_features = model.base.adapter(image_features) + image_features
        local_features = model.base.local_adapter(image_features) + image_features
        global_output = model.head(global_features)
        local_output = model.head(local_features)
        global_entropy = -torch.sum(global_output * torch.log(global_output), dim=1)
        local_entropy = -torch.sum(local_output * torch.log(local_output), dim=1)

        # if global entropy is larger than local entropy, then the label is 0 (out of domain)
        # if global entropy is smaller than local entropy, then the label is 1 (in domain)
        # k is a hyperparameter to control the threshold, as the global entropy is a more common sense
        pred_domain_label = (global_entropy*0.9 < local_entropy).long()
        pred_domain_labels.append(pred_domain_label)
    pred_domain_labels = torch.cat(pred_domain_labels, dim=0)
    return pred_domain_labels.cpu().numpy()

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

    # test clients
    start_time = time.time()
    all_clients_test_dataloader = concat_test_datasets(clients)

    client_own_domain_acc = []
    client_other_domain_acc = []
    for id, client in enumerate(clients):
        client.domain_label = generate_domain_label(id, clients)

        model = copy.deepcopy(client.model)
        model.eval()
        with torch.no_grad():
            client.pred_domain_label = max_entropy_classify(model, all_clients_test_dataloader, args)
            acc = np.sum(client.pred_domain_label == client.domain_label) / len(client.domain_label)
            print('accuracy:', np.sum(client.pred_domain_label == client.domain_label) / len(client.domain_label))

        client.tp_dataloader, client.tn_dataloader, client.fp_dataloader, client.fn_dataloader = \
            split_dataloader_by_labels(all_clients_test_dataloader, client.pred_domain_label, client.domain_label)

        # do test
        own_domain_preds, other_domain_preds = [], []
        own_domain_labels, other_domain_labels = [], []
        with torch.no_grad():
            for input, target in client.tp_dataloader:
                input, target = input.to(args.device), target.to(args.device)
                output = model.base.model.encode_image(input)
                output += model.base.local_adapter(output)
                output = model.head(output)
                pred = output.argmax(dim=1, keepdim=True)
                own_domain_preds.append(pred)
                own_domain_labels.append(target)

            for input, target in client.fn_dataloader:
                input, target = input.to(args.device), target.to(args.device)
                output = model.base.model.encode_image(input)
                output += model.base.adapter(output)
                output = model.head(output)
                pred = output.argmax(dim=1, keepdim=True)
                own_domain_preds.append(pred)
                own_domain_labels.append(target)

            for input, target in client.fp_dataloader:
                input, target = input.to(args.device), target.to(args.device)
                output = model.base.model.encode_image(input)
                output += model.base.local_adapter(output)
                output = model.head(output)
                pred = output.argmax(dim=1, keepdim=True)
                other_domain_preds.append(pred)
                other_domain_labels.append(target)

            for input, target in client.tn_dataloader:
                input, target = input.to(args.device), target.to(args.device)
                output = model.base.model.encode_image(input)
                output += model.base.adapter(output)
                output = model.head(output)
                pred = output.argmax(dim=1, keepdim=True)
                other_domain_preds.append(pred)
                other_domain_labels.append(target)

        # calculate acc
        own_domain_preds = torch.cat(own_domain_preds, dim=0)
        own_domain_labels = torch.cat(own_domain_labels, dim=0)
        other_domain_preds = torch.cat(other_domain_preds, dim=0)
        other_domain_labels = torch.cat(other_domain_labels, dim=0)
        all_domain_preds = torch.cat([own_domain_preds, other_domain_preds], dim=0)
        all_domain_labels = torch.cat([own_domain_labels, other_domain_labels], dim=0)
        own_domain_acc = own_domain_preds.eq(own_domain_labels.view_as(own_domain_preds)).sum().item() / len(own_domain_labels)
        other_domain_acc = other_domain_preds.eq(other_domain_labels.view_as(other_domain_preds)).sum().item() / len(other_domain_labels)
        print('own domain acc:', own_domain_preds.eq(own_domain_labels.view_as(own_domain_preds)).sum().item() / len(own_domain_labels))
        print('other domain acc:', other_domain_preds.eq(other_domain_labels.view_as(other_domain_preds)).sum().item() / len(other_domain_labels))
        print('all domain acc:', all_domain_preds.eq(all_domain_labels.view_as(all_domain_preds)).sum().item() / len(all_domain_labels))
        client_own_domain_acc.append(own_domain_acc)
        client_other_domain_acc.append(other_domain_acc)
    with open(f'./results/{args.algorithm}_split/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'a+') as f:
        json.dump({'round':0, 'own_acc': client_own_domain_acc, 'other_acc': client_other_domain_acc, 'domain_acc': acc}, f)
        f.write('\n')

    test_time = time.time() - start_time
    print(f'total test time cost: {test_time:.2f}s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DomainFL')
    parser.add_argument('-d','--dataset', type=str, default='data1', help='Dataset name')
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

    os.makedirs(f'./results/{args.algorithm}_split/', exist_ok=True)
    with open(f'./results/{args.algorithm}_split/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'w+') as f:
        json.dump(generate_json_config(args), f)
        f.write('\n')

    run(args)
