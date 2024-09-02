import json
import os
import copy
import time
import torch
import random
import argparse
from models.CLIP import *
from utils.get_data import domainnet
from utils.get_data import get_data
from utils.data_utils import build_subset, split_train_and_val
from utils.server import Server
from utils.clientours import Client
from utils.json_utils import generate_json_config
import warnings
import numpy as np
from collections import defaultdict
warnings.simplefilter("ignore")

torch.manual_seed(1)
torch.cuda.manual_seed(1) if torch.cuda.is_available() else None

def generate_protos_training_data(uploaded_protos, batchsize=10):
    classes = uploaded_protos[0].keys()
    protos = []
    labels = []
    for proto in uploaded_protos:
        for c in classes:
            protos.append(proto[c])
            labels.append(c)

    # generate batched data
    protos = torch.stack(protos)
    labels = torch.tensor(labels, dtype=torch.long)
    # print('protos:', protos.shape)
    protos = protos.view(-1, batchsize, protos.shape[-1])
    # print('protos:', protos.shape)
    labels = labels.view(-1, batchsize)
    protos = protos.to(torch.float32)
    # shuffle the training data
    for i, (proto, label) in enumerate(zip(protos, labels)):
        idx = torch.randperm(proto.shape[0])
        proto = proto[idx, :].view(proto.shape)
        label = label[idx].view(label.shape)
        protos[i] = proto
        labels[i] = label

    # generate training data
    training_data = []
    for i in range(protos.shape[0]):
        training_data.append((protos[i], labels[i]))
    # print('training_data:', training_data)
    print('the parameters of the training data:', protos.numel() * protos.element_size() * 8)
    return training_data

def send_adaptive_global_adapter(global_adapter, clientObjs):
    for client in clientObjs:
        client.model.base.global_adapter.load_state_dict(global_adapter)
        client.model.base.adapter.load_state_dict(global_adapter)
    return clientObjs

def send_global_head(global_cls_head, clientObjs):
    for client in clientObjs:
        client.model.head.load_state_dict(global_cls_head.state_dict())
    return clientObjs

def server_adative_training(training_data, server, threshold=0.001, num_losses=20):
    losses = []
    server.image_encoder.train()
    server.global_cls_head.train()
    server.freeze_except_global_adapter()
    optimizer = torch.optim.AdamW(server.image_encoder.global_adapter.parameters(), lr=1e-5)
    while True:
        for i, (proto, label) in enumerate(training_data):
            # print('proto:', proto.shape)
            # print('label:', label.shape)
            optimizer.zero_grad()
            proto = proto.to(server.device)
            label = label.to(server.device)
            rep = server.image_encoder.global_adapter(proto)
            rep = rep + proto
            output = server.global_cls_head(rep)
            loss = server.criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(server.image_encoder.global_adapter.parameters(), 1)
            optimizer.step()

            losses.append(loss.item())
        if np.std(losses[-num_losses:]) < threshold and len(losses) > num_losses:
            print(f'server epoch {i} loss std: {np.std(losses[-num_losses:])}')
            break

    return server.image_encoder.global_adapter

def receive_protos(clients):
    uploaded_ids = []
    uploaded_protos = []
    for client in clients:
        uploaded_ids.append(client.id)
        uploaded_protos.append(client.protos)
    return uploaded_protos

def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label

def calculate_fedts_weights(clients):
    # every client use the same weight
    weights = [1/len(clients) for c in clients]
    return weights


def proto_initialization(clientObjs, server):
    uploaded_protos = receive_protos(clientObjs)
    # global_protos = proto_aggregation(uploaded_protos) # do not aggregate the protos !!!
    training_data = generate_protos_training_data(uploaded_protos)
    global_adapter = server_adative_training(training_data, server)
    clientObjs = send_adaptive_global_adapter(global_adapter, clientObjs)
    clientObjs = send_global_head(server.global_cls_head, clientObjs)
    server.image_encoder.global_adapter.load_state_dict(global_adapter.state_dict())
    print('parameters of the global adapter:', sum(p.numel() * p.element_size() * 8 for p in global_adapter.parameters()))
    return clientObjs, server

def calculate_fedavg_weights(clients):
    total_train_num = 0
    num_list = []
    for c in clients:
        train_num = len(c.train_dataloader) * c.batch_size
        total_train_num += train_num
        num_list.append(train_num)
    weights = [num/total_train_num for num in num_list]
    return weights

def fedavg(weights, clientObjs, server):
    print("FedAvg... with weights: ", weights)
    # server receive the adapters from clients
    adapters = [c.model.base.adapter for c in clientObjs]

    # fedavg aggregation
    server_global_adapter = copy.deepcopy(server.image_encoder.global_adapter)
    for param in server_global_adapter.parameters():
        param.data.zero_()

    for adapter in adapters:
        for w, global_param, param in zip(weights, server_global_adapter.parameters(), adapter.parameters()):
            global_param.data += w * param.data.clone()
    # set the global adapter to the server
    server.image_encoder.global_adapter.load_state_dict(server_global_adapter.state_dict())

    # send the global adapter back to the clients
    # param will be covered as global param
    for id in range(len(clientObjs)):
        for param, global_param in zip(clientObjs[id].model.base.adapter.parameters(), server_global_adapter.parameters()):
            param.data = global_param.data.clone()

    return clientObjs, server

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from collections import defaultdict

def generate_fewshot_data(dataObject, few_shot_num, batch_size=8):
    """
    Create a DataLoader for few-shot learning, where each class contains only a certain percentage of its original samples.
    This version always selects the first N percent of samples from each class, where N is the percentage specified.

    Args:
    dataset (Dataset): The original dataset.
    few_shot_num (int): The number of samples to include from each class (e.g., 10 for 10-shot).
    classnames (list or dict): A list or dictionary mapping labels to class names.
    batch_size (int): The size of each batch (default is 32).

    Returns:
    DataLoader: A DataLoader containing a subset of the original dataset, sampled according to the specified percentage.
    """
    classnames = dataObject.classnames
    indices_per_class = defaultdict(list)

    train_labels = dataObject.train_dataset.image_labels
    unique_labels = np.unique(train_labels)

    for label in unique_labels:
        label_indices = np.where(train_labels == label)[0]
        indices_per_class[classnames[label]].extend(label_indices)

    selected_indices = []
    for indices in indices_per_class.values():
        num_samples = max(1, min(few_shot_num, len(indices)))
        # Select the first num_samples indices from the list
        selected_indices.extend(indices[:num_samples])

    subset = Subset(dataObject.train_dataset, selected_indices)

    dataObject.train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataObject

def run(args):
    # initialize server
    server = Server(args)

    # set dataset
    dataset = globals()[args.dataset]

    # initialize clients
    # client image encoder is the same as the global image encoder
    clients = []
    cls_heads = []
    few_shot_cds = []
    for id, data_name in enumerate(dataset):
        init_image_encoder = copy.deepcopy(server.image_encoder)
        cd = get_data(data_name, server.train_preprocess, server.val_preprocess, args.batch_size, args.num_workers)
        cd = build_subset(cd, args.subset_size)
        cd = split_train_and_val(cd)
        few_shot_cd = generate_fewshot_data(cd, args.fewshot)
        cls_head = server.generate_cls_head(cd, data_name)
        client = Client(args, id, cd.train_dataset, cd.test_dataset, cd.val_dataset, cd.train_loader, cd.test_loader, cd.val_loader, cd.classnames, init_image_encoder, cls_head, data_name)
        clients.append(client)
        cls_heads.append(cls_head)
        few_shot_cds.append(few_shot_cd)
        del cd

    # generate global cls head
    server.generate_global_cls_head(cls_heads)

    # print("clients[0].model.keys():", clients[0].model.state_dict().keys())
    # print("name of the parameters in clients[0].model:", [k for k,_ in clients[0].model.named_parameters()])
    print("the parameters that require grad in clients[0].model:", [k for k,p in clients[0].model.named_parameters() if p.requires_grad]) # make sure only fine tune the local adapter

    # train and test clients
    total_test_time, total_train_time = 0, 0

    global_adapter = torch.load(f'../weights/{args.image_encoder_name}/{args.dataset}_sub{args.subset_size}_ours/client_0_global_adapter.pth', map_location=args.device)
    clients = send_adaptive_global_adapter(global_adapter, clients)
    clients = send_global_head(server.global_cls_head, clients)

    start_time = time.time()
    # few shot
    for id in range(len(clients)):
        clients[id].fine_tune(train_dataloader = few_shot_cds[id].train_loader)
    total_train_time += time.time() - start_time

    start_time = time.time()
    client_acc = []
    for id, client in enumerate(clients):
        accs = client.test_on_all_clients(clients)
        client_acc.append(accs)
    total_test_time += time.time() - start_time
    with open(f'./results/ours/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}_fewshot_{args.fewshot}_kd{args.kd_loss_weight}.json', 'a+') as f:
        json.dump({'round':-1, 'acc': client_acc, 'total_test_time': total_test_time, 'total_train_time': total_train_time}, f)
        f.write('\n')
    total_time_cost = total_test_time + total_train_time
    print(f'Train time cost: {total_train_time:.2f}s')
    print(f'Test time cost: {total_test_time:.2f}s')
    print(f'Total time cost: {total_time_cost:.2f}s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DomainFL')
    parser.add_argument('-d','--dataset', type=str, default='domainnet', help='Dataset name')
    parser.add_argument('-ss','--subset_size', type=int, default=100, help='Subset size')
    parser.add_argument('-m','--model', type=str, default='CLIP', help='Model name')
    parser.add_argument('-ien','--image_encoder_name', type=str, default='ViT-B-32', help='Image encoder name')
    parser.add_argument('-optim','--optimizer', type=str, default='AdamW', help='Optimizer name')
    parser.add_argument('-lr','--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-clip','--clip', type=float, default=1, help='Gradient clip')
    parser.add_argument('-bs','--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-le','--local_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-warm_up','--warm_up', type=int, default=10, help='Warm up epochs')
    parser.add_argument('-gr','--global_rounds', type=int, default=200, help='Number of global rounds')
    parser.add_argument('-device','--device', type=str, default='cuda', help='Device')
    parser.add_argument('-num_workers','--num_workers', type=int, default=12, help='Number of workers')
    parser.add_argument('-eval','--eval_interval', type=int, default=200, help='Log interval')
    parser.add_argument('-did','--device_id', type=str, default=0, help='Device ID')
    parser.add_argument('-seed','--seed', type=int, default=1, help='Seed')
    parser.add_argument('-rw','--regularization_weight', type=float, default=0, help='Regularization weight')
    parser.add_argument('-kdw','--kd_loss_weight', type=float, default=10, help='KD loss weight')
    parser.add_argument('-fewshot','--fewshot', type=int, default=1, help='Few shot')

    args = parser.parse_args()

    if args.device == 'cuda':
        args.device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    os.makedirs(f'./results/ours/', exist_ok=True)
    with open(f'./results/ours/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}_fewshot_{args.fewshot}_kd{args.kd_loss_weight}.json', 'w+') as f:
        json.dump(generate_json_config(args), f)
        f.write('\n')

    run(args)
