import json
import os
import copy
import time
import torch
import random
import argparse
from models.CLIP import *
from utils.get_data import data1
from utils.get_data import data2, source
from utils.get_data import get_data
from utils.data_utils import build_subset
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
    protos = protos.view(-1, batchsize, protos.shape[-1])
    labels = labels.view(-1, batchsize)
    protos = protos.to(torch.float32)
    # shuffle the training data
    for i, (proto, label) in enumerate(zip(protos, labels)):
        idx = torch.randperm(proto.shape[0])
        proto = proto[idx, :].view(proto.shape)
        label = label[idx].view(label.shape)
        protos[i] = proto
        labels[i] = label
    print('labels:', labels.shape)
    print('protos:', protos.shape)
    # generate training data
    training_data = []
    for i in range(protos.shape[0]):
        training_data.append((protos[i], labels[i]))
    # print('training_data:', training_data)
    return training_data

def generate_mean_protos_training_data(uploaded_protos, batchsize=10):
    classes = uploaded_protos[0].keys()
    _protos = defaultdict(list)
    for proto in uploaded_protos:
        for c in classes:
            _protos[c].append(proto[c])
    protos = []
    for c in _protos.keys():
        proto = 0 * _protos[c][0]
        for i in _protos[c]:
            proto += i
        protos.append(proto / len(_protos[c]))

    labels = list(_protos.keys())

    # generate batched data
    protos = torch.stack(protos)
    labels = torch.tensor(labels, dtype=torch.long)
    protos = protos.view(-1, batchsize, protos.shape[-1])
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
    return training_data

def send_adaptive_global_adapter(global_adapter, clientObjs):
    for client in clientObjs:
        client.model.base.global_adapter.load_state_dict(global_adapter.state_dict())
        client.model.base.adapter.load_state_dict(global_adapter.state_dict())
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
    training_data = generate_protos_training_data(uploaded_protos)
    global_adapter = server_adative_training(training_data, server)
    clientObjs = send_adaptive_global_adapter(global_adapter, clientObjs)
    clientObjs = send_global_head(server.global_cls_head, clientObjs)
    server.image_encoder.global_adapter.load_state_dict(global_adapter.state_dict())
    return clientObjs, server

def mean_proto_initialization(clientObjs, server):
    uploaded_protos = receive_protos(clientObjs)
    training_data = generate_mean_protos_training_data(uploaded_protos)
    global_adapter = server_adative_training(training_data, server)
    clientObjs = send_adaptive_global_adapter(global_adapter, clientObjs)
    clientObjs = send_global_head(server.global_cls_head, clientObjs)
    server.image_encoder.global_adapter.load_state_dict(global_adapter.state_dict())
    return clientObjs, server

def run(args):
    # initialize server
    server = Server(args)

    # set dataset
    dataset = globals()[args.dataset]

    # initialize clients
    # client image encoder is the same as the global image encoder
    clients = []
    cls_heads = []
    for id, data_name in enumerate(dataset):
        init_image_encoder = copy.deepcopy(server.image_encoder)
        cd = get_data(data_name, server.train_preprocess, server.val_preprocess, f'./{args.dataset}/{data_name}', args.batch_size, args.num_workers)
        cd = build_subset(cd, args.subset_size)
        cls_head = server.generate_cls_head(cd, data_name)
        client = Client(args, id, cd.train_dataset, cd.test_dataset, cd.train_loader, cd.test_loader, cd.classnames, init_image_encoder, cls_head, data_name)
        clients.append(client)
        cls_heads.append(cls_head)
        del cd

    # generate global cls head
    server.generate_global_cls_head(cls_heads)

    # fine tune clients
    for id in range(len(clients)):
        clients[id].fine_tune(global_round=0)

    clients_p, server_p = proto_initialization(clients, server)
    os.makedirs('../weights/test_proto', exist_ok=True)
    torch.save(server_p.image_encoder.global_adapter.state_dict(), f'../weights/test_proto/not_mean.pth')


    clients_m, server_m = mean_proto_initialization(clients, server)
    torch.save(server_m.image_encoder.global_adapter.state_dict(), f'../weights/test_proto/mean.pth')

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DomainFL')
    parser.add_argument('-d','--dataset', type=str, default='data1', help='Dataset name')
    parser.add_argument('-ss','--subset_size', type=int, default=100, help='Subset size')
    parser.add_argument('-m','--model', type=str, default='CLIP', help='Model name')
    parser.add_argument('-ien','--image_encoder_name', type=str, default='ViT-B-32', help='Image encoder name')
    parser.add_argument('-optim','--optimizer', type=str, default='AdamW', help='Optimizer name')
    parser.add_argument('-lr','--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-clip','--clip', type=float, default=1, help='Gradient clip')
    parser.add_argument('-bs','--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-le','--local_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-warm_up','--warm_up', type=int, default=10, help='Warm up epochs')
    parser.add_argument('-gr','--global_rounds', type=int, default=50, help='Number of global rounds')
    parser.add_argument('-device','--device', type=str, default='cuda', help='Device')
    parser.add_argument('-num_workers','--num_workers', type=int, default=12, help='Number of workers')
    parser.add_argument('-eval','--eval_interval', type=int, default=1, help='Log interval')
    parser.add_argument('-did','--device_id', type=str, default=0, help='Device ID')
    parser.add_argument('-seed','--seed', type=int, default=1, help='Seed')
    parser.add_argument('-rw','--regularization_weight', type=float, default=0, help='Regularization weight')
    parser.add_argument('-kdw','--kd_loss_weight', type=float, default=0, help='KD loss weight')

    args = parser.parse_args()

    if args.device == 'cuda':
        args.device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    run(args)
