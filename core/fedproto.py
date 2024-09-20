import json
import os
import copy
import time
import torch
import argparse
from models.CLIP import *
from utils.get_data import domainnet, adaptiope, PACS
from utils.get_data import get_data
from utils.data_utils import build_subset, split_train_and_val
from utils.server import Server
from utils.clientproto import ClientProto
from utils.json_utils import generate_json_config
import warnings
from collections import defaultdict
warnings.simplefilter("ignore")

torch.manual_seed(1)
torch.cuda.manual_seed(1) if torch.cuda.is_available() else None

def get_model_parameters_size(model):
    parameters_size = 0
    for param_tensor in model.state_dict():
        parameters_size += model.state_dict()[param_tensor].numel() * model.state_dict()[param_tensor].element_size()
    return parameters_size

def get_protos_size(protos):
    total_size = 0
    total_size += protos.numel() * protos.element_size()
    return total_size

def send_protos(global_protos, clients):
    # Calculate communication cost for sending prototypes
    proto_size = sum(get_protos_size(proto) for proto in global_protos.values())
    total_communication_cost = proto_size * len(clients) * 2

    print(f"Communication cost for sending prototypes: {total_communication_cost / (1024 ** 2):.2f} MB")

    for client in clients:
        client.set_protos(global_protos)
    return clients, total_communication_cost

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

def calculate_fedavg_weights(clients):
    total_train_num = 0
    num_list = []
    for c in clients:
        train_num = len(c.train_dataloader) * c.batch_size
        total_train_num += train_num
        num_list.append(train_num)
    weights = [num/total_train_num for num in num_list]
    return weights

def fedproto(clientObjs):
    uploaded_protos = receive_protos(clientObjs)
    global_protos = proto_aggregation(uploaded_protos)
    clientObjs, commu_cost = send_protos(global_protos, clientObjs)

    return clientObjs, commu_cost

def send_global_head(global_cls_head, clientObjs):
    head_size = get_model_parameters_size(global_cls_head)
    total_communication_cost = head_size * len(clientObjs)

    print(f"Communication cost for sending global head: {total_communication_cost / (1024 ** 2):.2f} MB")

    for client in clientObjs:
        client.model.head.load_state_dict(global_cls_head.state_dict())
    return clientObjs, total_communication_cost

def run(args):
    # initialize server
    server = Server(args)

    # set dataset
    dataset = globals()[args.dataset]

    # initialize clients
    clients = []
    cls_heads = []
    for id, data_name in enumerate(dataset):
        init_image_encoder = copy.deepcopy(server.image_encoder)
        cd = get_data(data_name, server.train_preprocess, server.val_preprocess, args.batch_size, args.num_workers)
        cd = build_subset(cd, args.subset_size)
        cd = split_train_and_val(cd)
        cls_head = server.generate_cls_head(cd, data_name)
        client = ClientProto(args, id, cd.train_dataset, cd.test_dataset, cd.val_dataset, cd.train_loader, cd.test_loader, cd.val_loader, cd.classnames, init_image_encoder, cls_head, data_name)
        clients.append(client)
        cls_heads.append(cls_head)
        del cd

    # initialize communication cost
    communication_cost = 0
    communication_cost_head = 0

    # generate global cls head
    server.generate_global_cls_head(cls_heads)
    clients, commu_cost_head = send_global_head(server.global_cls_head, clients)
    communication_cost_head += commu_cost_head
    print(f"Communication cost of global head: {communication_cost_head / (1024 ** 2):.2f} MB")

    # print("clients[0].model.keys():", clients[0].model.state_dict().keys())
    # print("name of the parameters in clients[0].model:", [k for k,_ in clients[0].model.named_parameters()])
    print("the parameters that require grad in clients[0].model:", [k for k,p in clients[0].model.named_parameters() if p.requires_grad]) # make sure only fine tune the local adapter

    global_protos = [None for _ in range(args.subset_size)]
    total_test_time, total_train_time = 0, 0

    args.global_rounds = 1
    for r in range(args.global_rounds):
        print(f'==================== Round {r} ====================')
        # fine tune clients
        for id in range(len(clients)):
            clients[id].fine_tune()

        clients, commu_cost = fedproto(clients)
        communication_cost += commu_cost

    print(f"Communication cost of all: {communication_cost / (1024 ** 2):.2f} MB")
    print(f"Communication cost of all add Head: {(communication_cost + communication_cost_head)/ (1024 ** 2):.2f} MB")
    with open(f'./costs/fedproto/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'a+') as f:
            json.dump({"Communication_cost": f"{(communication_cost)/ (1024 ** 2):.4f}"}, f)
            f.write('\n')

    total_time_cost = total_test_time + total_train_time
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
    parser.add_argument('-lam', "--lamda", type=float, default=50, help="Regularization weight")

    args = parser.parse_args()

    if args.device == 'cuda':
        args.device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    os.makedirs(f'./costs/fedproto/', exist_ok=True)
    with open(f'./costs/fedproto/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'w+') as f:
        json.dump(generate_json_config(args), f)
        f.write('\n')

    run(args)
