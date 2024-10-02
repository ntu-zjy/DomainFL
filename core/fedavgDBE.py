import json
import os
import copy
import time
import math
import torch
import random
import argparse
from models.CLIP import *
from utils.get_data import domainnet
from utils.get_data import get_data
from utils.data_utils import build_subset, split_train_and_val, build_subset_mixed, concat_datasets
from utils.server import Server
from utils.clientavgDBE import Client
from utils.json_utils import generate_json_config
import warnings
import numpy as np
from collections import defaultdict

warnings.simplefilter("ignore")

torch.manual_seed(1)
torch.cuda.manual_seed(1) if torch.cuda.is_available() else None

def init_global_mean(weights, clientObjs):
    for id in range(len(clientObjs)):
        clientObjs[id].fine_tune()

    global_mean = 0
    for id in range(len(clientObjs)):
        global_mean += weights[id] * clientObjs[id].running_mean

    for id in range(len(clientObjs)):
        clientObjs[id].global_mean = global_mean.data.clone()
    return clientObjs

def calculate_fedavg_weights(clients):
    total_train_num = 0
    num_list = []
    for c in clients:
        train_num = len(c.train_dataloader) * c.batch_size
        total_train_num += train_num
        num_list.append(train_num)
    weights = [num/total_train_num for num in num_list]
    return weights
def get_model_parameters_size(model):
    """Calculate the size of the model parameters in bytes."""
    parameters_size = 0
    for param_tensor in model.state_dict():
        parameters_size += model.state_dict()[param_tensor].numel() * model.state_dict()[param_tensor].element_size()
    return parameters_size

def fedavg(weights, clientObjs, server):
    print("FedAvg... with weights: ", weights)
    # Calculate the communication cost
    adapter_size = get_model_parameters_size(clientObjs[0].model.base.adapter)
    total_communication_cost = len(clientObjs) * adapter_size * 2  # Sending and receiving

    print(f"Communication cost for this round: {total_communication_cost / (1024 ** 2):.2f} MB")

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
    for id in range(len(clientObjs)):
        for param, global_param in zip(clientObjs[id].model.base.adapter.parameters(), server_global_adapter.parameters()):
            param.data = global_param.data.clone()

    return clientObjs, server, total_communication_cost

def send_global_head(global_cls_head, clientObjs):
    # Calculate communication cost for sending the global classification head
    head_size = get_model_parameters_size(global_cls_head)
    total_communication_cost = head_size * len(clientObjs)  # Sending to each client

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
    domains_num = len(dataset)
    split_num = args.split_num
    clients_ids = []
    for i in range(domains_num):
        for j in range(split_num):
            clients_ids.append([(i, j)])
    split_ratios = [i * (1.0 / split_num) for i in range(1, split_num)]
    # clients_ids = [[(0, 0)],[(0, 1)],[(0, 2)],[(1, 0)],[(1, 1)],[(1, 2)],
    #                [(2, 0)],[(2, 1)],[(2, 2)], [(3, 0)],[(3, 1)],[(3, 2)],
    #                [(4, 0)], [(4, 1)], [(4, 2)], [(5, 0)],[(5, 1)],[(5, 2)]]
    # [[(0, 0)], [(0, 1)], [(0, 2)], [(0, 3)], [(1, 0)], [(1, 1)], [(1, 2)], [(1, 3)],
    #  [(2, 0)], [(2, 1)], [(2, 2)], [(2, 3)], [(3, 0)], [(3, 1)], [(3, 2)], [(3, 3)],
    #  [(4, 0)], [(4, 1)], [(4, 2)], [(4, 3)], [(5, 0)], [(5, 1)], [(5, 2)], [(5, 3)]]
    # clients_ids = [[(0, 1), (1, 0)], [(1, 1), (2, 0)], [(2, 1), (3, 0)], [(3, 1), (4, 0)], [(4, 1), (5, 0)],
    #                    [(5, 1), (0, 0)]]
    clients_subsets = []
    for id, data_name in enumerate(dataset):
        cds = get_data(data_name, server.train_preprocess, server.val_preprocess, args.batch_size, args.num_workers)
        cds = build_subset_mixed(cds, args.subset_size, ratios=split_ratios)
        # 没划分
        new_cds = []
        for cd in cds:
            new_cd = split_train_and_val(cd)
            new_cds.append(new_cd)
        clients_subsets.append(new_cds)

    for ist in range(len(clients_ids)):
        data_name = dataset[clients_ids[ist][0][0]]
        sub = clients_subsets[clients_ids[ist][0][0]][clients_ids[ist][0][1]]
        # sub2 = clients_subsets[clients_ids[ist][1][0]][clients_ids[ist][1][1]]
        init_image_encoder = copy.deepcopy(server.image_encoder)
        cd = sub
        # cd = split_train_and_val(cd)
        cls_head = server.generate_cls_head(cd, data_name)
        client = Client(args, ist, cd.train_dataset, cd.test_dataset, cd.val_dataset, cd.train_loader, cd.test_loader,
                        cd.val_loader, cd.classnames, init_image_encoder, cls_head, data_name)
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

    print("the parameters that require grad in clients[0].model:", [k for k,p in clients[0].model.named_parameters() if p.requires_grad])

    # train and test clients
    total_test_time, total_train_time = 0, 0

    patience = 10
    best_loss = float('inf')
    counter = 0
    early_stop = False
    clients = init_global_mean(calculate_fedavg_weights(clients), clients)
    # args.global_rounds = 1
    for r in range(args.global_rounds):
        print(f'==================== Round {r} ====================')
        val_loss = 0
        for id in range(len(clients)):
            val_loss += clients[id].cal_val_loss()
        print(f'Round {r} val loss: {val_loss:.4f}')
        if val_loss < best_loss and r != 0:
            best_loss = val_loss
            counter = 0
            print("save finetuned local models")
            for client in clients:
                client.save_adapter(args, algo='fedavgDBE')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at round {r}')
                early_stop = True

        print(f'Round {r} best val loss: {best_loss:.4f}, counter: {counter}')

        start_time = time.time()
        if (r % args.eval_interval == 0 or r == args.global_rounds - 1) or early_stop or counter == 0:
            client_acc = []
            for id, client in enumerate(clients):
                accs = client.test_on_all_clients(clients)
                client_acc.append(accs)

            with open(f'./results/fedavgDBE/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}_split{args.split_num}.json', 'a+') as f:
                json.dump({'round':r, 'acc': client_acc, 'total_test_time': total_test_time, 'total_train_time': total_train_time}, f)
                f.write('\n')

            if early_stop:
                break

        test_time = time.time() - start_time
        print(f'Round {r} test time cost: {test_time:.2f}s')

        start_time = time.time()
        for id in range(len(clients)):
            clients[id].fine_tune()
        train_time = time.time() - start_time
        print(f'Round {r} train time cost: {train_time:.2f}s')

        weights = calculate_fedavg_weights(clients)
        clients, server, commu_cost = fedavg(weights, clients, server)

        communication_cost += commu_cost

        total_test_time += test_time
        total_train_time += train_time

    print(f"Communication cost of all: {communication_cost / (1024 ** 2):.2f} MB")
    print(f"Communication cost of all add Head: {(communication_cost + communication_cost_head)/ (1024 ** 2):.2f} MB")
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
    parser.add_argument('-mo', "--momentum", type=float, default=0.01)
    parser.add_argument('-klw', "--kl_weight", type=float, default=1)
    parser.add_argument('-split_num', '--split_num', type=int, default=3, help='Split number (Max 3, Min 2)')

    args = parser.parse_args()

    if args.device == 'cuda':
        args.device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    os.makedirs(f'./results/fedavgDBE/', exist_ok=True)
    with open(f'./results/fedavgDBE/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}_split{args.split_num}.json', 'w+') as f:
        json.dump(generate_json_config(args), f)
        f.write('\n')

    run(args)
