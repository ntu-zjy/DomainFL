import json
import os
import copy
import time
import torch
import argparse
from models.CLIP import *
from utils.get_data import data1
from utils.get_data import data2, source, target
from utils.get_data import get_data
from utils.data_utils import build_subset, local_adaptation_subset_trainloader
from utils.server import Server
from utils.client import Client
from utils.json_utils import generate_json_config
import warnings
import random
import numpy as np
warnings.simplefilter("ignore")
torch.backends.cudnn.deterministic = True
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1) if torch.cuda.is_available() else None

def local_adaptation(clientObj, adapt_trainloader=None):
    adapt_trainloader = local_adaptation_subset_trainloader(clientObj.train_dataset, clientObj.train_dataloader, 10)

    clientObj.local_adaptation(adapt_trainloader)
    return clientObj

def calculate_fedts_weights(clients):
    # every client use the same weight
    weights = [1/len(clients) for c in clients]
    return weights

def fedts(weights, clientObjs, server):
    print("fedts... with weights: ", weights)
    # server receive the adapters from clients
    adapters = [c.model.base.adapter for c in clientObjs]

    # fedts aggregation
    server_global_adapter = copy.deepcopy(server.image_encoder.global_adapter)
    for param in server_global_adapter.parameters():
        param.data.zero_()

    for adapter in adapters:
        for w, global_param, param in zip(weights, server_global_adapter.parameters(), adapter.parameters()):
            global_param.data += w * param.data.clone()

    print("global data:", server_global_adapter.state_dict()["fc.2.weight"])
    # set the global adapter to the server
    server.image_encoder.global_adapter.load_state_dict(server_global_adapter.state_dict())

    # send the global adapter back to the clients
    # param will be covered as global param
    for id in range(len(clientObjs)):
        for param, global_param in zip(clientObjs[id].model.base.adapter.parameters(), server_global_adapter.parameters()):
            param.data = global_param.data.clone()

    return clientObjs, server


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
        client = Client(args, id, cd.train_dataset, cd.test_dataset, cd.train_loader, cd.test_loader, cd.classnames, init_image_encoder, cls_head, data_name)
        clients.append(client)
        del cd

    # print("clients[0].model.keys():", clients[0].model.state_dict().keys())
    # print("name of the parameters in clients[0].model:", [k for k,_ in clients[0].model.named_parameters()])
    print("the parameters that require grad in clients[0].model:", [k for k,p in clients[0].model.named_parameters() if p.requires_grad]) # make sure only fine tune the local adapter

    # train and test clients
    total_test_time, total_train_time, total_adapt_time = 0, 0, 0
    for r in range(args.global_rounds):
        print(f'==================== Round {r} ====================')
        start_time = time.time()
        # fine tune clients
        for id in range(len(clients)):
            clients[id].fine_tune()
        train_time = time.time() - start_time
        print(f'Round {r} train time cost: {train_time:.2f}s')

        # after fine tuning clients, we need to aggregate the adapters
        weights = calculate_fedts_weights(clients)
        # fedts algorithm
        clients, server = fedts(weights, clients, server)

        start_time = time.time()
        # after the aggregation, we need to do the local adaptation
        print("local adaptation...")
        # print("before local adaptation, clients[0].model.base.adapter.fc.2.weight:", clients[0].model.base.adapter.state_dict()["fc.2.weight"])
        for id, client in enumerate(clients):
            client = local_adaptation(client)
        # print("after local adaptation, clients[0].model.base.adapter.fc.2.weight:", clients[0].model.base.adapter.state_dict()["fc.2.weight"])
        print("local adaptation done")
        adapt_time = time.time() - start_time
        print(f'Round {r} adapt time cost: {adapt_time:.2f}s')

        start_time = time.time()
        if (r % args.eval_interval == 0 or r == args.global_rounds - 1) and r!=0:
            client_acc = []
            for id, client in enumerate(clients):
                accs = client.test_on_all_clients(clients)
                client_acc.append(accs)

            with open(f'./results/fedts_la_DG/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'a+') as f:
                json.dump({'round':r, 'acc': client_acc, 'total_test_time': total_test_time, 'total_train_time': total_train_time, 'total_adapt_time': total_adapt_time}, f)
                f.write('\n')

        test_time = time.time() - start_time
        print(f'Round {r} test time cost: {test_time:.2f}s')

        total_test_time += test_time
        total_train_time += train_time
        total_adapt_time += adapt_time

    total_time_cost = total_test_time + total_train_time
    print("save finetuned local models")
    for client in clients:
        client.save_adapter(args, algo='fedts_la_DG')
    print(f'Total time cost: {total_time_cost:.2f}s')

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

    args = parser.parse_args()

    if args.device == 'cuda':
        args.device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    os.makedirs(f'./results/fedts_la_DG/', exist_ok=True)
    with open(f'./results/fedts_la_DG/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'w+') as f:
        json.dump(generate_json_config(args), f)
        f.write('\n')

    run(args)
