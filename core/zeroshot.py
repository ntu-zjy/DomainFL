import json
import os
import copy
import time
import torch
import argparse
from utils.get_data import domainnet, adaptiope
from utils.get_data import get_data
from utils.data_utils import build_subset
from utils.server import Server
from utils.client import Client
from tqdm import tqdm
from utils.json_utils import generate_json_config
import warnings
warnings.simplefilter("ignore")

torch.manual_seed(1)
torch.cuda.manual_seed(1) if torch.cuda.is_available() else None


def run(args):
    # initialize server
    server = Server(args, zeroshot=True)

    # set dataset
    dataset = globals()[args.dataset]

    # initialize clients
    # client image encoder is the same as the global image encoder
    clients = []
    for id, data_name in enumerate(dataset):
        init_image_encoder = copy.deepcopy(server.image_encoder)
        cd = get_data(data_name, server.train_preprocess, server.val_preprocess, args.batch_size, args.num_workers)
        print(f'Client {id} [{data_name}] has {len(cd.train_dataset)} samples')
        cd = build_subset(cd, args.subset_size) if args.dataset == 'data' else cd
        print(f'Subset Client {id} [{data_name}] has {len(cd.train_dataset)} samples')
        cls_head = server.generate_cls_head(cd, data_name)
        client = Client(args, id, cd.train_dataset, cd.test_dataset, None, cd.train_loader, cd.test_loader, None, cd.classnames, init_image_encoder, cls_head, data_name)
        clients.append(client)
        del cd

    # print("name of the parameters in clients[0].model:", [k for k,_ in clients[0].model.named_parameters()])
    print("the parameters that require grad in clients[0].model:", [k for k,p in clients[0].model.named_parameters() if p.requires_grad]) # make sure only fine tune the local adapter

    # train and test clients
    total_test_time, total_train_time = 0, 0
    start_time = time.time()
    client_acc = []

    for id, client in enumerate(clients):
        accs = client.test_on_all_clients(clients)
        client_acc.append(accs)
    print("test data number of each client:", [len(client.test_dataset) for client in clients])
    with open(f'./results/zeroshot/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'a+') as f:
        json.dump\
            ({'acc': client_acc, 'total_test_time': total_test_time, 'total_train_time': total_train_time}, f)
        f.write('\n')

    test_time = time.time() - start_time
    print(f'test time cost: {test_time:.2f}s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DomainFL')
    parser.add_argument('-d','--dataset', type=str, default='domainnet', help='Dataset name')
    parser.add_argument('-ss','--subset_size', type=int, default=100, help='Subset size')
    parser.add_argument('-m','--model', type=str, default='CLIP', help='Model name')
    parser.add_argument('-ien','--image_encoder_name', type=str, default='ViT-B-32', help='Image encoder name')
    parser.add_argument('-optim','--optimizer', type=str, default='AdamW', help='Optimizer name')
    parser.add_argument('-lr','--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-clip','--clip', type=float, default=1, help='Gradient clip')
    parser.add_argument('-bs','--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('-le','--local_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-warm_up','--warm_up', type=int, default=20, help='Warm up epochs')
    parser.add_argument('-gr','--global_rounds', type=int, default=50, help='Number of global rounds')
    parser.add_argument('-device','--device', type=str, default='cuda', help='Device')
    parser.add_argument('-num_workers','--num_workers', type=int, default=12, help='Number of workers')
    parser.add_argument('-eval','--eval_interval', type=int, default=100, help='Log interval')
    parser.add_argument('-did','--device_id', type=str, default=0, help='Device ID')
    parser.add_argument('-seed','--seed', type=int, default=1, help='Seed')

    args = parser.parse_args()

    if args.device == 'cuda':
        args.device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    os.makedirs(f'./results/zeroshot/', exist_ok=True)
    with open(f'./results/zeroshot/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'w+') as f:
        json.dump(generate_json_config(args), f)
        f.write('\n')

    run(args)