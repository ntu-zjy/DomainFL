import json
import os
import copy
import time
import torch
import argparse
from utils.get_data import data1, data2
from utils.get_data import get_data
from utils.data_utils import build_subset, concat_datasets
from utils.server import Server
from utils.client import Client
from utils.json_utils import generate_json_config
from tqdm import tqdm
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
    cds = []
    cls_heads = []
    for id, data_name in enumerate(dataset):
        init_image_encoder = copy.deepcopy(server.image_encoder)
        cd = get_data(data_name, server.train_preprocess, server.val_preprocess, f'./{args.dataset}/{data_name}', args.batch_size, args.num_workers)
        cd = build_subset(cd, 100)
        cds.append(cd)
        cls_head = server.generate_cls_head(cd, data_name)
        cls_heads.append(cls_head)

    print("ensemble the classification heads...")
    mean_cls_head = copy.deepcopy(cls_head)
    for i in range(len(cls_heads) - 1):
        for param, other_param in zip(mean_cls_head.parameters(), cls_heads[i].parameters()):
            param.data += other_param.data
    for param in mean_cls_head.parameters():
        param.data /= len(cls_heads)
    conds = concat_datasets(cds)
    print("define the client...")
    client = Client(args, id, conds.train_dataset, conds.test_dataset, conds.train_loader, conds.test_loader, conds.classnames, init_image_encoder, mean_cls_head, data_name)

    # print("clients[0].model.keys():", clients[0].model.state_dict().keys())
    # print("name of the parameters in client.model:", [k for k,_ in client.model.named_parameters()])
    print("the parameters that require grad in client.model:", [k for k,p in client.model.named_parameters() if p.requires_grad]) # make sure only fine tune the local adapter

    # train and test clients
    # alpha_list, beta_list = [], []
    total_test_time, total_train_time = 0, 0
    for r in range(args.global_rounds):
        print(f'==================== Round {r} ====================')
        start_time = time.time()
        if r % args.eval_interval == 0 or r == args.global_rounds - 1:
            accs = []
            for cd in cds:
                stat = client.test(cd.test_loader)
                accs.append(stat[0])
            print(f'Round {r} accs: {accs}')

        test_time = time.time() - start_time
        print(f'Round {r} test time cost: {test_time:.2f}s')
        start_time = time.time()
        client.fine_tune(centralized=True)
        train_time = time.time() - start_time
        print(f'Round {r} train time cost: {train_time:.2f}s')
        total_test_time += test_time
        total_train_time += train_time
        with open(f'./results/centrailized/{args.image_encoder_name}_{args.dataset}.json', 'a+') as f:
            json.dump\
                ({'round':r, 'acc': accs, 'total_test_time': total_test_time, 'total_train_time': total_train_time}, f)
            f.write('\n')
        # save the adapter
        torch.save(client.model.base.adapter.state_dict(), f'../weights/centralized_adapter.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DomainFL')
    parser.add_argument('-d','--dataset', type=str, default='data1', help='Dataset name')
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
    os.makedirs(f'./results/centrailized/', exist_ok=True)
    with open(f'./results/centrailized/{args.image_encoder_name}_{args.dataset}.json', 'w+') as f:
        json.dump(generate_json_config(args), f)
        f.write('\n')
    run(args)