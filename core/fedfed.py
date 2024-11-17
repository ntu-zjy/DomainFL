import json
import os
import copy
import time
import torch
import argparse
from models.CLIP import *
from utils.get_data import domainnet
from utils.get_data import get_data
from utils.data_utils import build_subset, split_train_and_val
from utils.serverfed import Server
from utils.clientfed import Client
from utils.json_utils import generate_json_config
import warnings

warnings.simplefilter("ignore")

torch.manual_seed(1)
torch.cuda.manual_seed(1) if torch.cuda.is_available() else None

def send_global_head(global_cls_head, clientObjs):
    for client in clientObjs:
        client.model.head.load_state_dict(global_cls_head.state_dict())
    return clientObjs

def init_clients_and_server(args, server, dataset):
    """初始化客户端和服务器"""
    clients = []
    cls_heads = []

    for id, data_name in enumerate(dataset):
        init_image_encoder = copy.deepcopy(server.image_encoder)
        cd = get_data(data_name, server.train_preprocess, server.val_preprocess,
                      args.batch_size, args.num_workers)
        cd = build_subset(cd, args.subset_size)
        cd = split_train_and_val(cd)

        cls_head = server.generate_cls_head(cd, data_name)
        client = Client(args, id, cd.train_dataset, cd.test_dataset, cd.val_dataset,
                        cd.train_loader, cd.test_loader, cd.val_loader,
                        cd.classnames, init_image_encoder, cls_head, data_name)

        clients.append(client)
        cls_heads.append(cls_head)
        del cd

    # 生成全局分类头
    server.generate_global_cls_head(cls_heads)
    clients = send_global_head(server.global_cls_head, clients)

    return clients

def calculate_fedavg_weights(clients):
    total_train_num = 0
    num_list = []
    for c in clients:
        train_num = len(c.train_dataloader) * c.batch_size
        total_train_num += train_num
        num_list.append(train_num)
    weights = [num/total_train_num for num in num_list]
    return weights

def init_global_mean(weights, clientObjs):
    for id in range(len(clientObjs)):
        clientObjs[id].fine_tune()

    global_mean = 0
    for id in range(len(clientObjs)):
        global_mean += weights[id] * clientObjs[id].running_mean

    for id in range(len(clientObjs)):
        clientObjs[id].global_mean = global_mean.data.clone()
    return clientObjs

def train_fedfed(args, clients, server):
    """FedFed训练过程"""
    best_loss = float('inf')
    patience = args.patience if hasattr(args, 'patience') else 10
    counter = 0
    total_test_time = 0
    total_train_time = 0

    # 初始化全局特征
    clients = init_global_mean(calculate_fedavg_weights(clients), clients)

    for r in range(args.global_rounds):
        print(f'==================== Round {r} ====================')

        # 1. 评估阶段
        start_time = time.time()
        val_loss = sum(client.cal_val_loss() for client in clients)
        print(f'Round {r} val loss: {val_loss:.4f}')

        # 早停检查
        if val_loss < best_loss and r != 0:
            best_loss = val_loss
            counter = 0
            print("保存最佳模型")
            for client in clients:
                client.save_adapter(args, algo='fedfed')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at round {r}')
                break

        # 2. 测试阶段
        if (r % args.eval_interval == 0 or r == args.global_rounds - 1) or counter == 0:
            client_acc = []
            for client in clients:
                accs = client.test_on_all_clients(clients)
                client_acc.append(accs)

            # 保存结果
            with open(f'./results/fedfed/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json',
                      'a+') as f:
                json.dump({
                    'round': r,
                    'acc': client_acc,
                    'total_test_time': total_test_time,
                    'total_train_time': total_train_time
                }, f)
                f.write('\n')

        test_time = time.time() - start_time
        total_test_time += test_time

        # 3. 训练阶段
        start_time = time.time()

        # 本地训练
        for client in clients:
            client.fine_tune()
            client.update_feature_importance()  # 更新特征重要性

        # 特征聚合和分发
        server.update_global_model(clients)

        train_time = time.time() - start_time
        total_train_time += train_time

        print(f'Round {r} - Train time: {train_time:.2f}s, Test time: {test_time:.2f}s')
        print(f'Best val loss: {best_loss:.4f}, Counter: {counter}')

    return total_test_time + total_train_time


def run(args):
    # 初始化服务器
    server = Server(args)
    dataset = globals()[args.dataset]

    # 初始化客户端
    clients = init_clients_and_server(args, server, dataset)

    # FedFed训练
    total_time = train_fedfed(args, clients, server)
    print(f'Total time cost: {total_time:.2f}s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedFed Training')
    # 保留原有参数
    parser.add_argument('-d', '--dataset', type=str, default='domainnet')
    parser.add_argument('-ss', '--subset_size', type=int, default=10)
    parser.add_argument('-m', '--model', type=str, default='CLIP')
    parser.add_argument('-ien', '--image_encoder_name', type=str, default='ViT-B-32')
    parser.add_argument('-optim', '--optimizer', type=str, default='AdamW')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3)
    parser.add_argument('-clip', '--clip', type=float, default=1)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-le', '--local_epochs', type=int, default=1)
    parser.add_argument('-warm_up', '--warm_up', type=int, default=10)
    parser.add_argument('-gr', '--global_rounds', type=int, default=200)
    parser.add_argument('-device', '--device', type=str, default='cuda')
    parser.add_argument('-num_workers', '--num_workers', type=int, default=12)
    parser.add_argument('-eval', '--eval_interval', type=int, default=200)
    parser.add_argument('-did', '--device_id', type=str, default=0)
    parser.add_argument('-seed', '--seed', type=int, default=1)
    parser.add_argument('-mo', "--momentum", type=float, default=0.01)
    parser.add_argument('-klw', "--kl_weight", type=float, default=1)

    # 添加FedFed特定参数
    parser.add_argument('-dt', '--distill_temp', type=float, default=1.0,
                        help='Knowledge distillation temperature')
    parser.add_argument('-dw', '--distill_weight', type=float, default=0.1,
                        help='Knowledge distillation weight')
    parser.add_argument('-st', '--sensitive_threshold', type=float, default=0.5,
                        help='Threshold for sensitive feature selection')
    parser.add_argument('-patience', '--patience', type=int, default=10,
                        help='Patience for early stopping')

    args = parser.parse_args()

    # 设备配置
    if args.device == 'cuda':
        args.device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    # 创建结果目录
    os.makedirs(f'./results/fedfed/', exist_ok=True)
    with open(f'./results/fedfed/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'w+') as f:
        json.dump(generate_json_config(args), f)
        f.write('\n')

    run(args)
