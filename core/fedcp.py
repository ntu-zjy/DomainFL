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
from utils.serverCP import ServerCP  # 导入 ServerCP
from utils.clientCP import ClientCP  # 导入 ClientCP
from utils.json_utils import generate_json_config
import warnings
warnings.simplefilter("ignore")

torch.manual_seed(1)
torch.cuda.manual_seed(1) if torch.cuda.is_available() else None

def calculate_fedcp_weights(clients):
    total_train_num = 0
    num_list = []
    for c in clients:
        train_num = len(c.train_dataloader) * c.batch_size
        total_train_num += train_num
        num_list.append(train_num)
    weights = [num/total_train_num for num in num_list]
    return weights

def fedcp_aggregation(weights, clients, server):
    print("FedCP aggregation with weights: ", weights)
    # 收集所有客户端的本地质心和类别计数
    client_centroids = [client.local_centroids for client in clients]
    client_counts = [client.class_counts for client in clients]

    # 更新服务器的全局质心
    server.update_global_centroids(client_centroids, client_counts, weights)

    # 聚合模型参数（除了策略网络）
    # server receive the adapters from clients
    adapters = [c.model.base.adapter for c in clients]

    # fedavg aggregation
    server_global_adapter = copy.deepcopy(server.image_encoder.global_adapter)
    for param in server_global_adapter.parameters():
        param.data.zero_()

    for adapter in adapters:
        for w, global_param, param in zip(weights, server_global_adapter.parameters(), adapter.parameters()):
            global_param.data += w * param.data.clone()

    # set the global adapter to the server
    server.image_encoder.global_adapter.load_state_dict(server_global_adapter.state_dict())

    # 将更新后的模型参数和全局质心发送回客户端
    for client in clients:
        for param, global_param in zip(client.model.base.adapter.parameters(),
                                       server_global_adapter.parameters()):
            param.data = global_param.data.clone()
        client.global_centroids = server.global_centroids.clone()

    return clients, server

def run(args):
    # 初始化服务器
    server = ServerCP(args)

    # 设置数据集
    dataset = globals()[args.dataset]

    # 初始化客户端
    clients = []
    cls_heads = []
    for id, data_name in enumerate(dataset):
        init_image_encoder = copy.deepcopy(server.image_encoder)
        cd = get_data(data_name, server.train_preprocess, server.val_preprocess, args.batch_size, args.num_workers)
        cd = build_subset(cd, args.subset_size)
        cd = split_train_and_val(cd)
        cls_head = server.generate_cls_head(cd, data_name)
        client = ClientCP(
            args=args,
            id=id,
            train_dataset=cd.train_dataset,
            test_dataset=cd.test_dataset,
            val_dataset=cd.val_dataset,
            train_dataloader=cd.train_loader,
            test_dataloader=cd.test_loader,
            val_dataloader=cd.val_loader,
            classnames=cd.classnames,
            image_encoder=init_image_encoder,
            cls_head=cls_head,
            data_name=data_name
        )
        clients.append(client)
        cls_heads.append(cls_head)
        del cd

    # 生成全局分类头
    server.generate_global_cls_head(cls_heads)
    for client in clients:
        client.model.head.load_state_dict(server.global_cls_head.state_dict())

    print("The parameters that require grad in clients[0].model:",
          [k for k,p in clients[0].model.named_parameters() if p.requires_grad])

    # 训练和测试客户端
    total_test_time, total_train_time = 0, 0

    patience = 10
    best_loss = float('inf')
    counter = 0
    early_stop = False

    # 初始化服务器的全局质心
    num_classes = len(clients[0].classnames)
    server.initialize_centroids(num_classes)

    for r in range(args.global_rounds):
        print(f'==================== Round {r} ====================')
        # 计算验证损失
        val_loss = 0
        for client in clients:
            val_loss += client.cal_val_loss()
        print(f'Round {r} val loss: {val_loss:.4f}')

        if val_loss < best_loss and r != 0:
            best_loss = val_loss
            counter = 0
            print("Saving fine-tuned local models")
            for client in clients:
                client.save_adapter(args, algo='fedcp')
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
                acc, auc, f1, precision, recall = client.test()
                print(f'Client {id} [{client.data_name}] - Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}')
                client_acc.append(acc)

            avg_acc = sum(client_acc) / len(client_acc)
            with open(f'./results/fedcp/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'a+') as f:
                json.dump({'round': r, 'acc': client_acc, 'average_acc': avg_acc,
                           'total_test_time': total_test_time, 'total_train_time': total_train_time}, f)
                f.write('\n')

            if early_stop:
                break

        test_time = time.time() - start_time
        print(f'Round {r} test time cost: {test_time:.2f}s')

        start_time = time.time()
        # 本地训练
        for client in clients:
            client.fine_tune()
        train_time = time.time() - start_time
        print(f'Round {r} train time cost: {train_time:.2f}s')

        # 聚合模型和更新全局质心
        weights = calculate_fedcp_weights(clients)
        clients, server = fedcp_aggregation(weights, clients, server)

        total_test_time += test_time
        total_train_time += train_time

    total_time_cost = total_test_time + total_train_time

    print(f'Total time cost: {total_time_cost:.2f}s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedCP')
    parser.add_argument('-d', '--dataset', type=str, default='domainnet', help='Dataset name')
    parser.add_argument('-ss', '--subset_size', type=int, default=10, help='Subset size')
    parser.add_argument('-m', '--model', type=str, default='CLIP', help='Model name')
    parser.add_argument('-ien', '--image_encoder_name', type=str, default='ViT-B-32', help='Image encoder name')
    parser.add_argument('-optim', '--optimizer', type=str, default='AdamW', help='Optimizer name')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-clip', '--clip', type=float, default=1, help='Gradient clip')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-le', '--local_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-warm_up', '--warm_up', type=int, default=10, help='Warm up epochs')
    parser.add_argument('-gr', '--global_rounds', type=int, default=200, help='Number of global rounds')
    parser.add_argument('-device', '--device', type=str, default='cuda', help='Device')
    parser.add_argument('-num_workers', '--num_workers', type=int, default=12, help='Number of workers')
    parser.add_argument('-eval', '--eval_interval', type=int, default=200, help='Evaluation interval')
    parser.add_argument('-did', '--device_id', type=str, default=0, help='Device ID')
    parser.add_argument('-seed', '--seed', type=int, default=1, help='Seed')
    parser.add_argument('-mo', "--momentum", type=float, default=0.9, help='Momentum for centroids')
    parser.add_argument('-klw', "--kl_weight", type=float, default=1)
    parser.add_argument('--feature_dim', type=int, default=512, help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension in policy network')
    parser.add_argument('--centroid_weight', type=float, default=0.1, help='Weight for centroid alignment loss')
    parser.add_argument('--separation_weight', type=float, default=0.1, help='Weight for feature separation loss')
    parser.add_argument('--policy_lr', type=float, default=0.001, help='Learning rate for policy network')

    args = parser.parse_args()

    if args.device == 'cuda':
        args.device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    os.makedirs(f'./results/fedcp/', exist_ok=True)
    with open(f'./results/fedcp/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'w+') as f:
        json.dump(generate_json_config(args), f)
        f.write('\n')

    run(args)
