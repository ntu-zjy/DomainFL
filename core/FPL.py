import json
import os
import copy
import time
import torch
import argparse
from models.CLIP import *
from utils.get_data import domainnet, adaptiope
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

from utils.finch import FINCH

def get_protos_size(protos):
    """
    Calculate the total size of the prototypes in bytes.
    Args:
        protos (dict): A dictionary where keys are labels and values are prototype tensors.
    Returns:
        int: Total size of the prototypes in bytes.
    """
    total_size = 0
    for label, proto in protos.items():
        total_size += proto.numel() * proto.element_size()
    return total_size

def send_protos(global_protos, clients):
    # Calculate communication cost for sending prototypes
    proto_size = sum(get_protos_size(proto) for proto in global_protos.values())
    total_communication_cost = proto_size * len(clients)

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
            agg_protos_label[label].extend(local_protos[label])

    # 使用 FINCH 聚类算法
    global_protos = {}
    for label, proto_list in agg_protos_label.items():
        if len(proto_list) > 1:
            data = torch.stack(proto_list).cpu().numpy()
            c, num_clust, req_c = FINCH(data)
            # 使用第一个聚类结果
            cluster_indices = c[:, 0]
            unique_clusters = np.unique(cluster_indices)
            cluster_means = [data[cluster_indices == uc].mean(axis=0) for uc in unique_clusters]
            global_protos[label] = torch.tensor(cluster_means).to(proto_list[0].device)
        else:
            global_protos[label] = torch.stack(proto_list)

    return global_protos

def calculate_fedavg_weights(clients):
    total_train_num = 0
    num_list = []
    for c in clients:
        train_num = len(c.train_dataloader) * c.batch_size
        total_train_num += train_num
        num_list.append(train_num)
    weights = [num/total_train_num for num in num_list]
    return weights

def fpl(clientObjs):
    uploaded_protos = receive_protos(clientObjs)
    global_protos = proto_aggregation(uploaded_protos)
    clientObjs, commu_cost = send_protos(global_protos, clientObjs)

    return clientObjs, commu_cost


def send_global_head(global_cls_head, clientObjs):
    head_size = get_protos_size(global_cls_head)
    total_communication_cost = head_size * len(clientObjs)

    print(f"Communication cost for sending global head: {total_communication_cost / (1024 ** 2):.2f} MB")

    for client in clientObjs:
        client.model.head.load_state_dict(global_cls_head.state_dict())
    return clientObjs, total_communication_cost

def run(args):
    # 初始化服务器
    server = Server(args)

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
        client = ClientProto(args, id, cd.train_dataset, cd.test_dataset, cd.val_dataset, cd.train_loader, cd.test_loader, cd.val_loader, cd.classnames, init_image_encoder, cls_head, data_name)
        clients.append(client)
        cls_heads.append(cls_head)
        del cd

    # 初始化通信成本
    communication_cost = 0
    communication_cost_head = 0

    # 生成全局分类头
    server.generate_global_cls_head(cls_heads)
    clients, commu_cost_head = send_global_head(server.global_cls_head, clients)
    communication_cost_head += commu_cost_head

    print("the parameters that require grad in clients[0].model:", [k for k,p in clients[0].model.named_parameters() if p.requires_grad])

    global_protos = [None for _ in range(args.subset_size)]
    total_test_time, total_train_time = 0, 0

    patience = 10
    best_loss = float('inf')
    counter = 0
    early_stop = False

    for r in range(args.global_rounds):
        print(f'==================== Round {r} ====================')
        val_loss = 0
        for id in range(len(clients)):
            val_loss += clients[id].cal_val_loss()
        print(f'Round {r} val loss: {val_loss:.4f}')
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            print("save finetuned local models")
            for client in clients:
                client.save_adapter(args, algo='fpl')
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

            with open(f'./results/fpl/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'a+') as f:
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

        clients, commu_cost = fpl(clients)
        communication_cost += commu_cost

        total_test_time += test_time
        total_train_time += train_time

    print(f"Communication cost of all: {communication_cost / (1024 ** 2):.2f} MB")
    print(f"Communication cost of all add Head: {(communication_cost + communication_cost_head) / (1024 ** 2):.2f} MB")
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

    os.makedirs(f'./results/fedproto/', exist_ok=True)
    with open(f'./results/fedproto/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'w+') as f:
        json.dump(generate_json_config(args), f)
        f.write('\n')

    run(args)
