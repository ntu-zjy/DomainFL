import json
import os
import copy
import time
import torch
import argparse
from models.CLIP import *
from utils.get_data import data1
from utils.get_data import data2
from utils.get_data import get_data
from utils.data_utils import build_subset, concat_test_datasets, generate_domain_label, split_dataloader_by_labels
from utils.server import Server
from utils.client import Client
from utils.json_utils import generate_json_config
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix

import warnings
warnings.simplefilter("ignore")

torch.manual_seed(1)
torch.cuda.manual_seed(1) if torch.cuda.is_available() else None

def classify_with_one_class_svm(train_features, test_features, domain_label):
    # 数据归一化
    scaler = StandardScaler()
    train_features_norm = scaler.fit_transform(train_features)
    test_features_norm = scaler.transform(test_features)

    # 初始化One-Class SVM
    oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.5)  # gamma和nu是重要的参数，需要根据数据调整
    oc_svm.fit(train_features_norm)  # 使用训练数据训练模型

    # 预测测试数据
    predicted_labels = oc_svm.predict(test_features_norm)
    # OneClassSVM标签是1和-1，1表示正常，-1表示异常

    # convert -1 to 0
    predicted_labels = (predicted_labels + 1) / 2

    acc = np.mean(predicted_labels == domain_label)
    cm = confusion_matrix(domain_label, predicted_labels)
    return predicted_labels, acc, cm

def classify_with_knn(train_features, test_features, domain_label):
    # 数据归一化
    scaler = StandardScaler()
    train_features_norm = scaler.fit_transform(train_features)
    test_features_norm = scaler.transform(test_features)

    # 应用PCA进行降维
    pca = PCA(n_components=128)
    train_features_reduced = pca.fit_transform(train_features_norm)
    test_features_reduced = pca.transform(test_features_norm)
    print('train_features_reduced shape:', train_features_reduced.shape)

    # 使用最近邻进行分类
    nbrs = NearestNeighbors(n_neighbors=20)
    nbrs.fit(train_features_norm)
    distances, indices = nbrs.kneighbors(test_features_norm)
    # predicted_labels = distances.flatten() <= np.mean(distances)  # 使用中位数作为阈值
    # predicted_labels = predicted_labels.astype(int)

    # acc = np.mean(predicted_labels == domain_label)

    # 使用多数投票规则来确定预测标签
    # 假设train_features的类别标签存储在train_labels中
    train_labels = np.ones([1 if i < len(train_features) / 2 else 0 for i in range(len(train_features))])  # 示例生成标签
    # train的标签全部为1

    votes = train_labels[indices]  # 根据索引获取邻居的标签
    predicted_labels = np.array([np.bincount(vote).argmax() for vote in votes])  # 对每个测试样本，计算最频繁的类别

    acc = np.mean(predicted_labels == domain_label)
    cm = confusion_matrix(domain_label, predicted_labels)
    return predicted_labels, acc, cm

def classify_with_dbscan(train_features, test_features, domain_label, eps=0.3, min_samples=5):
    # 数据归一化
    scaler = StandardScaler()
    train_features_norm = scaler.fit_transform(train_features)
    test_features_norm = scaler.transform(test_features)

    # 使用DBSCAN进行测试数据的聚类
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(test_features_norm)
    labels = db.labels_

    # 将所有-1标签（噪声）转化为0
    labels[labels == -1] = 0

    # 使用最近邻来确定哪一类和train_features更接近
    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(train_features_norm)

    # 计算每个类别的中心点
    class_0_center = np.mean(test_features_norm[labels == 0], axis=0, keepdims=True)
    class_1_center = np.mean(test_features_norm[labels == 1], axis=0, keepdims=True)

    # 找到每个类别中心的最近train_features
    distances_0, _ = nbrs.kneighbors(class_0_center)
    distances_1, _ = nbrs.kneighbors(class_1_center)

    # 确定哪个类别更接近train_features
    if distances_0 < distances_1:
        labels[labels == 1] = 2  # 临时标记为2
        labels[labels == 0] = 1  # 0类更接近，把原1类标记为2，原0类标记为1
        labels[labels == 2] = 0  # 把临时标记2转换为0

    predicted_labels = labels

    # 计算准确率和混淆矩阵
    acc = np.mean(predicted_labels == domain_label)
    cm = confusion_matrix(domain_label, predicted_labels)

    return predicted_labels, acc, cm

def post_process_dbscan_clusters(labels, features):
    from collections import Counter

    # 计数每个聚类中的点
    cluster_counts = Counter(labels)

    # 将-1的点改为0


    # 选择最大的两个聚类
    if len(cluster_counts) >= 2:
        main_clusters = cluster_counts.most_common(2)
        main_clusters = [item[0] for item in main_clusters]
    else:
        # 如果不足两个聚类，则认为所有非噪声点属于一个聚类
        main_clusters = cluster_counts.most_common(1)
        main_clusters = [item[0] for item in main_clusters] * 2  # 重复使用同一聚类

    # 为这两个聚类分配标签 0 和 1
    cluster_label_map = {main_clusters[0]: 0, main_clusters[1]: 1}

    # 使用最近邻重新分配剩余点的聚类标签
    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(features[labels == main_clusters[0]], features[labels == main_clusters[1]])
    distances, indices = nbrs.kneighbors(features[labels == -1])

    # 映射剩余点到最近的主聚类
    for idx, point in enumerate(labels == -1):
        if point:
            nearest_cluster = main_clusters[indices[idx][0]]
            labels[idx] = cluster_label_map[nearest_cluster]

    # 映射主聚类的所有点
    for label in np.unique(labels):
        if label in cluster_label_map:
            labels[labels == label] = cluster_label_map[label]

    return labels

def classify_with_pca_and_knn(train_features, test_features, domain_label):
    # 数据归一化
    scaler = StandardScaler()
    train_features_norm = scaler.fit_transform(train_features)
    test_features_norm = scaler.transform(test_features)

    # 应用PCA进行降维
    pca = PCA(n_components=256)
    train_features_reduced = pca.fit_transform(train_features_norm)
    test_features_reduced = pca.transform(test_features_norm)
    print('train_features_reduced shape:', train_features_reduced.shape)

    # 使用最近邻进行分类
    nbrs = NearestNeighbors(n_neighbors=20)
    nbrs.fit(train_features_reduced)
    distances, indices = nbrs.kneighbors(test_features_reduced)
    # predicted_labels = distances.flatten() <= np.mean(distances)  # 使用中位数作为阈值
    # predicted_labels = predicted_labels.astype(int)

    # acc = np.mean(predicted_labels == domain_label)

    # 使用多数投票规则来确定预测标签
    # 假设train_features的类别标签存储在train_labels中
    train_labels = np.array([1 if i < len(train_features) / 2 else 0 for i in range(len(train_features))])  # 示例生成标签
    votes = train_labels[indices]  # 根据索引获取邻居的标签
    predicted_labels = np.array([np.bincount(vote).argmax() for vote in votes])  # 对每个测试样本，计算最频繁的类别

    acc = np.mean(predicted_labels == domain_label)
    cm = confusion_matrix(domain_label, predicted_labels)
    return predicted_labels, acc, cm


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
        # by setting test_split=True, we can load the pretrained weight of the global adapter and local adapter
        client = Client(args, id, cd.train_dataset, cd.test_dataset, cd.train_loader, cd.test_loader, cd.classnames, init_image_encoder, cls_head, data_name, test_split=True)
        clients.append(client)
        del cd

    # test clients
    start_time = time.time()
    all_clients_test_dataloader = concat_test_datasets(clients)

    client_own_domain_acc = []
    client_other_domain_acc = []
    for id, client in enumerate(clients):
        client.domain_label = generate_domain_label(id, clients)
        print(f'client {id} domain label:', client.domain_label)
        model = copy.deepcopy(client.model)
        model.eval()
        with torch.no_grad():
            # generate train_features
            train_features = client.reference.cpu().detach().numpy()

            test_features = []
            # generate test_features
            for input, _ in all_clients_test_dataloader:
                input = input.to(args.device)
                test_feature = model.base.model.encode_image(input)
                test_feature += model.base.local_adapter(test_feature)
                test_features.append(test_feature.cpu().detach().numpy())

            test_features = np.vstack(test_features)

            print('train_features shape:', train_features.shape)
            print('test_features shape:', test_features.shape)

        client.pred_domain_label, acc, cm = classify_with_dbscan(train_features, test_features, client.domain_label)
        print('predicted_labels:', client.pred_domain_label)
        print('confusion matrix:\n', cm)
        print('acc:', acc)

        client.tp_dataloader, client.tn_dataloader, client.fp_dataloader, client.fn_dataloader = \
            split_dataloader_by_labels(all_clients_test_dataloader, client.pred_domain_label, client.domain_label)

        # do test
        own_domain_preds, other_domain_preds = [], []
        own_domain_labels, other_domain_labels = [], []
        with torch.no_grad():
            for input, target in client.tp_dataloader:
                input, target = input.to(args.device), target.to(args.device)
                output = model.base.model.encode_image(input)
                output += model.base.local_adapter(output)
                output = model.head(output)
                pred = output.argmax(dim=1, keepdim=True)
                own_domain_preds.append(pred)
                own_domain_labels.append(target)

            for input, target in client.fn_dataloader:
                input, target = input.to(args.device), target.to(args.device)
                output = model.base.model.encode_image(input)
                output += model.base.adapter(output)
                output = model.head(output)
                pred = output.argmax(dim=1, keepdim=True)
                own_domain_preds.append(pred)
                own_domain_labels.append(target)

            for input, target in client.fp_dataloader:
                input, target = input.to(args.device), target.to(args.device)
                output = model.base.model.encode_image(input)
                output += model.base.local_adapter(output)
                output = model.head(output)
                pred = output.argmax(dim=1, keepdim=True)
                other_domain_preds.append(pred)
                other_domain_labels.append(target)

            for input, target in client.tn_dataloader:
                input, target = input.to(args.device), target.to(args.device)
                output = model.base.model.encode_image(input)
                output += model.base.adapter(output)
                output = model.head(output)
                pred = output.argmax(dim=1, keepdim=True)
                other_domain_preds.append(pred)
                other_domain_labels.append(target)

        # calculate acc
        own_domain_preds = torch.cat(own_domain_preds, dim=0)
        own_domain_labels = torch.cat(own_domain_labels, dim=0)
        other_domain_preds = torch.cat(other_domain_preds, dim=0)
        other_domain_labels = torch.cat(other_domain_labels, dim=0)
        all_domain_preds = torch.cat([own_domain_preds, other_domain_preds], dim=0)
        all_domain_labels = torch.cat([own_domain_labels, other_domain_labels], dim=0)
        own_domain_acc = own_domain_preds.eq(own_domain_labels.view_as(own_domain_preds)).sum().item() / len(own_domain_labels)
        other_domain_acc = other_domain_preds.eq(other_domain_labels.view_as(other_domain_preds)).sum().item() / len(other_domain_labels)
        print('own domain acc:', own_domain_preds.eq(own_domain_labels.view_as(own_domain_preds)).sum().item() / len(own_domain_labels))
        print('other domain acc:', other_domain_preds.eq(other_domain_labels.view_as(other_domain_preds)).sum().item() / len(other_domain_labels))
        print('all domain acc:', all_domain_preds.eq(all_domain_labels.view_as(all_domain_preds)).sum().item() / len(all_domain_labels))

        client_own_domain_acc.append(own_domain_acc)
        client_other_domain_acc.append(other_domain_acc)
    with open(f'./results/{args.algorithm}_split/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'a+') as f:
        json.dump({'round':0, 'own_acc': client_own_domain_acc, 'other_acc': client_other_domain_acc, 'domain_acc': acc}, f)
        f.write('\n')

    test_time = time.time() - start_time
    print(f'total test time cost: {test_time:.2f}s')

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

    # parser only for auto test split
    parser.add_argument('-algo','--algorithm', type=str, default='fedts', help='Algorithm name')
    # parser.add_argument()
    args = parser.parse_args()

    if args.device == 'cuda':
        args.device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    os.makedirs(f'./results/{args.algorithm}_split/', exist_ok=True)
    with open(f'./results/{args.algorithm}_split/{args.image_encoder_name}_{args.dataset}_sub{args.subset_size}.json', 'w+') as f:
        json.dump(generate_json_config(args), f)
        f.write('\n')

    run(args)
