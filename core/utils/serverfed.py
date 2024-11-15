import copy
import torch
import open_clip
from tqdm import tqdm
from utils.templates import get_templates
import sys

sys.path.append('..')
from models.CLIP import ClassificationHead, Adapter, ImageEncoder
import torch.nn.functional as F
import numpy as np


d = {'RN50':'yfcc15m',
    'RN101': 'cc12m',
    'ViT-B-32': 'laion2b_s34b_b79k',
    'ViT-B-16': 'laion2b_s34b_b88k',
    'ViT-L-14': 'laion2b_s32b_b82k',
    'convnext_base': 'laion400m_s13b_b51k',
    'nllb-clip-base': 'v1',
    "RN50-quickgelu": 'cc12m'}


class Server(torch.nn.Module):
    def __init__(self, args, zeroshot=False):
        super().__init__()
        # 保留原有的初始化代码
        name = args.image_encoder_name
        pretrained = d[name]

        self.pretrained_model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)

        self.device = args.device
        self.warm_up = args.warm_up
        self.max_epochs = args.local_epochs * args.global_rounds
        self.learning_rate = args.lr
        self.pretrained_model.to(self.device)
        self.image_encoder = ImageEncoder(args, zeroshot).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

        # 添加FedFed所需的新属性
        self.feature_dim = None  # 特征维度
        self.global_features = None  # 全局特征
        self.feature_importance = None  # 全局特征重要性
        self.sensitive_threshold = args.sensitive_threshold if hasattr(args, 'sensitive_threshold') else 0.5
        self.num_clients = args.num_clients if hasattr(args, 'num_clients') else 0


        # 特征聚合权重
        self.client_weights = None

    def aggregate_features(self, clients):
        """聚合客户端的特征"""
        # 初始化全局特征
        if self.feature_dim is None:
            self.feature_dim = clients[0].feature_dim
            self.global_features = torch.zeros((self.feature_dim,), device=self.device)
            self.feature_importance = torch.zeros((self.feature_dim,), device=self.device)

        # 计算客户端权重（基于数据量）
        total_samples = sum([len(client.train_dataset) for client in clients])
        self.client_weights = [len(client.train_dataset) / total_samples for client in clients]

        # 聚合性能敏感特征
        aggregated_sensitive_features = torch.zeros_like(self.global_features)
        aggregated_importance = torch.zeros_like(self.feature_importance)

        for idx, client in enumerate(clients):
            # 聚合性能敏感特征
            client_features = client.running_mean.detach()
            client_importance = client.feature_importance.detach()

            # 只聚合性能敏感特征
            sensitive_mask = client.sensitive_features
            aggregated_sensitive_features += self.client_weights[idx] * (client_features * sensitive_mask)
            aggregated_importance += self.client_weights[idx] * client_importance

        # 更新全局特征和重要性
        self.global_features = aggregated_sensitive_features.detach()
        self.feature_importance = aggregated_importance.detach()

    def distribute_features(self, clients):
        """向客户端分发全局特征"""
        for client in clients:
            # 分发全局特征（只分发性能敏感特征）
            client.global_features = self.global_features.clone().detach()

            # 更新客户端的特征重要性阈值
            importance_threshold = torch.quantile(self.feature_importance, self.sensitive_threshold)
            client.sensitive_features = (self.feature_importance > importance_threshold).float()
            client.robust_features = 1 - client.sensitive_features

    def evaluate_global_performance(self, clients):
        """评估全局模型性能"""
        total_correct = 0
        total_samples = 0

        for client in clients:
            client.model.eval()
            with torch.no_grad():
                for inputs, labels in client.test_dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # 提取特征
                    features = client.extract_features(inputs)

                    # 应用全局特征
                    sensitive_features = features * client.sensitive_features
                    robust_features = features * client.robust_features
                    combined_features = sensitive_features + robust_features

                    # 预测
                    outputs = client.model.head(combined_features)
                    _, predicted = torch.max(outputs.data, 1)

                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

        return total_correct / total_samples * 100

    def generate_cls_head(self, dataObject, data_name):
        print(f"build data {data_name} classification head")
        template = get_templates(data_name)

        logit_scale = self.pretrained_model.logit_scale
        self.pretrained_model.eval()
        self.pretrained_model.to(self.device)

        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(dataObject.classnames):
                texts = []
                for t in template:
                    texts.append(t(classname))
                texts = open_clip.tokenize(texts).to(self.device)  # tokenize
                embeddings = self.pretrained_model.encode_text(texts)  # embed with text encoder
                embeddings /= embeddings.norm(dim=-1, keepdim=True)

                embeddings = embeddings.mean(dim=0, keepdim=True)
                embeddings /= embeddings.norm()

                zeroshot_weights.append(embeddings)

            zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(self.device)
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
            zeroshot_weights *= logit_scale.exp()

            zeroshot_weights = zeroshot_weights.squeeze().float()
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

        classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

        return classification_head

    def generate_global_cls_head(self, cls_heads):
        global_cls_head = copy.deepcopy(cls_heads[0])
        for param in global_cls_head.parameters():
            param.data.zero_()
        for cls_head in cls_heads:
            for global_param, param in zip(global_cls_head.parameters(), cls_head.parameters()):
                global_param.data += param.data.clone()
        # mean
        for global_param in global_cls_head.parameters():
            global_param.data /= len(cls_heads)
        self.global_cls_head = global_cls_head
        self.global_cls_head.to(self.device)

    def freeze_except_global_adapter(self):
        for name, params in self.image_encoder.named_parameters():
            if 'global_adapter' not in name:
                params.requires_grad = False
        for params in self.global_cls_head.parameters():
            params.requires_grad = False

    def update_global_model(self, clients):
        """更新全局模型"""
        # 1. 特征聚合
        self.aggregate_features(clients)

        # 2. 分发全局特征
        self.distribute_features(clients)

        # 3. 评估全局性能
        global_performance = self.evaluate_global_performance(clients)

        return global_performance
