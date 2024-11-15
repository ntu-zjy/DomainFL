import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torch import nn
from tqdm import tqdm

from utils.clientavgDBE import Client


class ClientCP(Client):
    def __init__(self, args, id, train_dataset, test_dataset, val_dataset,
                 train_dataloader, test_dataloader, val_dataloader,
                 classnames, image_encoder, cls_head, data_name,
                 load_local_adapter=False, test_split=False):
        super().__init__(args, id, train_dataset, test_dataset, val_dataset,
                         train_dataloader, test_dataloader, val_dataloader,
                         classnames, image_encoder, cls_head, data_name,
                         load_local_adapter, test_split)

        # 初始化特征维度
        self.feature_dim = args.feature_dim
        self.num_classes = len(classnames)

        # 初始化质心相关参数
        self.local_centroids = torch.zeros((self.num_classes, self.feature_dim)).to(self.device)
        self.global_centroids = None
        self.class_counts = torch.zeros(self.num_classes).to(self.device)

        # 初始化策略网络
        self.policy_net = self._init_policy_network()
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=args.policy_lr
        )

        # 损失权重
        self.centroid_weight = args.centroid_weight
        self.separation_weight = args.separation_weight

    def _init_policy_network(self):
        """初始化特征分离策略网络"""
        policy_net = nn.Sequential(
            nn.Linear(self.feature_dim, self.args.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.hidden_dim, 2)
        ).to(self.device)
        return policy_net

    def update_local_centroids(self, features, labels):
        """更新本地特征质心"""
        for feat, label in zip(features, labels):
            self.local_centroids[label] = (
                                                  self.local_centroids[label] * self.class_counts[label] + feat
                                          ) / (self.class_counts[label] + 1)
            self.class_counts[label] += 1

    def compute_centroid_loss(self, features, labels):
        """计算与全局质心的对齐损失"""
        if self.global_centroids is None:
            return torch.tensor(0.0).to(self.device)

        centroid_loss = 0
        for feat, label in zip(features, labels):
            centroid_loss += F.mse_loss(
                feat,
                self.global_centroids[label]
            )
        return centroid_loss / len(features)

    def compute_separation_loss(self, global_features, personal_features):
        """计算特征分离损失"""
        # 正交性损失
        similarity = F.cosine_similarity(global_features, personal_features, dim=1)
        orthogonal_loss = torch.mean(similarity ** 2)

        # 特征大小约束
        magnitude_loss = torch.abs(
            torch.norm(global_features, dim=1) - torch.norm(personal_features, dim=1)
        ).mean()

        return orthogonal_loss + 0.1 * magnitude_loss

    def separate_features(self, features):
        """使用策略网络分离全局和个性化特征"""
        weights = torch.softmax(self.policy_net(features), dim=1)
        global_features = weights[:, 0].unsqueeze(1) * features
        personal_features = weights[:, 1].unsqueeze(1) * features
        return global_features, personal_features

    def fine_tune(self, centralized=None):
        """重写fine_tune方法以适应FedCP"""
        self.model.train()
        self.reset_running_stats()

        for epoch in range(self.local_epochs):
            pbar = tqdm(enumerate(self.train_dataloader),
                        total=len(self.train_dataloader))

            for i, (inputs, labels) in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 优化器梯度清零
                self.optimizer.zero_grad()
                self.policy_optimizer.zero_grad()

                # 提取基础特征
                base_features = self.model.base.model.encode_image(inputs)
                features = base_features + self.model.base.adapter(base_features)

                # 更新运行平均
                running_mean = torch.mean(features, dim=0)
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked.add_(1)
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                    self.momentum * running_mean

                # 特征分离
                global_features, personal_features = self.separate_features(features)

                # 更新本地质心
                self.update_local_centroids(global_features.detach(), labels)

                # 计算各种损失
                combined_features = global_features + personal_features
                cls_loss = self.loss(self.model.head(combined_features), labels)
                centroid_loss = self.compute_centroid_loss(global_features, labels)
                separation_loss = self.compute_separation_loss(
                    global_features, personal_features
                )

                # 总损失
                total_loss = cls_loss + \
                             self.centroid_weight * centroid_loss + \
                             self.separation_weight * separation_loss

                # 反向传播和优化
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.args.clip)
                self.optimizer.step()
                self.policy_optimizer.step()

                # 更新进度条
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_description(
                    f'Client {self.id}: [{self.data_name}], '
                    f'Local Epoch: {epoch}, Iter:{i}, '
                    f'Loss: {round(total_loss.item(), 5)}, '
                    f'lr: {lr}'
                )

            self.scheduler.step()

    def test(self, test_dataloader=None):
        """重写test方法以适应FedCP"""
        test_dataloader = self.test_dataloader if test_dataloader is None \
            else test_dataloader

        self.model.eval()
        predicted_list = []
        labels_list = []
        prob_list = []

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 提取和分离特征
                base_features = self.model.base.model.encode_image(inputs)
                features = base_features + self.model.base.adapter(base_features)
                global_features, personal_features = self.separate_features(features)

                # 组合特征进行预测
                outputs = self.model.head(global_features + personal_features)
                prob = F.softmax(outputs, 1)

                _, predicted = torch.max(prob, 1)
                prob_list.append(prob.cpu().numpy())
                predicted_list.extend(predicted.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        prob_list = np.vstack(prob_list)

        # 计算评估指标
        acc = 100 * accuracy_score(labels_list, predicted_list)
        auc = 100 * roc_auc_score(labels_list, prob_list, multi_class='ovo')
        f1 = 100 * f1_score(labels_list, predicted_list, average='macro')
        precision = 100 * precision_score(labels_list, predicted_list, average='macro')
        recall = 100 * recall_score(labels_list, predicted_list, average='macro')

        return round(acc, 4), round(auc, 4), round(f1, 4), \
            round(precision, 4), round(recall, 4)
