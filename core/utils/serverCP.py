import torch

from utils.server import Server


class ServerCP(Server):
    def __init__(self, args, zeroshot=False):
        super().__init__(args, zeroshot)

        # 初始化质心相关属性
        self.global_centroids = None
        self.class_counts = None
        self.feature_dim = args.feature_dim
        self.momentum = args.momentum  # 用于质心更新的动量参数

        # 特征对齐相关参数
        self.centroid_weight = args.centroid_weight
        self.separation_weight = args.separation_weight

        # 用于存储全局特征统计信息
        self.feature_mean = None
        self.feature_var = None

    def initialize_centroids(self, num_classes):
        """初始化全局特征质心"""
        self.global_centroids = torch.zeros((num_classes, self.feature_dim)).to(self.device)
        self.class_counts = torch.zeros(num_classes).to(self.device)
        self.feature_mean = torch.zeros(self.feature_dim).to(self.device)
        self.feature_var = torch.ones(self.feature_dim).to(self.device)

    def update_global_centroids(self, client_centroids, client_counts, client_weights):
        """
        更新全局特征质心
        Args:
            client_centroids: 列表，包含所有客户端的本地质心
            client_counts: 列表，包含所有客户端的类别计数
            client_weights: 列表，包含所有客户端的聚合权重
        """
        if self.global_centroids is None:
            num_classes = client_centroids[0].shape[0]
            self.initialize_centroids(num_classes)

        # 计算加权平均的新质心
        new_centroids = torch.zeros_like(self.global_centroids)
        new_counts = torch.zeros_like(self.class_counts)

        for centroids, counts, weight in zip(client_centroids, client_counts, client_weights):
            new_centroids += weight * centroids * counts.unsqueeze(1)
            new_counts += weight * counts

        # 避免除零
        valid_counts = (new_counts > 0).float()
        new_centroids = new_centroids / (new_counts.unsqueeze(1) + 1e-8)

        # 使用动量更新全局质心
        self.global_centroids = (1 - self.momentum) * self.global_centroids + \
                                self.momentum * new_centroids
        self.class_counts = new_counts

    def update_feature_statistics(self, client_features):
        """更新特征统计信息"""
        batch_mean = torch.mean(client_features, dim=0)
        batch_var = torch.var(client_features, dim=0)

        if self.feature_mean is None:
            self.feature_mean = batch_mean
            self.feature_var = batch_var
        else:
            self.feature_mean = (1 - self.momentum) * self.feature_mean + \
                                self.momentum * batch_mean
            self.feature_var = (1 - self.momentum) * self.feature_var + \
                               self.momentum * batch_var

    def normalize_features(self, features):
        """标准化特征"""
        return (features - self.feature_mean) / (torch.sqrt(self.feature_var + 1e-8))

    def aggregate_models(self, clients, weights):
        """聚合客户端模型"""
        # 1. 聚合模型参数
        aggregated_state = {}
        for key in clients[0].model.state_dict().keys():
            if 'policy_net' not in key:  # 不聚合策略网络参数
                aggregated_state[key] = sum(
                    w * client.model.state_dict()[key].data.clone()
                    for w, client in zip(weights, clients)
                )

        # 2. 更新服务器模型
        self.image_encoder.load_state_dict(aggregated_state)

        return aggregated_state

    def compute_similarity_matrix(self):
        """计算质心之间的相似度矩阵"""
        norm_centroids = self.global_centroids / \
                         (self.global_centroids.norm(dim=1, keepdim=True) + 1e-8)
        similarity_matrix = torch.mm(norm_centroids, norm_centroids.t())
        return similarity_matrix

    def get_centroid_statistics(self):
        """获取质心的统计信息"""
        if self.global_centroids is None:
            return None

        stats = {
            'mean_norm': torch.mean(torch.norm(self.global_centroids, dim=1)).item(),
            'std_norm': torch.std(torch.norm(self.global_centroids, dim=1)).item(),
            'mean_distance': torch.mean(torch.pdist(self.global_centroids)).item(),
            'active_classes': torch.sum(self.class_counts > 0).item()
        }
        return stats

    def save_centroids(self, path):
        """保存质心状态"""
        if self.global_centroids is not None:
            state = {
                'global_centroids': self.global_centroids,
                'class_counts': self.class_counts,
                'feature_mean': self.feature_mean,
                'feature_var': self.feature_var
            }
            torch.save(state, path)

    def load_centroids(self, path):
        """加载质心状态"""
        state = torch.load(path)
        self.global_centroids = state['global_centroids']
        self.class_counts = state['class_counts']
        self.feature_mean = state['feature_mean']
        self.feature_var = state['feature_var']

    def get_centroid_for_class(self, class_idx):
        """获取特定类别的质心"""
        if self.global_centroids is None or class_idx >= self.global_centroids.shape[0]:
            return None
        return self.global_centroids[class_idx]

    def update_single_centroid(self, class_idx, new_features, weight=1.0):
        """更新单个类别的质心"""
        if self.global_centroids is None:
            return

        current_centroid = self.global_centroids[class_idx]
        new_centroid = torch.mean(new_features, dim=0)
        self.global_centroids[class_idx] = (1 - weight) * current_centroid + \
                                           weight * new_centroid
        self.class_counts[class_idx] += len(new_features)
