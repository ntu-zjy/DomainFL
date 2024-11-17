import copy
import os
import json
import numpy as np
from models.CLIP import *
import math
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
from .json_utils import generate_json_config


class Client(nn.Module):
    def __init__(self, args, id, train_dataset, test_dataset, val_dataset, train_dataloader,
                 test_dataloader, val_dataloader, classnames, image_encoder, cls_head,
                 data_name, load_local_adapter=False, test_split=False):
        # 保留原有的初始化代码
        super().__init__()
        self.args = args
        self.id = id
        self.device = args.device
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.max_epochs = self.global_rounds * self.local_epochs
        self.batch_size = args.batch_size
        self.warm_up = args.warm_up
        self.lr = args.lr
        self.data_name = data_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader
        self.classnames = classnames
        self.image_encoder = copy.deepcopy(image_encoder)
        self.cls_head = copy.deepcopy(cls_head)
        self.model = self.construct_model()
        # for auto test data split
        self.domain_label = None
        self.pred_domain_label = None

        self.tp_dataloader = None
        self.tn_dataloader = None
        self.fp_dataloader = None
        self.fn_dataloader = None

        if test_split:
            self.load_local_and_global_adapter()
        elif load_local_adapter:
            self.load_local_adapter()
        else:
            None  # load the local pretrained model for FL
        self.freeze_except_adapter()  # freeze the image encoder and the head, only train the adapter

        self.model.to(self.device)

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=0.1)
        self.loss = torch.nn.CrossEntropyLoss()

        # warmup + cosine annealing lr scheduler on every epoch
        def lr_lambda(current_epoch):
            if current_epoch < self.warm_up:
                return (float(current_epoch) + 1) / float(max(1, self.warm_up))
            else:
                # Cosine annealing
                return 0.5 * (1 + math.cos(math.pi * (current_epoch - self.warm_up) / (self.max_epochs - self.warm_up)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.start_phase = True

        self.klw = args.kl_weight
        self.momentum = args.momentum
        self.global_mean = None

        for x, y in self.train_dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                rep = self.model.base(x).detach()
            break
        self.running_mean = torch.zeros_like(rep[0])
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=self.device)

        self.client_mean = nn.Parameter(Variable(torch.zeros_like(rep[0])))
        self.client_mean.requires_grad_(True)
        self.opt_client_mean = torch.optim.AdamW([self.client_mean], lr=self.lr)

        # 添加FedFed所需的新属性
        self.feature_dim = None  # 特征维度，将在第一次前向传播时设置
        self.sensitive_features = None  # 性能敏感特征的掩码
        self.robust_features = None  # 性能稳健特征的掩码
        self.feature_importance = None  # 特征重要性分数

        # 添加特征蒸馏相关的超参数
        self.distill_temp = args.distill_temp if hasattr(args, 'distill_temp') else 1.0
        self.distill_weight = args.distill_weight if hasattr(args, 'distill_weight') else 0.1

        # 全局特征存储
        self.global_features = None

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.num_batches_tracked.zero_()

    def detach_running(self):
        self.running_mean.detach_()

    def construct_model(self):
        model = ImageClassifier(self.image_encoder, self.cls_head)
        return model

    def load_local_and_global_adapter(self):
        # load the local adapter
        local_path = f"../weights/{self.args.image_encoder_name}/{self.args.dataset}_sub{self.args.subset_size}_{self.args.algorithm}/client_{self.id}_local_adapter.pth"
        global_path = f"../weights/{self.args.image_encoder_name}/{self.args.dataset}_sub{self.args.subset_size}_{self.args.algorithm}/client_{self.id}_global_adapter.pth"
        if os.path.exists(local_path) and os.path.exists(global_path):
            self.model.base.adapter.load_state_dict(torch.load(global_path))
            self.model.base.local_adapter.load_state_dict(torch.load(local_path))
            print(f'Client {self.id} [{self.data_name}] local and global adapter loaded')
        else:
            print(f'Client {self.id} [{self.data_name}] local and global adapter not found')

    def load_local_adapter(self):
        # load the local adapter
        path = f"../weights/{self.args.image_encoder_name}/{self.args.dataset}_sub{self.args.subset_size}_local/client_{self.id}_adapter.pth"
        if os.path.exists(path):
            self.model.base.adapter.load_state_dict(torch.load(path))
            self.model.base.local_adapter.load_state_dict(torch.load(path))
            print(f'Client {self.id} [{self.data_name}] local adapter loaded')
        else:
            print(f'Client {self.id} [{self.data_name}] local adapter not found')

    # fine-tune on the whole image encoder
    def freeze_except_base(self):
        for name, param in self.model.named_parameters():
            if 'base' not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    def freeze_except_adapter(self):
        for name, param in self.model.named_parameters():
            if 'adapter' not in name or 'global_adapter' in name or 'local_adapter' in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)


    def cal_val_loss(self):
        val_loss = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                output = self.model(inputs)
                val_loss += self.loss(output, labels).item()
        return val_loss

    def extract_features(self, inputs):
        """提取模型特征"""
        self.model.eval()
        with torch.no_grad():
            rep = self.model.base.model.encode_image(inputs)
            rep = rep + self.model.base.adapter(rep)
            if self.feature_dim is None:
                self.feature_dim = rep.shape[1]
                self._initialize_feature_masks()
            return rep

    def _initialize_feature_masks(self):
        """初始化特征掩码"""
        if self.sensitive_features is None:
            # 初始时将一半特征设为性能敏感特征
            self.sensitive_features = torch.zeros(self.feature_dim, device=self.device)
            num_sensitive = self.feature_dim // 2
            self.sensitive_features[:num_sensitive] = 1
            self.robust_features = 1 - self.sensitive_features

    def update_feature_importance(self):
        """更新特征重要性"""
        # 使用梯度信息计算特征重要性
        importance_scores = []
        self.model.train()

        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            rep = self.model.base.model.encode_image(inputs)
            rep = rep + self.model.base.adapter(rep)
            outputs = self.model.head(rep)

            rep.retain_grad()
            loss = self.loss(outputs, labels)
            loss.backward()

            # 使用梯度的绝对值作为重要性度量
            importance = torch.abs(rep.grad).mean(0)
            importance_scores.append(importance.detach())

        # 更新特征重要性
        self.feature_importance = torch.stack(importance_scores).mean(0)

        # 更新特征掩码
        _, indices = torch.sort(self.feature_importance, descending=True)
        num_sensitive = self.feature_dim // 2
        self.sensitive_features.zero_()
        self.sensitive_features[indices[:num_sensitive]] = 1
        self.robust_features = 1 - self.sensitive_features

    def fine_tune(self, centralized=None):
        """修改后的训练过程，包含特征蒸馏"""
        self.model.train()
        self.reset_running_stats()

        for epoch in range(self.local_epochs):
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
            for i, (inputs, labels) in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.opt_client_mean.zero_grad()
                self.optimizer.zero_grad()

                # 提取特征
                rep = self.model.base.model.encode_image(inputs)
                rep = rep + self.model.base.adapter(rep)

                if self.feature_dim is None:
                    self.feature_dim = rep.shape[1]
                    self._initialize_feature_masks()

                # 分离性能敏感和稳健特征
                sensitive_rep = rep * self.sensitive_features
                robust_rep = rep * self.robust_features

                # 计算running mean
                running_mean = torch.mean(rep, dim=0)
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked.add_(1)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * running_mean

                # 计算损失
                total_loss = 0

                # 1. 分类损失
                output = self.model.head(rep + self.client_mean)
                cls_loss = self.loss(output, labels)
                total_loss += cls_loss

                # 2. 特征蒸馏损失（如果有全局特征）
                if self.global_features is not None:
                    distill_loss = F.mse_loss(
                        F.normalize(sensitive_rep, dim=1),
                        F.normalize(self.global_features, dim=1)
                    )
                    total_loss += self.distill_weight * distill_loss

                # 3. 正则化损失
                if self.global_mean is not None:
                    reg_loss = torch.mean(0.5 * (self.running_mean - self.global_mean) ** 2)
                    total_loss += reg_loss * self.klw

                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.args.clip)
                torch.nn.utils.clip_grad_norm_([self.client_mean], self.args.clip)
                self.optimizer.step()
                self.opt_client_mean.step()
                self.detach_running()

                lr = self.optimizer.param_groups[0]['lr']

                if centralized:
                    pbar.set_description(
                        f'Centralized Epoch: {epoch}, Iter:{i}, Loss: {round(total_loss.item(), 5)}, lr: {lr}')
                else:
                    pbar.set_description(
                        f'Client {self.id}: [{self.data_name}], Local Epoch: {epoch}, Iter:{i}, Loss: {round(total_loss.item(), 5)}, lr: {lr}')

            self.scheduler.step()

    def get_sensitive_features(self):
        """获取性能敏感特征"""
        return self.sensitive_features

    def update_global_features(self, global_features):
        """更新全局特征"""
        self.global_features = global_features.to(self.device)

    def test(self, test_dataloader=None):
        # use own test_dataloader if not provided
        test_dataloader = self.test_dataloader if test_dataloader is None else test_dataloader

        self.model.eval()
        predicted_list = []
        labels_list = []
        prob_list = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                rep = self.model.base.model.encode_image(inputs)
                rep = rep + self.model.base.adapter(rep)
                outputs = self.model.head(rep + self.client_mean)
                prob = F.softmax(outputs, 1)
                _, predicted = torch.max(prob, 1)
                prob_list.append(prob.cpu().numpy())
                predicted_list.extend(predicted.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        prob_list = np.vstack(prob_list)

        acc = 100 * accuracy_score(labels_list, predicted_list)
        auc = 100 * roc_auc_score(labels_list, prob_list, multi_class='ovo')
        f1 = 100 * f1_score(labels_list, predicted_list, average='macro')
        precision = 100 * precision_score(labels_list, predicted_list, average='macro')
        recall = 100 * recall_score(labels_list, predicted_list, average='macro')
        return round(acc, 4), round(auc, 4), round(f1, 4), round(precision, 4), round(recall, 4)

    def test_on_all_clients(self, clients):
        # test on all clients
        accs = []
        for client in clients:
            acc, auc, f1, precision, recall = self.test(client.test_dataloader)
            accs.append(acc)
        print(f'Client {self.id} [{self.data_name}] on all the other clients accuracy: {accs}')
        return accs

    def save_adapter(self, args, algo):
        os.makedirs("../weights", exist_ok=True)
        os.makedirs(f"../weights/{args.image_encoder_name}", exist_ok=True)
        os.makedirs(f"../weights/{args.image_encoder_name}/{args.dataset}_sub{args.subset_size}_{algo}", exist_ok=True)
        dir = f"../weights/{args.image_encoder_name}/{args.dataset}_sub{args.subset_size}_{algo}"
        config = generate_json_config(args)
        if algo != 'local':
            # algorithm except local use 'adapter' as the global adapter to train and aggregate during the federated learning
            path = f"{dir}/client_{self.id}_global_adapter.pth"
            torch.save(self.model.base.adapter.state_dict(), path)
            path = f"{dir}/client_{self.id}_local_adapter.pth"
            torch.save(self.model.base.local_adapter.state_dict(), path)
        else:
            path = f"{dir}/client_{self.id}_adapter.pth"
            torch.save(self.model.base.adapter.state_dict(), path)

        # write config file
        with open(f"{dir}/config.json", 'w+') as f:
            json.dump(config, f)
