import os
import json
import torch
from tqdm import tqdm
import numpy as np
import copy
import sys
sys.path.append('..')
from models.CLIP import *
import torch.nn.functional as F
import math
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from .json_utils import generate_json_config

class Client(nn.Module):
    def __init__(self, args, id, train_dataset, test_dataset, train_dataloader, test_dataloader, classnames, image_encoder, cls_head, data_name, load_local_adapter=True):
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
        self.classnames = classnames
        self.image_encoder = copy.deepcopy(image_encoder)
        self.cls_head = copy.deepcopy(cls_head)
        self.model = self.construct_model()

        self.load_local_adapter() if load_local_adapter else None # load the local pretrained model for FL
        self.freeze_except_adapter() # freeze the image encoder and the head, only train the adapter

        self.model.to(self.device)

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=self.lr)
        self.loss = torch.nn.CrossEntropyLoss()
        # warmup + cosine annealing lr scheduler on every epoch
        def lr_lambda(current_epoch):
            if current_epoch < self.warm_up:
                return (float(current_epoch) + 1) / float(max(1, self.warm_up))
            else:
                # Cosine annealing
                return 0.5 * (1 + math.cos(math.pi * (current_epoch - self.warm_up) / (self.max_epochs - self.warm_up)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def construct_model(self):
        model = ImageClassifier(self.image_encoder, self.cls_head)
        return model

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

    def local_adaptation(self, adapt_trainloader, threshold=0.1):
        # only train the adapter weight to find the balance between the global adapter and the local adapter
        for param in self.model.parameters():
            param.requires_grad_(False)
        output_dim =self.model.base.output_dim
        global_adapter_weights = nn.Parameter(torch.eye(n=output_dim, dtype=torch.float32).to(self.args.device)) # global adapter weights
        Identical_matrix = torch.eye(n=output_dim, dtype=torch.float32).to(self.args.device)
        params = [global_adapter_weights]
        # print('params:', params)
        optimizer = torch.optim.AdamW(params, lr=self.lr)

        while True:
            losses = []
            for i, (inputs, labels) in enumerate(adapt_trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                image_features = self.model.base.model.encode_image(inputs)
                global_adapter_features = self.model.base.adapter(image_features)
                local_adapter_features = self.model.base.local_adapter(image_features)
                local_shifted_adapter_features = local_adapter_features @ (Identical_matrix - global_adapter_weights)
                global_shifted_adapter_features = global_adapter_features @ global_adapter_weights
                combined_features = local_shifted_adapter_features + global_shifted_adapter_features + image_features

                outputs = self.model.head(combined_features)
                loss = self.loss(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.args.clip)
                optimizer.step()

                losses.append(loss.item())
            if np.std(losses) < threshold:
                print(f'Client {self.id} [{self.data_name}] local adaptation loss std: {np.std(losses)}')
                break
        print('params:', params)
        # merge the local adapter and the global adapter with glocal_adapter_weight
        local_adapter = self.model.base.local_adapter.state_dict()['fc.2.weight']
        global_adapter = self.model.base.adapter.state_dict()['fc.2.weight']
        for name, param in self.model.named_parameters():
            if name == 'base.adapter.fc.2.weight':
                param.data = global_adapter_weights @ global_adapter.data.clone()
            elif name == 'base.local_adapter.fc.2.weight':
                param.data = (Identical_matrix - global_adapter_weights) @ local_adapter.data.clone() +  global_adapter_weights @ global_adapter.data.clone()
        self.freeze_except_adapter()

    def local_adapt_train2(self, adapt_trainloader, threshold=0.1):
        # only train the adapter weight to find the balance between the global adapter and the local adapter
        for param in self.model.parameters():
            param.requires_grad_(False)
        output_dim =self.model.base.output_dim
        global_adapter_weights = nn.Parameter(torch.eye(n=output_dim, dtype=torch.float32).to(self.args.device)) # global adapter weights
        # Identical_matrix = torch.eye(n=output_dim, dtype=torch.float32).to(self.args.device)
        params = [global_adapter_weights]
        # print('params:', params)
        optimizer = torch.optim.AdamW(params, lr=self.lr)

        while True:
            losses = []
            for i, (inputs, labels) in enumerate(adapt_trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                image_features = self.model.base.model.encode_image(inputs)
                local_adapter_features = self.model.base.local_adapter(image_features)
                global_adapter_features = self.model.base.adapter(image_features)
                global_shifted_adapter_features = global_adapter_features @ global_adapter_weights
                combined_features = global_shifted_adapter_features + image_features

                outputs = self.model.head(combined_features)
                mse_loss = torch.nn.MSELoss(reduction='mean')
                mseloss = mse_loss(local_adapter_features, global_adapter_features)
                loss = self.loss(outputs, labels) + mseloss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5)
                optimizer.step()

                losses.append(loss.item())
            if np.std(losses) < threshold:
                print(f'Client {self.id} [{self.data_name}] local adaptation loss std: {np.std(losses)}')
                break
        # print('params:', params)
        # merge the local adapter and the global adapter with glocal_adapter_weight
        for name, param in self.model.named_parameters():
            if name == 'base.adapter.fc.2.weight':
                param.data = torch.matmul(global_adapter_weights.data.clone(), param.data.clone())
        self.freeze_except_adapter()

    def local_adapt_train3(self, adapt_trainloader, threshold=0.2):
        # only train the adapter weight to find the balance between the global adapter and the local adapter
        for param in self.model.parameters():
            param.requires_grad_(False)
        output_dim =self.model.base.output_dim
        invariant_adapter_weights = nn.Parameter(torch.eye(n=output_dim, dtype=torch.float32).to(self.args.device)) # global adapter weights
        params = [invariant_adapter_weights]
        optimizer = torch.optim.AdamW(params, lr=self.lr)

        while True:
            losses = []
            for i, (inputs, labels) in enumerate(adapt_trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                image_features = self.model.base.model.encode_image(inputs)
                local_adapter_features = self.model.base.local_adapter(image_features)
                global_adapter_features = self.model.base.adapter(image_features)

                global_invariant_adapter_features = global_adapter_features @ invariant_adapter_weights
                local_invariant_adapter_features = local_adapter_features @ invariant_adapter_weights

                global_features = image_features + global_invariant_adapter_features
                local_features = image_features + local_invariant_adapter_features

                global_invariant_outputs = self.model.head(global_features)
                local_invariant_outputs = self.model.head(local_features)

                invariant_classification_loss = self.loss(global_invariant_outputs, labels) + self.loss(local_invariant_outputs, labels)

                mse_loss = torch.nn.MSELoss(reduction='mean')
                invariant_mseloss = 100 * mse_loss(global_features, local_features)
                # print(f"invariant_classification_loss: {invariant_classification_loss.item()}, invariant_mseloss: {invariant_mseloss.item()}")
                loss = invariant_classification_loss + invariant_mseloss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5)
                optimizer.step()

                losses.append(loss.item())
            if np.std(losses) < threshold:
                print(f'Client {self.id} [{self.data_name}], invariant classification loss: {invariant_classification_loss.item()}, invariant mse loss: {invariant_mseloss.item()}, std: {np.std(losses)}')
                break
        # print('params:', params)
        # merge the local adapter and the global adapter with glocal_adapter_weight
        for name, param in self.model.named_parameters():
            if name == 'base.adapter.fc.2.weight':
                param.data = torch.matmul(invariant_adapter_weights.data.clone(), param.data.clone())

        self.freeze_except_adapter()

    def fine_tune(self, centralized=None):
        self.model.train()
        for epoch in range(self.local_epochs):
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
            for i, (inputs, labels) in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.args.clip)
                self.optimizer.step()
                lr = self.optimizer.param_groups[0]['lr']

                if centralized:
                    pbar.set_description\
                        (f'Centralized Epoch: {epoch}, Iter:{i}, Loss: {round(loss.item(), 5)}, lr: {lr}')
                else:
                    pbar.set_description\
                        (f'Client {self.id}: [{self.data_name}], Local Epoch: {epoch}, Iter:{i}, Loss: {round(loss.item(), 5)}, lr: {lr}')
            self.scheduler.step()

    def normalized_l2_loss(self, teacher_features, student_features, alpha=0.1):
        # 归一化特征向量
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)
        student_norm = F.normalize(student_features, p=2, dim=1)
        # 计算L2范数
        l2_distance = (teacher_norm - student_norm).norm(p=2, dim=1)
        # 最大化L2范数
        loss = -l2_distance.mean()
        return alpha * loss

    def feature_invariant_fine_tune(self, centralized=None, global_round=0):
        output_dim = self.model.base.output_dim
        invariant_shift_weight = nn.Parameter(torch.eye(n=output_dim, dtype=torch.float32).to(self.args.device)) # global adapter weights
        optimizer = torch.optim.AdamW([invariant_shift_weight], lr=self.lr)
        def lr_lambda(current_epoch):
            if current_epoch < self.warm_up:
                return (float(current_epoch) + 1) / float(max(1, self.warm_up))
            else:
                # Cosine annealing
                return 0.5 * (1 + math.cos(math.pi * (current_epoch - self.warm_up) / (self.max_epochs - self.warm_up)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        self.model.train()
        for epoch in range(self.local_epochs):
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
            for i, (inputs, labels) in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                image_features = self.model.base.model.encode_image(inputs)
                local_adapter_features = self.model.base.local_adapter(image_features)
                global_adapter_features = self.model.base.adapter(image_features)

                global_invariant_adapter_features = global_adapter_features @ invariant_shift_weight
                # local_invariant_adapter_features = local_adapter_features @ invariant_shift_weight

                global_features = image_features + global_invariant_adapter_features
                local_features = image_features + local_adapter_features

                global_invariant_outputs = self.model.head(global_features)
                # local_invariant_outputs = self.model.head(local_features)

                invariant_classification_loss = self.loss(global_invariant_outputs, labels)

                neg_dis_loss = self.normalized_l2_loss(global_features, local_features)
                loss = invariant_classification_loss + neg_dis_loss
                # print(f'invariant_classification_loss: {invariant_classification_loss.item()}, kl_loss: {neg_dis_loss.item()}')

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.args.clip)
                optimizer.step()
                lr = optimizer.param_groups[0]['lr']

                if centralized:
                    pbar.set_description\
                        (f'Centralized Epoch: {epoch}, Iter:{i}, Loss: {round(loss.item(), 5)}, lr: {lr}')
                else:
                    pbar.set_description\
                        (f'Client {self.id}: [{self.data_name}], Local Epoch: {epoch}, Iter:{i}, Loss: {round(loss.item(), 5)}, lr: {lr}')
            scheduler.step()
        for name, param in self.model.named_parameters():
            if name == 'base.adapter.fc.2.weight':
                param.data = torch.matmul(invariant_shift_weight.data.clone(), param.data.clone())

    def whitebox_domain_adaptive_test(self, clients):
        # test on all clients
        accs = []
        for id, client in enumerate(clients):
            # use local adapter
            if id == self.id:
                acc, auc, f1, precision, recall = self.local_adapter_test(client.test_dataloader)
            # use global adapter
            else:
                acc, auc, f1, precision, recall = self.test(client.test_dataloader)
            accs.append(acc)
        print(f'Client {self.id} [{self.data_name}] on all the other clients accuracy: {accs}')
        return accs

    def domain_adaptive_test(self, clients=None, train_subset_dataloader=None, true_domain_id=None):
        import numpy as np
        import torch
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics import accuracy_score

        # 设置PCA降维后的主成分数目
        n_components = 50  # 你可以根据需要调整这个值

        # 获取所有测试数据和训练数据的特征，不再区分不同客户端的测试集
        test_features = []
        test_labels = []
        train_features = []

        # 无需梯度计算
        with torch.no_grad():
            # 获取训练数据特征
            for inputs, labels in train_subset_dataloader:
                inputs = inputs.to(self.device)
                image_features = self.model.base(inputs)
                train_features.append(image_features.cpu().numpy())
                break # we only choose one batch data to train the nearest neighbor model

            train_features = np.vstack(train_features)

            # 获取测试数据特征
            for client in clients:
                test_dataloader = client.test_dataloader
                for inputs, labels in test_dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    image_features = self.model.base(inputs)

                    test_features.append(image_features.cpu().numpy())
                    # 确定每个样本的标签
                    domain_label = np.ones((labels.size(0), 1)) if client.id == true_domain_id else np.zeros((labels.size(0), 1))
                    test_labels.append(domain_label)

            test_features = np.vstack(test_features)
            test_labels = np.vstack(test_labels).ravel()  # 将标签展平

        import time
        s = time.time()
        # 数据归一化
        scaler = StandardScaler()
        train_features_norm = scaler.fit_transform(train_features)
        test_features_norm = scaler.transform(test_features)
        t1 = time.time()
        print(f"Normalization completed in {t1 - s:.2f} seconds")

        # 应用PCA进行降维
        pca = PCA(n_components=0.1, random_state=self.args.seed)
        train_features_reduced = pca.fit_transform(train_features_norm)
        print('PCA n_components:', pca.n_components_)
        test_features_reduced = pca.transform(test_features_norm)
        print('PCA n_components:', pca.n_components_)
        print('PCA reduction completed')
        t2 = time.time()
        print(f"PCA completed in {t2 - t1:.2f} seconds")

        # 初始化最近邻模型
        nbrs = NearestNeighbors(n_neighbors=1)
        nbrs.fit(train_features_reduced)

        # 对所有测试数据进行最近邻查询
        distances, indices = nbrs.kneighbors(test_features_reduced)
        # 最近邻标签为1，远离标签为0
        predicted_labels = np.zeros_like(test_labels)
        threshold_distance = np.median(distances)  # 使用中位数作为阈值
        predicted_labels[distances.flatten() <= threshold_distance] = 1

        # 计算准确度
        accuracy = accuracy_score(test_labels, predicted_labels)
        print(f"Accuracy: {accuracy:.2f}")
        t3 = time.time()
        print(f"Nearest neighbor search completed in {t3 - t2:.2f} seconds")

        # test on all clients
        accs = []
        for id, client in enumerate(clients):
            # use local adapter
            if id == self.id:
                acc, auc, f1, precision, recall = self.local_adapter_test(client.test_dataloader)
            # use global adapter
            else:
                acc, auc, f1, precision, recall = self.test(client.test_dataloader)
            accs.append(acc)
        print(f'Client {self.id} [{self.data_name}] on all the other clients accuracy: {accs}')
        return accs

    def local_adapter_test(self, test_dataloader=None):
        # use own test_dataloader if not provided
        test_dataloader = self.test_dataloader if test_dataloader is None else test_dataloader

        # print('local_adapter weight:', self.model.base.local_adapter.state_dict()['fc.2.weight'])
        # print('adapter weight:', self.model.base.adapter.state_dict()['fc.2.weight'])
        # print('global_adapter weight:', self.model.base.global_adapter.state_dict()['fc.2.weight'])

        self.model.eval()
        predicted_list = []
        labels_list = []
        prob_list = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                image_features = self.model.base.model.encode_image(inputs)
                global_adapter_features = self.model.base.local_adapter(image_features)
                global_adapter_features = image_features + global_adapter_features
                outputs = self.model.head(global_adapter_features)
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

    def global_adapter_test(self, test_dataloader=None):
        # use own test_dataloader if not provided
        test_dataloader = self.test_dataloader if test_dataloader is None else test_dataloader

        self.model.eval()
        predicted_list = []
        labels_list = []
        prob_list = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                image_features = self.model.base.model.encode_image(inputs)
                global_adapter_features = self.model.base.global_adapter(image_features)
                global_adapter_features += image_features
                outputs = self.model.head(global_adapter_features)
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

    # return acc, auc, f1, precision, recall
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
                outputs = self.model(inputs)
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

