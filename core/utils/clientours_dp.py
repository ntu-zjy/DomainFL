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
from collections import defaultdict

class Client(nn.Module):
    def __init__(self, args, id, train_dataset, test_dataset, val_dataset, train_dataloader, test_dataloader, val_dataloader, classnames, image_encoder, cls_head, data_name, load_local_adapter=False, test_split=False):
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

        self.protos = None
        self.global_protos = None
        self.mse_loss = nn.MSELoss()
        self.kd_loss = nn.KLDivLoss()
        self.rw = args.regularization_weight
        self.kdw = args.kd_loss_weight
        self.sra = args.sample_ratio
        self.sram = args.sample_ratio_method
        self.dp = args.diff_privacy

        # for auto test data split
        self.domain_label = None
        self.pred_domain_label = None

        self.reference = self.generate_reference()
        self.tp_dataloader = None
        self.tn_dataloader = None
        self.fp_dataloader = None
        self.fn_dataloader = None


        if test_split:
            self.load_local_and_global_adapter()
        elif load_local_adapter:
            self.load_local_adapter()
        else:
            None # load the local pretrained model for FL
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
        self.start_phase = True

    def generate_reference(self, batch_num=20):
        with torch.no_grad():
            references = []
            for i, (inputs, _) in enumerate(self.train_dataloader):
                if i < batch_num:
                    inputs = inputs.to(self.device)
                    image_features = self.model.base.model.encode_image(inputs)
                    reference = image_features + self.model.base.local_adapter(image_features)
                    references.append(reference)
            references = torch.cat(references, dim=0)
            return references

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

    def fine_tune(self, centralized=None, global_round=None, train_dataloader=None):
        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
        protos = defaultdict(list)
        self.model.train()
        for epoch in range(self.local_epochs):
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))

            for i, (inputs, labels) in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                rep = self.model.base.model.encode_image(inputs)
                local_rep = rep + self.model.base.adapter(rep)

                outputs = self.model.head(local_rep)
                loss = self.loss(outputs, labels)
                if global_round != 0:
                    # print('knowledge distillation...')
                    global_rep = rep + self.model.base.global_adapter(rep)
                    global_outputs = self.model.head(global_rep)

                    source = F.log_softmax(outputs, dim=1)
                    target = F.softmax(global_outputs, dim=1)
                    kd_loss = self.kd_loss(source, target)

                    loss = loss + self.kdw * kd_loss
                    # print('cross loss:', loss.item(), 'mse loss:', mse_loss.item(), 'kd loss:', kd_loss.item())

                for i, yy in enumerate(labels):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

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

            self.protos = agg_func(protos, sample_ratio=self.sra, sample_method=self.sram, epsilon=self.dp)

            self.scheduler.step()

    def set_protos(self, global_protos):
        self.global_protos = global_protos

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
                local_rep = rep + self.model.base.adapter(rep)
                outputs = self.model.head(local_rep)

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
        return round(acc, 4)


    def test_on_all_clients(self, clients):
        # test on all clients
        accs = []
        for client in clients:
            acc = self.test(client.test_dataloader)
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

def agg_func(protos, sample_ratio=0.9, sample_method='cluster', svd_ratio=0.9, epsilon=100):
    """
    use svd to aggregate the prototypes
    """
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            prototype = proto_list[0].data
            for i in proto_list[1:]:
                new = i.data
                new = new
                prototype = torch.vstack((prototype,new))

            # sample some of the prototypes
            if sample_method == 'average':
                prototype = average_sample(prototype.T).T
            elif sample_method == 'random':
                prototype = random_sample(prototype.T, sample_ratio).T
            elif sample_method == 'cluster':
                prototype = cluster_sample(prototype.T, sample_ratio).T

            # if svd_ratio >= 1:
            #     pass
            # else:
            #     # use svd decomposition to compress the prototype
            #     U, S, Vt = torch.linalg.svd(prototype, full_matrices=False)

            #     # print("S.shape:", S.shape)
            #     k = math.ceil(S.shape[0] * svd_ratio)
            #     Uk = U[:, :k]
            #     Sk = torch.diag(S[:k])
            #     Vtk = Vt[:k, :]
            #     prototype = Uk @ Sk @ Vtk

            # differential privacy
            prototype = dp(prototype, epsilon=epsilon)
            protos[label] = prototype
        else:
            prototype = proto_list[0]

            # differential privacy
            prototype = dp(prototype, epsilon=epsilon)
            protos[label] = prototype
    # protos[label] = prototype (embedding_nums, feature_dim)
    return protos

# some more sampling methods
def average_sample(proto):
    # proto.shape = (feature_dim, number of samples in this label)
    # return proto (feature_dim, 1)
    proto = torch.mean(proto, dim=1, keepdim=True)
    return proto


# ramdom sample
import random
def random_sample(proto, sample_ratio=0.1):
    # proto.shape = (feature_dim, number of samples in this label)
    # return proto (feature_dim, sample_num)
    sample_num = math.ceil(proto.shape[1] * sample_ratio)
    sample_idx = random.sample(range(proto.shape[1]), sample_num)
    proto = proto[:, sample_idx]
    return proto

# sample by clustering (k-means)
from sklearn.cluster import KMeans
def cluster_sample(proto, sample_ratio=0.1):
    # print("proto.shape:", proto.shape)
    # proto.shape = (feature_dim, number of samples in this label)
    # We use k-means to cluster the samples. The number of cluster is equal to the number of samples we want to sample.
    # Then we use the cluster center as the samples.
    # return proto (feature_dim, sample_num)
    device = proto.device
    proto = proto.cpu().numpy()
    cluster_num = math.ceil(proto.shape[1] * sample_ratio)
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(proto.T)
    cluster_center = kmeans.cluster_centers_
    proto = cluster_center.T
    proto = torch.tensor(proto).to(device)
    proto = proto.to(torch.float32)
    return proto


# differential privacy
def dp(proto, epsilon=10):
    if epsilon == 0:
        return proto
    # proto.shape = (feature_dim, number of samples in this label)
    # return proto (feature_dim, sample_num)
    # add noise to the prototype
    device = proto.device
    noise = torch.normal(0, epsilon, proto.shape)
    noise = noise.to(device)
    proto = proto + noise
    return proto