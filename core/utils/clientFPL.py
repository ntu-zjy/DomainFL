import os
import json
import torch
from tqdm import tqdm
import numpy as np
import copy
import sys

from core.utils.json_utils import generate_json_config

sys.path.append('..')
from models.CLIP import *
from torch import nn, optim
import torch.nn.functional as F
import math
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
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

        if test_split:
            self.load_local_and_global_adapter()
        elif load_local_adapter:
            self.load_local_adapter()
        else:
            None  # Load the local pretrained model for FL
        self.freeze_except_adapter()  # Freeze the image encoder and the head, only train the adapter

        self.model.to(self.device)

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=self.lr)
        self.loss = torch.nn.CrossEntropyLoss()

        def lr_lambda(current_epoch):
            if current_epoch < self.warm_up:
                return (float(current_epoch) + 1) / float(max(1, self.warm_up))
            else:
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


    def fine_tune(self, centralized=None):
        protos = defaultdict(list)
        self.model.train()
        for epoch in range(self.local_epochs):
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
            for i, (inputs, labels) in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # 获取图像特征
                rep = self.model.base.model.encode_image(inputs)
                local_rep = rep + self.model.base.adapter(rep)
                outputs = self.model.head(local_rep)

                # 计算交叉熵损失
                lossCE = self.loss(outputs, labels)

                # 如果存在全局原型，计算 InfoNCE 损失
                if self.global_protos is not None:
                    all_global_protos_keys = np.array(list(self.global_protos.keys()))
                    all_f = []
                    mean_f = []
                    for protos_key in all_global_protos_keys:
                        temp_f = self.global_protos[protos_key]
                        temp_f = torch.cat(temp_f, dim=0).to(self.device)
                        all_f.append(temp_f.cpu())
                        mean_f.append(torch.mean(temp_f, dim=0).cpu())
                    all_f = [item.detach() for item in all_f]
                    mean_f = [item.detach() for item in mean_f]

                    i = 0
                    loss_InfoNCE = None
                    for label in labels:
                        if label.item() in self.global_protos.keys():
                            f_now = local_rep[i].unsqueeze(0)
                            loss_instance = self.hierarchical_info_loss(
                                f_now, label, all_f, mean_f, all_global_protos_keys
                            )
                            if loss_InfoNCE is None:
                                loss_InfoNCE = loss_instance
                            else:
                                loss_InfoNCE += loss_instance
                        i += 1
                    loss_InfoNCE = loss_InfoNCE / i if loss_InfoNCE is not None else 0
                else:
                    loss_InfoNCE = 0

                # 总损失
                loss = lossCE + loss_InfoNCE
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.args.clip)
                self.optimizer.step()
                lr = self.optimizer.param_groups[0]['lr']

                if centralized:
                    pbar.set_description(
                        f'Centralized Epoch: {epoch}, Iter:{i}, Loss: {round(loss.item(), 5)}, lr: {lr}'
                    )
                else:
                    pbar.set_description(
                        f'Client {self.id}: [{self.data_name}], Local Epoch: {epoch}, Iter:{i}, Loss: {round(loss.item(), 5)}, lr: {lr}'
                    )

            self.protos = agg_func(protos)
            self.scheduler.step()

    def test(self, test_dataloader=None):
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

    def save_adapter(self, args, algo):
        os.makedirs("../weights", exist_ok=True)
        os.makedirs(f"../weights/{args.image_encoder_name}", exist_ok=True)
        os.makedirs(f"../weights/{args.image_encoder_name}/{args.dataset}_sub{args.subset_size}_{algo}", exist_ok=True)
        dir = f"../weights/{args.image_encoder_name}/{args.dataset}_sub{args.subset_size}_{algo}"
        config = generate_json_config(args)
        if algo != 'local':
            path = f"{dir}/client_{self.id}_global_adapter.pth"
            torch.save(self.model.base.adapter.state_dict(), path)
            path = f"{dir}/client_{self.id}_local_adapter.pth"
            torch.save(self.model.base.local_adapter.state_dict(), path)
        else:
            path = f"{dir}/client_{self.id}_adapter.pth"
            torch.save(self.model.base.adapter.state_dict(), path)

        with open(f"{dir}/config.json", 'w+') as f:
            json.dump(config, f)

    def set_protos(self, global_protos):
        self.global_protos = global_protos

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos