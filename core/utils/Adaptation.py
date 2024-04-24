import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from typing import List, Tuple
from tqdm import tqdm

class Adaptation:
    def __init__(self,
                 loss,
                 rand_percent=1,
                 batch_size=128,
                 eta=0.001,
                 num_pre_loss=10,
                 threshold=0.1):
        self.loss = loss
        self.rand_percent = rand_percent
        self.batch_size = batch_size
        self.eta = eta
        self.num_pre_loss = num_pre_loss
        self.threshold = threshold
        self.global_adapter = None
        self.local_adapter = None
        self.local_model = None

    def adaptive_local_aggregation(self, global_adapter, local_adapter, local_model, train_data, device):
        # randomly sample partial local training data
        rand_ratio =  rand_percent / 100
        rand_num = int(rand_ratio*len(train_data))
        rand_idx = random.randint(0, len(train_data)-rand_num)
        # print('rand_idx:', rand_idx, 'rand_num:', rand_num)
        subset_train_data = Subset(train_data, range(rand_idx, rand_idx+rand_num))
        rand_loader = DataLoader(subset_train_data,  batch_size, drop_last=False, shuffle=True, num_workers=12)


        # obtain the references of the parameters
        params_g = list(global_adapter.parameters())
        params = list(local_adapter.parameters())

        # deactivate at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return params

        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        # params_t only contains the parameters of the adapter
        params_t = list(model_t.base.adapter.parameters())
        for name, param in model_t.named_parameters():
            if 'adapter' not in name or 'global_adapter' in name:
                param.requires_grad = False

        # # only consider higher layers
        # params_p = params[- layer_idx:]
        # params_gp = params_g[- layer_idx:]
        # params_tp = params_t[- layer_idx:]

        # used to obtain the gradient of adapter
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.AdamW(params_t, lr=0)

        # initialize the weight to all ones in the beginning
        weights = [torch.ones_like(param.data).to(device) for param in params]
        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_t, params, params_g, weights):
            param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        while True:
            # print('ALA epochs:', cnt)
            for x, y in rand_loader:
                if type(x) == type([]):
                    x[0] = x[0].to( device)
                else:
                    x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                output = model_t(x)
                loss_value = loss(output, y) # modify according to the local objective
                loss_value.backward()

                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_t, params,
                                                        params_g,  weights):
                    weight.data = torch.clamp(
                        weight - eta * (param_t.grad * (param_g - param)), 0, 1)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_t, params,
                                                        params_g,  weights):
                    param_t.data = param + (param_g - param) * weight
                # print('weight:', weight[0])
                # print('Loss:', loss_value.item())

            losses.append(loss_value.item())
            cnt += 1

            # train the weight until convergence
            if len(losses) > num_pre_loss and np.std(losses[- num_pre_loss:]) <  threshold:
                print('Std:', np.std(losses[- num_pre_loss:]),
                    '\tALA epochs:', cnt)
                break

        start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params, params_t):
            param.data = param_t.data.clone()

        return params