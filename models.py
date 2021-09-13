import logging
import time
import copy

from abc import ABC, abstractmethod
import math
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.autograd as autograd
from torch import nn

from ref import make_tensor

LOG_INTERVAL = 300

class IRMBase(ABC):
    '''Base class for all IRM implementations'''
    def __init__(self):
        '''Ptype: p for regression problem, cls for binary classification'''
        self.model = None


    @abstractmethod
    def train(self, data, y_all, environments, seed, args):
        pass

    @abstractmethod
    def predict(self, data, phi_params):
        pass

    def mean_nll(self, logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)
    
    def mean_accuracy(self, logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def penalty(self, logits, y):
        scale = torch.tensor(1.).requires_grad_()
        loss = self.mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

class LinearInvariantRiskMinimization(IRMBase):
    """Object Wrapper around IRM"""

    def __init__(self):
        super().__init__()

    def train(self, envs, seed, args, batching=False):
        ''':param envs: if batching=False two possibilities list of training env
                        dstructs {'x':data (npArray), 'y':labels (npArray)}.
                        If true list of dataloaders of each envs'''

        def update_params(envs, model, optimizer, args):
            ''':param envs: list of training env data structures, of form
                             {'x':data (npArray) or (torch.Tensor), 'y':labels (npArray) or torch.Tensor}'''
            e_comp = {}
            for i, e in enumerate(envs):
                e_comp[i] = {}
                data, y_all = e['x'], e['y']
                logits = (make_tensor(data) @ phi @ w).squeeze() #Note - for given loss this is raw output
                labels = make_tensor(y_all).squeeze()
                e_comp[i]['nll'] = self.mean_nll(logits, labels)
                e_comp[i]['acc'] = self.mean_accuracy(logits, labels)
                e_comp[i]['penalty'] = self.penalty(logits, labels)

            train_nll = torch.stack([e_comp[e]['nll'] \
                                     for e in e_comp]).mean()
            train_acc = torch.stack([e_comp[e]['acc']
                                     for e in e_comp]).mean()
            train_penalty = torch.stack([e_comp[e]['penalty']
                                         for e in e_comp]).mean()
            loss = train_nll.clone()

            #Regularize the weights
            weight_norm = phi.norm().pow(2)
            loss += args['l2_reg'] * weight_norm

            #Add the invariance penalty
            penalty_weight = (args['pen_wgt']
                              if step >= args['penalty_anneal_iters'] else 1.0)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0: # Rescale big loss
                loss /= penalty_weight

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss, train_nll, train_acc, train_penalty

        #######
        if batching:
            dim_x = envs[0].dataset.dim
        else:
            dim_x = envs[0]['x'].shape[1]

        errors = []
        penalties = []
        losses = []

        phi = torch.nn.Parameter(torch.empty(dim_x, 1 \
                                            ).normal_(\
                                            generator=torch.manual_seed(seed)))
        w = torch.ones(1, 1)
        w.requires_grad = True
        optimizer = torch.optim.Adam([phi], lr=args['lr'])

        logging.info('[step, train nll, train acc, train penalty, test acc]')

        #Start the training loop
        t_last = time.time()
        for step in tqdm(range(args['n_iterations'])):
            if batching:
                nbatch = math.ceil((len(envs[0])/envs[0].batch_size))  #Assume all envs have the same #batches/iter
                for input_envs in zip(*envs):
                    # import pdb; pdb.set_trace()
                    loss, train_nll, train_acc, train_penalty = \
                        update_params(input_envs, phi, optimizer, {'l2_reg':args['l2_reg'], \
                                        'pen_wgt':args['pen_wgt'], \
                                        'penalty_anneal_iters':args['penalty_anneal_iters']})

            else:
                loss, train_nll, train_acc, train_penalty = \
                              update_params(envs, phi, optimizer, {'l2_reg':args['l2_reg'], \
                                            'pen_wgt':args['pen_wgt'], \
                                            'penalty_anneal_iters':args['penalty_anneal_iters']})

            #Printing and Logging
            if (step % 5 == 0) or ((time.time() - t_last) > LOG_INTERVAL):
                logging.info('step: {}, loss: {}, acc: {}, ipen: {}, time: {:+.2f}'.format(\
                                 np.int32(step), \
                                 train_nll.detach().cpu().numpy(), \
                                 train_acc.detach().cpu().numpy(), \
                                 train_penalty.detach().cpu().numpy(), \
                                 (time.time() - t_last))
                             )
                t_last = time.time()

            errors.append(train_nll.detach().numpy())
            penalties.append(train_penalty.detach().numpy())
            losses.append(loss.detach().numpy())

        self.model = phi
        return errors, penalties, losses

    def predict(self, data):
        '''
        :param data: the dataset (nparray)
        :param phi_params: The state dict of the MLP'''
        def sigmoid(x):
            return 1/(1 + np.exp(-x))

        assert self.model is not None
        #Handle case of no data
        if data.shape[0] == 0:
            assert False

        phi = self.model.detach().numpy()
        w = np.ones([phi.shape[1], 1])
        logits = (data @ (phi @ w).ravel()).squeeze()
        return sigmoid(logits)
