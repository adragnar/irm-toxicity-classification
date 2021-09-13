'Variety of Utility Functions'

import logging
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F

def pred_binarize(v, b=0.5):
    '''Convert all values to 0 if <0.5, 1 otherwise
    :param v: npArray with dim=1
    :param b: binarization threshold
    :return npArray with dim=1'''
    def thresh(x):
        if (x >= b): return 1
        else: return 0
    if v.shape == ():
        return thresh(v)
    else:
        return np.array([thresh(e) for e in v])

def compute_loss(pred, ground, ltype='MSE', b=0.5):
    '''Compute loss between two prediction vectors
    :param pred: The final predictions (not logits) of classifier.
    :param ground: The ground truth labels
    :param ltype: what evaluation metric to compute
    :param b: binarization threshold
    '''
    
    #Inputs can be any sort of vector - normalize dims
    if (pred.shape == ()) and (ground.shape == ()):
        import pdb; pdb.set_trace()
        pred, ground = np.expand_dims(pred, axis=0), np.expand_dims(ground, axis=0)
    elif (pred.shape == (1,)) and (ground.shape == (1,)):
        pass
    else:
        pred, ground = pred.squeeze(), ground.squeeze()

    try:
        assert (pred.shape == ground.shape) and (len(pred.shape) == 1)
    except:
        raise ValueError('Inputs are not the right dimensions')

    if ltype == 'MSE':
        return float(F.mse_loss(torch.tensor(pred).float(), torch.tensor(ground).float()).numpy())
    elif ltype == 'BCE':
        return float(F.binary_cross_entropy(torch.tensor(pred).float(), torch.tensor(ground).float()).numpy())
    elif ltype == 'ACC':
        pred = pred_binarize(pred, b=b)
        ground = pred_binarize(ground, b=b)
        return float(1 - np.mean(np.abs(pred - ground)))  #
    elif ltype == 'CONF':
        pred = pred_binarize(pred, b=b)
        return confusion_matrix(ground, pred)

def evaluate(envs, model, b=0.5, ltype=['ACC']):
    ''':param envs - list of dataloaders, each with data from diff env
       :param model: a base model with .predict fnc'''

    tot_samples = sum([len(dl.dataset) for dl in envs])

    acc = 0
    loss = 0
    conf_mat = np.zeros((2,2))
    for dl in envs:
        for batch_idx, sample_batch in enumerate(dl):
            probs = model.predict(sample_batch['x'].detach().numpy())
            labels = sample_batch['y'].detach().numpy().squeeze()
            if (probs.shape == ()) and (labels.shape == ()):
                probs, labels = np.expand_dims(probs, axis=0), np.expand_dims(labels, axis=0)
            if 'ACC' in ltype:
                batch_acc = compute_loss(probs, labels, ltype='ACC', b=b)
                acc += float( batch_acc * \
                           len(labels) / tot_samples)
        
            if 'BCE' in ltype:
                batch_loss = compute_loss(probs, labels, ltype='BCE', b=b)
                loss += float(batch_loss * \
                              len(labels)/tot_samples)
            if 'CONF' in ltype:
                conf_mat += compute_loss(probs, labels, ltype='CONF', b=b)

            if np.isnan(batch_acc) or np.isnan(batch_loss):
                logging.debug('Nan loss or acc - probs {}'.format(str(probs)))

    results = []
    if 'BCE' in ltype:
        results.append(loss)
    if 'ACC' in ltype:
        results.append(acc)
    if 'CONF' in ltype:
        results.append(conf_mat)
    return tuple(results)

def make_tensor(arr):
    '''Convert either np array or arbitrary pytorch tensor into a float tensor'''
    if type(arr) == torch.Tensor:
        return arr.float()
    elif type(arr) == np.ndarray:
        return torch.from_numpy(arr).float()
    else:
        raise Exception('Unimplemented')
