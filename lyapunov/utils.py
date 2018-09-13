"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import pickle
import numpy as np
import torch


def gpu_helper(gpu):
    if gpu >= -1:
        def to_gpu(x):
            x = x.cuda()
            return x
        return to_gpu
    else:
        def no_op(x):
            return x
        return no_op

def save_weights(module,file):
    weights = []
    for p in module.parameters():
        weights += [p.cpu().data.numpy()]
    pickle.dump(weights,open(file,'wb'))

def load_weights(module,file):
    weights = pickle.load(open(file,'rb'))
    for p,w in zip(module.parameters(),weights):
        p.data = torch.from_numpy(w)
    return weights
