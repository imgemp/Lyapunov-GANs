import pickle
import torch
from lyapunov.core import Map

from IPython import embed


class Rescale(Map):
    req_aux = False
    def __init__(self,manager):
        super(Rescale, self).__init__(manager)
        self.epsilon = 1e-10
        self.load_scale()

    def load_scale(self):
        aux_d = pickle.load(open(self.m.params['disc_aux_weights'], 'rb'))
        aux_g = pickle.load(open(self.m.params['gen_aux_weights'], 'rb'))
        self.d_scale = [self.m.to_gpu(torch.from_numpy(w)) for w in aux_d]
        self.g_scale = [self.m.to_gpu(torch.from_numpy(w)) for w in aux_g]

    def map(self, aux_d, aux_g, d_error_grad, g_error_grad, d_aux_map, g_aux_map, V, norm_d, norm_g):
        # 1. Rescale main map
        d_map = [g/(torch.sqrt(a+self.epsilon)) for a, g in zip(self.d_scale, d_error_grad)]
        g_map = [g/(torch.sqrt(a+self.epsilon)) for a, g in zip(self.g_scale, g_error_grad)]

        return [aux_d, aux_g, d_map, g_map, d_aux_map, g_aux_map, V, norm_d, norm_g]