import torch
from lyapunov.core import Map

from IPython import embed


class RMSProp(Map):
    req_aux = True
    def __init__(self,manager):
        super(RMSProp, self).__init__(manager)
        self.alpha = 0.9
        self.epsilon = 1e-10

    def map(self, aux_d, aux_g, d_error_grad, g_error_grad, d_aux_map, g_aux_map, V, norm_d, norm_g):
        # 1. Compute auxiliary map
        d_a_map = [(1-self.alpha)/self.m.params['disc_learning_rate']*(a-g**2.) for a, g in zip(aux_d, d_error_grad)]
        g_a_map = [(1-self.alpha)/self.m.params['gen_learning_rate']*(a-g**2.) for a, g in zip(aux_g, g_error_grad)]

        # 2. Compute main map
        d_map = [g/(torch.sqrt(self.alpha*a+(1-self.alpha)*g**2.+self.epsilon)) for a, g in zip(aux_d, d_error_grad)]
        g_map = [g/(torch.sqrt(self.alpha*a+(1-self.alpha)*g**2.+self.epsilon)) for a, g in zip(aux_g, g_error_grad)]

        return [aux_d, aux_g, d_map, g_map, d_a_map, g_a_map, V, norm_d, norm_g]