import torch
from lyapunov.core import Map

from IPython import embed


class RMSProp(Map):
    req_aux = True
    def __init__(self,manager):
        super(RMSProp, self).__init__(manager)

    def map(self, aux_d, aux_g, d_error_grad, g_error_grad, d_aux_map, g_aux_map, V, norm_d, norm_g):
        # 1. Compute auxiliary map
        gamma = 0.9
        d_a_map = [(1-gamma)/self.m.params['disc_learning_rate']*(a-g**2.) for a, g in zip(aux_d, d_error_grad)]
        g_a_map = [(1-gamma)/self.m.params['gen_learning_rate']*(a-g**2.) for a, g in zip(aux_g, g_error_grad)]

        # 2. Compute main map
        epsilon = 1e-10
        # d_map = [g/torch.sqrt(gamma*a+(1-gamma)*g**2.+epsilon) for a, g in zip(aux_d, d_error_grad)]
        # g_map = [g/torch.sqrt(gamma*a+(1-gamma)*g**2.+epsilon) for a, g in zip(aux_g, g_error_grad)]
        d_map = [g/(torch.sqrt(gamma*a+(1-gamma)*g**2.)+epsilon) for a, g in zip(aux_d, d_error_grad)]
        g_map = [g/(torch.sqrt(gamma*a+(1-gamma)*g**2.)+epsilon) for a, g in zip(aux_g, g_error_grad)]

        return [aux_d, aux_g, d_map, g_map, d_a_map, g_a_map, V, norm_d, norm_g]
