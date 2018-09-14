import torch
from lyapunov.core import Map

from IPython import embed


class Consensus(Map):
    def __init__(self,manager):
        super(Consensus, self).__init__(manager)

    def map(self, aux_d, aux_g, d_error_grad, g_error_grad, d_aux, g_aux, V, norm_d, norm_g):
        # 1. Compute squared norm of gradient and differentiate
        norm = 0.5*(norm_d+norm_g)
        # if discriminator last layer is linear and div is Wasserstein then the Discriminator bias
        # (constant weight vector) disappears from minimax objective
        norm_d_grad = torch.autograd.grad(norm, self.m.D.parameters(), create_graph=True, allow_unused=True)
        norm_g_grad = torch.autograd.grad(norm, self.m.G.parameters(), create_graph=True)
        gammaJTF_d = [self.m.params['gamma']*g if g is not None else torch.zeros_like(p) for g,p in zip(norm_d_grad, self.m.D.parameters())]
        gammaJTF_g = [self.m.params['gamma']*g for g in norm_g_grad]

        # 2. Sum terms
        d_map = [a+b for a, b in zip(d_error_grad, gammaJTF_d)]
        g_map = [a+b for a, b in zip(g_error_grad, gammaJTF_g)]
        
        return [aux_d, aux_g, d_map, g_map, d_aux, g_aux, V, norm_d, norm_g]
