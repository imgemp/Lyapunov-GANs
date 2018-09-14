import torch
from lyapunov.core import Map

from IPython import embed


class SimGD(Map):
    def __init__(self,manager):
        super(SimGD, self).__init__(manager)

    def map(self, real_data, fake_data, aux_d=None, aux_g=None, d_error_grad=None, g_error_grad=None, d_aux=None, g_aux=None, V=None, norm_d=None, norm_g=None):
        # 1. Discriminate real and fake data
        real_decision, fake_decision = self.m.get_decisions([real_data, fake_data])

        # 2. Define and record losses
        V_real, V_fake, V_fake_g = self.m.get_V(self.m.params['batch_size'], real_decision, fake_decision)
        Vsum = V_real + V_fake
        d_error = -Vsum
        g_error = V_fake_g

        # 3. Compute gradients
        d_error_grad = torch.autograd.grad(d_error, self.m.D.parameters(), create_graph=True)
        g_error_grad = torch.autograd.grad(g_error, self.m.G.parameters(), create_graph=True)
        norm_d = sum([torch.sum(g**2.) for g in d_error_grad])
        norm_g = sum([torch.sum(g**2.) for g in g_error_grad])

        return [aux_d, aux_g, d_error_grad, g_error_grad, None, None, Vsum, norm_d, norm_g]
