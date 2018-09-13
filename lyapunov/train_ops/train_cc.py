import torch
from lyapunov.core import Train

from IPython import embed


params = dict(
    batch_size=512,
    divergence='JS',  # JS
    disc_optim='RMSProp',  # Adam
    disc_learning_rate=1e-3,
    disc_n_hidden=16,  # 128
    disc_n_layer=4,  # 2
    gen_optim='RMSProp', # Adam
    gen_learning_rate=1e-3,
    gen_n_hidden=16,  # 128
    gen_n_layer=4,  # 2
    betas=(0.5,0.999),
    epsilon=1e-8,
    max_iter=50000,  # 100001
    viz_every=500,  # 1000
    z_dim=16,  # 256
    x_dim=2,
    gamma=10.,
    gammaT=10.,
    step=1e-3,
    saveto='results/cc/laptop/'
)

class CrossCurl(Train):
    def __init__(self,manager):
        super(CrossCurl, self).__init__(manager)

    def train_op(self):
        self.m.D.zero_grad()
        self.m.G.zero_grad()

        # 1. Record x_k
        Gp = self.m.G.get_param_data()
        Dp = self.m.D.get_param_data()

        # 2. Gather real and fake (generated) data
        real_data = self.m.get_real(self.m.params['batch_size'])
        fake_data = self.m.get_fake(self.m.params['batch_size'], self.m.params['z_dim'])
        
        # 3. Discriminate real and fake data
        real_decision, fake_decision = self.m.get_decisions([real_data, fake_data])

        # 4. Define and record losses
        V_real, V_fake, V_fake_g = self.m.get_V(self.m.params['batch_size'], real_decision, fake_decision)
        Vsum = V_real + V_fake
        d_error = -Vsum
        g_error = V_fake_g
        V = Vsum.item()  # record minimax objective

        # 5. Compute gradients
        d_error_grad = torch.autograd.grad(d_error, self.m.D.parameters(), create_graph=True)
        g_error_grad = torch.autograd.grad(g_error, self.m.G.parameters(), create_graph=True)

        # 6. Compute squared norm of gradient and differentiate
        norm_d = sum([torch.sum(g**2.) for g in d_error_grad])
        norm_g = sum([torch.sum(g**2.) for g in g_error_grad])
        norm = 0.5*(norm_d+norm_g)
        if self.m.params['divergence'] == 'Wasserstein':
            Dparams_reachable, zero_grads = self.m.D.wasserstein_hack()
            norm_d_grad = torch.autograd.grad(norm, Dparams_reachable, retain_graph=True) + zero_grads
        else:
            norm_d_grad = torch.autograd.grad(norm, self.m.D.parameters(), retain_graph=True)
        norm_g_grad = torch.autograd.grad(norm, self.m.G.parameters(), retain_graph=True)
        gammaJTF_d = [self.m.params['gammaT']*g for g in norm_d_grad]
        gammaJTF_g = [self.m.params['gammaT']*g for g in norm_g_grad]

        norm_d = norm_d.item() # float(norm_d.cpu().detach().numpy())
        norm_g = norm_g.item() # float(norm_g.cpu().detach().numpy())

        # 7. Accumulate F(x_k) and take a step (i.e., x_k - step*F(x_k))
        self.m.D.accumulate_gradient(d_error_grad) # compute/store gradients, but don't change params
        self.m.G.accumulate_gradient(g_error_grad)

        self.d_sgd.step()  # Optimizes D's parameters; changes based on stored gradients from backward()
        self.g_sgd.step()  # Optimizes G's parameters

        ### DO NOT ZERO GRADIENT - WILL USE F(x_k) ------ (I - gamma*(J-J^T))F(x_k)
        if self.m.params['kappa'] == 0.:
            self.m.D.zero_grad()
            self.m.G.zero_grad()
        elif self.m.params['kappa'] != 1.:
          self.m.D.multiply_gradient(self.m.params['kappa'])
          self.m.G.multiply_gradient(self.m.params['kappa'])

        # 8. Discriminate real and fake data using new discriminator weights (x_k+1)
        real_decision, fake_decision = self.m.get_decisions([real_data, fake_data])

        # 9. Define losses
        V_real, V_fake, V_fake_g = self.m.get_V(self.m.params['batch_size'], real_decision, fake_decision)
        Vsum = V_real + V_fake
        d_error = -Vsum
        g_error = V_fake_g
        
        # 10. Compute gradient, F(x_k+1), to approximate -JF = [F(x_k+1)-F(x_k)]/step
        d_error_grad_plus1 = torch.autograd.grad(d_error, self.m.D.parameters(), retain_graph=True)
        g_error_grad_plus1 = torch.autograd.grad(g_error, self.m.G.parameters())
        rescale = self.m.params['gamma']/self.m.params['step']
        mgammaJF_d = [rescale*(g_plus1-g) for g, g_plus1 in zip(d_error_grad, d_error_grad_plus1)]
        mgammaJF_g = [rescale*(g_plus1-g) for g, g_plus1 in zip(g_error_grad, g_error_grad_plus1)]

        # 11. Accumulate -gamma*JF
        self.m.D.accumulate_gradient(mgammaJF_d) # compute/store gradients, but don't change params
        self.m.G.accumulate_gradient(mgammaJF_g)

        # 11. Accumulate gamma*JTF
        self.m.D.accumulate_gradient(gammaJTF_d) # compute/store gradients, but don't change params
        self.m.G.accumulate_gradient(gammaJTF_g)

        # 12. Reset weights to x_k
        self.m.D.set_param_data(Dp)
        self.m.G.set_param_data(Gp)

        # 13. Step
        self.d_optimizer.step()  # Optimizes D's parameters; changes based on stored gradients from backward()
        self.g_optimizer.step()  # Optimizes G's parameters

        return norm_d, norm_g, V
