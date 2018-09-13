import torch
from lyapunov.core import Train

from IPython import embed


params = dict(
    batch_size=512,
    divergence='JS',  # JS
    disc_optim='RMSProp',  # Adam
    disc_learning_rate=1e-4,
    disc_n_hidden=16,  # 128
    disc_n_layer=4,  # 2
    gen_optim='RMSProp', # Adam
    gen_learning_rate=1e-4,
    gen_n_hidden=16,  # 128
    gen_n_layer=4,  # 2
    betas=(0.5,0.999),
    epsilon=1e-8,
    max_iter=50000,  # 100001
    viz_every=500,  # 1000
    z_dim=16,  # 256
    x_dim=2,
    gamma=0.5,
    saveto='results/reg_alt/laptop/'
)

class Regularized_Alt(Train):
    def __init__(self,manager):
        super(Regularized_Alt, self).__init__(manager)

    def train_op(self):
        ### Discimrinator Step
        self.m.D.zero_grad()

        # D1. Gather real and fake (generated) data
        real_data = self.m.get_real(self.m.params['batch_size'])
        fake_data = self.m.get_fake(self.m.params['batch_size'], self.m.params['z_dim'])

        # D2. Discriminate real and fake data
        real_decision, fake_decision = self.m.get_decisions([real_data, fake_data])

        # D3. Define and record losses
        V_real, V_fake, _ = self.m.get_V(self.m.params['batch_size'], real_decision, fake_decision)
        Vsum = V_real + V_fake
        d_error = -Vsum

        # D4. Accumulate F(x_k)
        d_error_grad = torch.autograd.grad(d_error, self.m.D.parameters(), create_graph=True)
        self.m.D.accumulate_gradient(d_error_grad) # compute/store gradients, but don't change params
        self.d_optimizer.step()  # Optimizes D's parameters; changes based on stored gradients from backward()

        ### Generator Step
        self.m.D.zero_grad()
        self.m.G.zero_grad()

        # G1. Gather real and fake (generated) data
        real_data = self.m.get_real(self.m.params['batch_size'])
        fake_data = self.m.get_fake(self.m.params['batch_size'], self.m.params['z_dim'])

        # G2. Discriminate real and fake data
        real_decision, fake_decision = self.m.get_decisions([real_data, fake_data])

        # G3. Define and record losses
        V_real, V_fake, V_fake_g = self.m.get_V(self.m.params['batch_size'], real_decision, fake_decision)
        Vsum = V_real + V_fake
        d_error = -Vsum
        g_error = V_fake_g
        V = Vsum.data.numpy()[0]  # record minimax objective

        # G5. Compute gradients
        d_error_grad = torch.autograd.grad(d_error, self.m.D.parameters(), create_graph=True)
        g_error_grad = torch.autograd.grad(g_error, self.m.G.parameters(), create_graph=True)

        # G6. Compute squared norm of gradient and differentiate
        norm_d = torch.sum(torch.cat([torch.sum(g**2.) for g in d_error_grad]))
        norm_g = torch.sum(torch.cat([torch.sum(g**2.) for g in g_error_grad]))
        norm_d_grad = torch.autograd.grad(norm_d, self.m.G.parameters(), create_graph=True)
        gammaReg = [self.m.params['gamma']*g for g in norm_d_grad]

        # G7. Accumulate F(x_k)
        self.m.G.accumulate_gradient(g_error_grad)

        # G8. Add gamma*Reg to gradients
        self.m.G.accumulate_gradient(gammaReg) # compute/store gradients, but don't change params

        self.g_optimizer.step()  # Optimizes G's parameters

        return norm_d.data.numpy()[0], norm_g.data.numpy()[0], V
