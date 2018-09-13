import torch
from lyapunov.core import Train

from IPython import embed



class Regularized(Train):
    def __init__(self,manager):
        super(Regularized, self).__init__(manager)

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
        V = Vsum.data.numpy()[0]  # record minimax objective
        
        # 5. Compute gradients
        d_error_grad = torch.autograd.grad(d_error, self.m.D.parameters(), create_graph=True)
        g_error_grad = torch.autograd.grad(g_error, self.m.G.parameters(), create_graph=True)

        # 6. Compute squared norm of gradient and differentiate
        norm_d = torch.sum(torch.cat([torch.sum(g**2.) for g in d_error_grad]))
        norm_g = torch.sum(torch.cat([torch.sum(g**2.) for g in g_error_grad]))
        norm_d_grad = torch.autograd.grad(norm_d, self.m.G.parameters(), create_graph=True)
        gammaReg = [self.m.params['gamma']*g for g in norm_d_grad]

        # 7. Accumulate F(x_k)
        self.m.D.accumulate_gradient(d_error_grad) # compute/store gradients, but don't change params
        self.m.G.accumulate_gradient(g_error_grad)

        # 8. Add gamma*Reg to G's gradients
        self.m.G.accumulate_gradient(gammaReg) # compute/store gradients, but don't change params

        # 13. Step
        self.d_optimizer.step()  # Optimizes D's parameters; changes based on stored gradients from backward()
        self.g_optimizer.step()  # Optimizes G's parameters

        return norm_d.data.numpy()[0], norm_g.data.numpy()[0], V
