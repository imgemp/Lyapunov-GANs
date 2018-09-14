import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Uniform

from IPython import embed

# https://github.com/locuslab/gradient_regularized_gan/blob/master/gaussian-toy-regularized.py
# https://github.com/devnag/pytorch-generative-adversarial-networks/blob/master/gan_pytorch.py
# https://github.com/GKalliatakis/Delving-deep-into-GANs
# https://github.com/wiseodd/generative-models/blob/master/VAE/vanilla_vae/vae_pytorch.py
# https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py
# https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
# https://github.com/caogang/wgan-gp
# http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
# https://github.com/LMescheder/TheNumericsOfGANs
# http://www.inference.vc/my-notes-on-the-numerics-of-gans/
# https://github.com/poolio/unrolled_gan/blob/master/Unrolled%20GAN%20demo.ipynb
# https://github.com/locuslab/gradient_regularized_gan/blob/master/gaussian-toy-regularized.py
# https://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/

# https://github.com/pytorch/pytorch/releases/tag/v0.4.0
# http://pytorch.org/2018/04/22/0_4_0-migration-guide.html
# https://github.com/pytorch/pytorch/commit/bc7a41af7d541e64f8b8f7318a7a2248c0119632

 
class Data(object):
    '''
    Data Generator object. Main functionality is contained in `sample' method.
    '''
    def __init__(self):
        return None

    '''
    Takes batch size as input and returns torch tensor containing batch_size rows and
    num_feature columns
    '''
    def sample(self, batch_size):
        return None

    def plot_current(self):
        return None

    def plot_series(self):
        return None


class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        self.layers = None
            
    def forward(self,x):
        return None

    def init_weights(self):
        return None

    def get_param_data(self):
        return [p.data.detach() for p in self.parameters()]

    def set_param_data(self, data, req_grad=False):
        for p,d in zip(self.parameters(),data):
            if req_grad:
                p.data = d
            else:
                p.data = d.detach()

    def accumulate_gradient(self, grad):
        for p,g in zip(self.parameters(),grad):
            if p.grad is None:
                p.grad = g.detach()
            else:
                p.grad += g.detach()

    def multiply_gradient(self, scalar):
        for p in self.parameters():
            if p.grad is not None:
                p.grad *= scalar

    def zero_grad(self):  # overwriting zero_grad of super because it has a bug
        r"""Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            if p.grad is not None:
                # p.grad.detach_()  # replace this line - bug
                p.grad.detach()
                p.grad.zero_()

class DNet(nn.Module):
    # assumption: output_dim = 1
    def __init__(self):
        super(DNet, self).__init__()
        self.Dp_norm_d_grad = None
        self.zero_grads = None
        self.layers = None

    def forward(self,x):
        return None

    def init_weights(self):
        for layer in self.layers:
            nn.init.orthogonal(layer.weight.data, gain=0.8)
            layer.bias.data.zero_()

    def get_param_data(self):
        return [p.data.detach() for p in self.parameters()]

    def set_param_data(self, data, req_grad=False):
        for p,d in zip(self.parameters(),data):
            if req_grad:
                p.data = d
            else:
                p.data = d.detach()

    def accumulate_gradient(self, grad):
        for p,g in zip(self.parameters(),grad):
            if p.grad is None:
                p.grad = g.detach()
            else:
                p.grad += g.detach()

    def multiply_gradient(self, scalar):
        for p in self.parameters():
            if p.grad is not None:
                p.grad *= scalar

    def zero_grad(self):  # overwriting zero_grad of super because it has a bug
        r"""Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            if p.grad is not None:
                # p.grad.detach_()  # replace this line - bug
                p.grad.detach()
                p.grad.zero_()


class Manager(object):
    def __init__(self, data, D, G, params, to_gpu):
        self.data = data
        self.D = D
        self.G = G
        self.params = params
        self.to_gpu = to_gpu
        self.z_rand = Uniform(to_gpu(torch.tensor(0.0)), to_gpu(torch.tensor(1.0)))
        if params['divergence'] == 'JS' or params['divergence'] == 'standard':
            loss = nn.BCEWithLogitsLoss()
            self.criterion = lambda dec, label: -loss(dec, label)
        elif params['divergence'] == 'Wasserstein':
            self.criterion = lambda dec, label: torch.mean(dec*(2.*label-1.))  #loss(dec, label) #torch.sum(dec)  #torch.sum(dec*(2.*label-1.))

    def get_real(self, batch_size):
        return self.to_gpu(self.data.sample(batch_size))

    def get_z(self, batch_size, z_dim):
        z = self.z_rand.sample((batch_size,z_dim))
        return z

    def get_fake(self, batch_size, z_dim):
        return self.G(self.get_z(batch_size, z_dim))

    def get_decisions(self, data):
        decisions = []
        for datum in data:
            if isinstance(datum,np.ndarray):
                datum = self.to_gpu(Variable(torch.from_numpy(datum).float()))
            decisions += [self.D(datum).squeeze()]
        return decisions

    def get_V(self, batch_size, real_dec=None, fake_dec=None):
        res = []
        if real_dec is not None:
            V_real = self.criterion(real_dec, self.to_gpu(Variable(torch.ones(real_dec.shape[0]))))  # ones = true
            res += [V_real]
        if fake_dec is not None:
            V_fake = self.criterion(fake_dec, self.to_gpu(Variable(torch.zeros(fake_dec.shape[0]))))  # zeros = fake
            res += [V_fake]
            if self.params['divergence'] == 'standard':
                V_fake_mod = -self.criterion(fake_dec, self.to_gpu(Variable(torch.ones(fake_dec.shape[0]))))  # we want to fool, so pretend it's all genuine
                res += [V_fake_mod]
            elif self.params['divergence'] == 'JS' or self.params['divergence'] == 'Wasserstein':
                res += [V_fake]
            else:
                raise NotImplementedError(self.params['divergence'])
        return res


class Train(object):
    def __init__(self, manager):
        self.m = manager

        optimizers = []
        for net, ps in zip(['disc','gen'],[self.m.D.parameters(),self.m.G.parameters()]):
            optimizers += [optim.SGD(ps, lr=self.m.params[net+'_learning_rate'], momentum=0., weight_decay=0.)]
        self.d_optimizer, self.g_optimizer = optimizers

        # Note: SimgGD should always be last in maps list
        self.maps = [mp(manager).map for mp in self.m.params['maps']]
        self.cmap = self.compose(*self.maps)  # [f,g] becomes f(g(x))

        self.epsilon = self.m.params['psi_epsilon']

        self.req_aux = any(mp.req_aux for mp in self.m.params['maps'])

        if self.req_aux:
            # Initiate auxiliary parameters
            self.aux_d = [torch.zeros_like(p, requires_grad=False) for p in self.m.D.parameters()]
            self.aux_g = [torch.zeros_like(p, requires_grad=False) for p in self.m.G.parameters()]
        else:
            self.aux_d = None
            self.aux_g = None

        self.K = self.m.params['K']

        d_shapes = [list(p.shape) for p in self.m.D.parameters()]
        d_dims = [np.prod(sh) for sh in d_shapes]
        self.num_d = len(d_dims)
        g_shapes = [list(p.shape) for p in self.m.G.parameters()]
        g_dims = [np.prod(sh) for sh in g_shapes]
        self.num_g = len(g_dims)
        psi = np.vstack([np.eye(self.K), np.zeros((2*sum(d_dims)+2*sum(g_dims)-self.K, self.K))])
        psi_split = np.split(psi, indices_or_sections=np.cumsum(2*(d_dims+g_dims))[:-1], axis=0)

        Ks = range(self.K)
        self.psi_d = [[torch.FloatTensor(psi_split[i][:,k].reshape(*d_shapes[i]), requires_grad=False) for i in range(len(d_dims))] for k in Ks]
        self.psi_g = [[torch.FloatTensor(psi_split[i+len(d_dims)][:,k].reshape(*g_shapes[i]), requires_grad=False) for i in range(len(g_dims))] for k in Ks]
        if self.req_aux:
            self.psi_d_a = [[torch.FloatTensor(psi_split[i+len(d_dims)+len(g_dims)][:,k].reshape(*d_shapes[i]), requires_grad=False) for i in range(len(d_dims))] for k in Ks]
            self.psi_g_a = [[torch.FloatTensor(psi_split[i+2*len(d_dims)+len(g_dims)][:,k].reshape(*g_shapes[i]), requires_grad=False) for i in range(len(g_dims))] for k in Ks]

        self.lams = np.zeros(self.K)

        steps = [self.m.params['disc_learning_rate'], self.m.params['gen_learning_rate']] * (1 + self.req_aux)
        self.delta_ts = [step*self.m.params['LE_freq'] for step in steps]

        self.real_data_fixed = self.m.get_real(self.m.params['batch_size'])
        self.fake_z_fixed = self.m.get_z(self.m.params['batch_size'], self.m.params['z_dim'])

    def train_op(self, it):
        self.m.D.zero_grad()
        self.m.G.zero_grad()

        # 1. Record x_k
        Gp = self.m.G.get_param_data()
        Dp = self.m.D.get_param_data()

        if self.req_aux:
            # 2. Record aux
            daux = [p.data.detach() for p in self.aux_d]
            gaux = [p.data.detach() for p in self.aux_g]

        # 3. Get real data and samples from p(z) to pass to generator
        if it == self.m.params['start_lam_it']:
            # Reinitialize maps with ncreased batch size for reduced stochasticity, i.e., ~deterministic
            self.m.params['batch_size'] *= self.m.params['LE_batch_mult']
        # real_data = self.m.get_real(self.m.params['batch_size'])
        # fake_z = self.m.get_z(self.m.params['batch_size'], self.m.params['z_dim'])
        real_data = self.real_data_fixed
        fake_z = self.fake_z_fixed

        # 4. Evaluate Map
        _, _, map_d, map_g, map_aux_d, map_aux_g, V, norm_d, norm_g = self.cmap([real_data, self.m.G(fake_z), self.aux_d, self.aux_g])

        # 5. Compute Lyapunov exponents after initial "burn-in"
        if it >= self.m.params['start_lam_it']:
            # 5a-e. Loop over psis, perturb, and update: psi = [F(x_k + self.epsilon*psi) - F(x_k)]/self.epsilon
            for k in range(self.K):
                # 5a. Perturb parameters
                for i,p in enumerate(self.m.D.parameters()):
                    p.data = (p.data + self.epsilon*self.psi_d[k][i]).detach()
                for i,p in enumerate(self.m.G.parameters()):
                    p.data = (p.data + self.epsilon*self.psi_g[k][i]).detach()
                if self.req_aux:
                    for i,a in enumerate(self.aux_d):
                        a.data = (a.data + self.epsilon*self.psi_d_a[k][i]).detach()
                    for i,a in enumerate(self.aux_g):
                        a.data = (a.data + self.epsilon*self.psi_g_a[k][i]).detach()
                # 5b. Evaluate map
                _, _, map_psi_d, map_psi_g, map_psi_aux_d, map_psi_aux_g, _, _, _ = self.cmap([real_data, self.m.G(fake_z), self.aux_d, self.aux_g])
                # 5c. Update psi[k]
                for i,psi in enumerate(self.psi_d[k]):
                    psi.sub_(self.m.params['disc_learning_rate']*(map_psi_d[i]-map_d[i])/self.epsilon)
                for i,psi in enumerate(self.psi_g[k]):
                    psi.sub_(self.m.params['gen_learning_rate']*(map_psi_g[i]-map_g[i])/self.epsilon)
                if self.req_aux:
                    for i,psi in enumerate(self.psi_d_a[k]):
                        psi.sub_(self.m.params['disc_learning_rate']*(map_psi_aux_d[i]-map_aux_d[i])/self.epsilon)
                    for i,psi in enumerate(self.psi_g_a[k]):
                        psi.sub_(self.m.params['gen_learning_rate']*(map_psi_aux_g[i]-map_aux_g[i])/self.epsilon)
                # 5d. Reset weights to x_k
                self.m.D.set_param_data(Dp)
                self.m.G.set_param_data(Gp)
                # 5e. Reset auxiliary vars to aux_k
                if self.req_aux:
                    for i,a in enumerate(self.aux_d):
                        a.data = daux[i].detach()
                    for i,a in enumerate(self.aux_g):
                        a.data = gaux[i].detach()

            # 5f-i. Orthogonalize psis, compute norms, normalize psis, and update Lyapunov exponents
            if (it+1 - self.m.params['start_lam_it']) % self.m.params['LE_freq'] == 0:
                # 5f. Compute norms of columns of Psi
                psi_d_norms_squared = [sum([torch.sum(psi**2) for psi in psi_k]) for psi_k in self.psi_d]
                psi_g_norms_squared = [sum([torch.sum(psi**2) for psi in psi_k]) for psi_k in self.psi_g]
                norms = [psi_d_norms_squared, psi_g_norms_squared]
                psis = [self.psi_d, self.psi_g]
                if self.req_aux:
                    psi_d_a_norms_squared = [sum([torch.sum(psi**2) for psi in psi_k]) for psi_k in self.psi_d_a]
                    psi_g_a_norms_squared = [sum([torch.sum(psi**2) for psi in psi_k]) for psi_k in self.psi_g_a]
                    norms += [psi_d_a_norms_squared, psi_g_a_norms_squared]
                    psis += [self.psi_d_a, self.psi_g_a]
                psi_norms = [torch.sqrt(sum(norm)) for norm in zip(*norms)]
                psis_sh = [functools.reduce(lambda x,y: x+y, ps) for ps in zip(*psis)]  # psis is 4 parameter dims x K x num weights
                # this turns it into K x total num weights

                # 5g. Gram Schmidt
                psis_temp = list(zip(*self.GramSchmidt(psis_sh, psi_norms)))
                self.psi_d = [list(el) for el in zip(*psis_temp[:self.num_d])]
                self.psi_g = [list(el) for el in zip(*psis_temp[self.num_d:self.num_d+self.num_g])]
                if self.req_aux:
                    self.psi_d_a = [list(el) for el in zip(*psis_temp[self.num_d+self.num_g:2*self.num_d+self.num_g])]
                    self.psi_g_a = [list(el) for el in zip(*psis_temp[2*self.num_d+self.num_g:])]

                # 5h. Normalize psis
                for k in range(self.K):
                    for i in range(len(self.psi_d[k])):
                        self.psi_d[k][i] /= psi_norms[k]
                    for i in range(len(self.psi_g[k])):
                        self.psi_g[k][i] /= psi_norms[k]
                if self.req_aux:
                    for k in range(self.K):
                        for i in range(len(self.psi_d_a[k])):
                            self.psi_d_a[k][i] /= psi_norms[k]
                        for i in range(len(self.psi_g_a[k])):
                            self.psi_g_a[k][i] /= psi_norms[k]

                # 5i. Update Lyapunov exponents (lambdas)
                new_lam_dts = [np.log(psi_norm.item()) for psi_norm in psi_norms]  # actually equal to lambda*dt
                Ts = [dt*(it - self.m.params['start_lam_it']) for dt in self.delta_ts]
                self.lams = np.array([(lam*T+new_lam_dt)/(T+dt) for lam, new_lam_dt, T, dt in zip(self.lams, new_lam_dts, Ts, self.delta_ts)])

        # 6. Accumulate F(x_k)
        self.m.D.accumulate_gradient(map_d) # compute/store map, but don't change params
        self.m.G.accumulate_gradient(map_g)

        # 7. Update network parameters
        self.d_optimizer.step()  # Optimizes D's parameters; changes based on stored map from backward()
        self.g_optimizer.step()  # Optimizes G's parameters

        if self.req_aux:
            # 8. Update auxiliary parameters
            for i,a in enumerate(self.aux_d):
                a.sub_(self.m.params['disc_learning_rate']*map_aux_d[i])  # in place add: a = a + lr*map
            for i,a in enumerate(self.aux_g):
                a.sub_(self.m.params['gen_learning_rate']*map_aux_g[i])  # in place add: a = a + lr*map

        return self.lams, norm_d.item(), norm_g.item(), V.item()

    @staticmethod
    def GramSchmidt(A, norms):
        # only orthogonalizes, no normalization
        # assumes A is list of vectors
        if len(A) > 1:
            for i in range(len(A)):
                vi = A[i]
                proj = [torch.zeros_like(vip) for vip in vi]
                for j in range(i):
                    uj = A[j]
                    viuj = sum([torch.sum(vip*ujp) for vip, ujp in zip(vi,uj)])
                    factor = viuj/norms[j]
                    for p in range(len(proj)):
                        proj[p] += factor*uj[p]
                for p in range(len(A[i])):
                    A[i][p] -= proj[p]
        return A

    @staticmethod
    def compose(*functions):
        '''
        https://mathieularose.com/function-composition-in-python/
        '''
        return functools.reduce(lambda f, g: lambda x: f(g(*x)), functions, lambda x: x)
        

class Map(object):
    req_aux = False
    def __init__(self, manager):
        self.m = manager

    def map(self, F=None):
        # return [d_map, g_map, V]
        return None, None, None
