import functools
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Uniform

from utils import detach_all

from IPython import embed

 
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

    def init_weights_from_file(self, filepath):
        weights = pickle.load(open(filepath, 'rb'))
        for p,w in zip(self.parameters(), weights):
            p.data = torch.from_numpy(w)
        print('NOTE: Loaded generator weights from pickle file.')

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
        return None

    def init_weights_from_file(self, filepath):
        weights = pickle.load(open(filepath, 'rb'))
        for p,w in zip(self.parameters(), weights):
            p.data = torch.from_numpy(w)
        print('NOTE: Loaded discriminator weights from pickle file.')

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
    def __init__(self, data, D, G, params, to_gpu, to_gpu_alt):
        self.data = data
        self.D = D
        self.G = G
        self.params = params
        self.to_gpu = to_gpu
        self.to_gpu_alt = to_gpu_alt
        self.z_rand = Uniform(to_gpu(torch.tensor(0.0)), to_gpu(torch.tensor(1.0)))
        if params['divergence'] == 'JS' or params['divergence'] == 'standard':
            loss = nn.BCEWithLogitsLoss()
            self.criterion = lambda dec, label: -loss(dec, label)
        elif params['divergence'] == 'Wasserstein':
            self.criterion = lambda dec, label: torch.mean(dec*(2.*label-1.))  #loss(dec, label) #torch.sum(dec)  #torch.sum(dec*(2.*label-1.))
        elif params['divergence'] == 'DummyTest':
            self.criterion = lambda dec, label: torch.mean(dec*(2.*label-1.))**2
        else:
            raise NotImplementedError(params['divergence']+' divergence not implemented.')

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
            elif self.params['divergence'] == 'JS' or self.params['divergence'] == 'Wasserstein' or self.params['divergence'] == 'DummyTest':
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
            try:
                aux_d = pickle.load(open(self.m.params['disc_aux_weights']), 'rb')
                self.aux_d = [torch.from_numpy(w, requires_grad=False) for w in aux_d]
                print('NOTE: Loaded auxiliary discriminator weights from pickle file.')
            except:
                self.aux_d = [torch.zeros_like(p, requires_grad=False) for p in self.m.D.parameters()]
            try:
                aux_g = pickle.load(open(self.m.params['disc_aux_weights']), 'rb')
                self.aux_g = [torch.from_numpy(w, requires_grad=False) for w in aux_g]
                print('NOTE: Loaded auxiliary generator weights from pickle file.')
            except:
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

        print('NOTE: Discriminator dimensionality = {:d}.'.format(sum(d_dims)))
        print('NOTE: Generator dimensionality = {:d}.'.format(sum(g_dims)))

        Ks = range(self.K)
        self.psi_d = [[self.m.to_gpu_alt(torch.tensor(psi_split[i][:,k].reshape(*d_shapes[i]).astype('float32'), requires_grad=False)) for i in range(len(d_dims))] for k in Ks]
        self.psi_g = [[self.m.to_gpu_alt(torch.tensor(psi_split[i+len(d_dims)][:,k].reshape(*g_shapes[i]).astype('float32'), requires_grad=False)) for i in range(len(g_dims))] for k in Ks]
        if self.req_aux:
            self.psi_d_a = [[self.m.to_gpu_alt(torch.tensor(psi_split[i+len(d_dims)+len(g_dims)][:,k].reshape(*d_shapes[i]).astype('float32'), requires_grad=False)) for i in range(len(d_dims))] for k in Ks]
            self.psi_g_a = [[self.m.to_gpu_alt(torch.tensor(psi_split[i+2*len(d_dims)+len(g_dims)][:,k].reshape(*g_shapes[i]).astype('float32'), requires_grad=False)) for i in range(len(g_dims))] for k in Ks]

        self.lams = np.zeros(self.K)

        steps = [self.m.params['disc_learning_rate'], self.m.params['gen_learning_rate']] * (1 + self.req_aux)
        self.delta_ts = [step*self.m.params['LE_freq'] for step in steps]

        self.real_data_fixed = self.m.get_real(self.m.params['batch_size'])
        self.fake_z_fixed = self.m.get_z(self.m.params['batch_size'], self.m.params['z_dim'])

    def train_op(self, it):
        self.m.D.zero_grad()
        self.m.G.zero_grad()

        # 1. Record x_k
        Dp = self.m.D.get_param_data()
        Gp = self.m.G.get_param_data()

        if self.req_aux:
            # 2. Record aux
            daux = [p.data.detach() for p in self.aux_d]
            gaux = [p.data.detach() for p in self.aux_g]

        # 3. Get real data and samples from p(z) to pass to generator
        if it == self.m.params['start_lam_it']:
            # Increase batch size for reduced stochasticity, i.e., ~deterministic
            self.m.params['batch_size'] *= self.m.params['LE_batch_mult']
            if self.m.params['deterministic']:
                self.real_data_fixed = self.m.get_real(self.m.params['batch_size'])
                self.fake_z_fixed = self.m.get_z(self.m.params['batch_size'], self.m.params['z_dim'])
        if self.m.params['deterministic']:
            real_data = self.real_data_fixed
            fake_z = self.fake_z_fixed
        else:
            real_data = self.m.get_real(self.m.params['batch_size'])
            fake_z = self.m.get_z(self.m.params['batch_size'], self.m.params['z_dim'])

        # 4. Evaluate Map
        freeze_d = (it in range(*self.m.params['freeze_d_its']))
        freeze_g = (it in range(*self.m.params['freeze_g_its']))
        _, _, map_d, map_g, map_aux_d, map_aux_g, V, norm_d, norm_g = detach_all(self.cmap([real_data, self.m.G(fake_z), freeze_d, freeze_g, self.aux_d, self.aux_g]))

        proj_weights = np.zeros(self.K)

        # 5. Compute Lyapunov exponents after initial "burn-in"
        if it >= self.m.params['start_lam_it']:
            # 5a. Project weights
            psi_norms_old = np.zeros(self.K)
            for k in range(self.K):
                for i in range(len(self.psi_d[k])):
                    proj_weights[k] += torch.sum(self.psi_d[k][i]*Dp[i]).item()
                    psi_norms_old[k] += torch.sum(self.psi_d[k][i]**2).item()
                for i in range(len(self.psi_g[k])):
                    proj_weights[k] += torch.sum(self.psi_g[k][i]*Gp[i]).item()
                    psi_norms_old[k] += torch.sum(self.psi_g[k][i]**2).item()
            if self.req_aux:
                for k in range(self.K):
                    for i in range(len(self.psi_d_a[k])):
                        proj_weights[k] += torch.sum(self.psi_d_a[k][i]*Dp[i]).item()
                        psi_norms_old[k] += torch.sum(self.psi_d_a[k][i]**2).item()
                    for i in range(len(self.psi_g_a[k])):
                        proj_weights[k] += torch.sum(self.psi_g_a[k][i]*Gp[i]).item()
                        psi_norms_old[k] += torch.sum(self.psi_g_a[k][i]**2).item()
            proj_weights /= psi_norms_old
            if np.any(np.abs(psi_norms_old-1.)>=0.01):
                print('Lyapunov Vectors (Psi) have drifted away from unit-norm!')
                embed()

            # 5b-f. Loop over psis, perturb, and update: psi = [F(x_k + self.epsilon*psi) - F(x_k)]/self.epsilon
            for k in range(self.K):
                # 5b. Perturb parameters
                for i,p in enumerate(self.m.D.parameters()):
                    p.data = (p.data + self.epsilon*self.psi_d[k][i]).detach()
                for i,p in enumerate(self.m.G.parameters()):
                    p.data = (p.data + self.epsilon*self.psi_g[k][i]).detach()
                if self.req_aux:
                    for i,a in enumerate(self.aux_d):
                        a.data = (a.data + self.epsilon*self.psi_d_a[k][i]).detach()
                    for i,a in enumerate(self.aux_g):
                        a.data = (a.data + self.epsilon*self.psi_g_a[k][i]).detach()

                # 5c. Evaluate map
                _, _, map_psi_d, map_psi_g, map_psi_aux_d, map_psi_aux_g, _, _, _ = detach_all(self.cmap([real_data, self.m.G(fake_z), freeze_d, freeze_g, self.aux_d, self.aux_g]))
                
                # 5d. Update psi[k]
                for i,psi in enumerate(self.psi_d[k]):
                    psi.sub_(self.m.params['disc_learning_rate']*(map_psi_d[i]-map_d[i])/self.epsilon)
                for i,psi in enumerate(self.psi_g[k]):
                    psi.sub_(self.m.params['gen_learning_rate']*(map_psi_g[i]-map_g[i])/self.epsilon)
                if self.req_aux:
                    for i,psi in enumerate(self.psi_d_a[k]):
                        psi.sub_(self.m.params['disc_learning_rate']*(map_psi_aux_d[i]-map_aux_d[i])/self.epsilon)
                    for i,psi in enumerate(self.psi_g_a[k]):
                        psi.sub_(self.m.params['gen_learning_rate']*(map_psi_aux_g[i]-map_aux_g[i])/self.epsilon)
                
                # 5e. Reset weights to x_k
                self.m.D.set_param_data(Dp)
                self.m.G.set_param_data(Gp)
                
                # 5f. Reset auxiliary vars to aux_k
                if self.req_aux:
                    for i,a in enumerate(self.aux_d):
                        a.data = daux[i].detach()
                    for i,a in enumerate(self.aux_g):
                        a.data = gaux[i].detach()

            # 5g-j. Orthogonalize psis, compute norms, normalize psis, and update Lyapunov exponents
            if (it+1 - self.m.params['start_lam_it']) % self.m.params['LE_freq'] == 0:

                # 5g. Reshape psis into matrix psis_sh (K x total num weights)
                psis = [self.psi_d, self.psi_g]
                if self.req_aux:
                    psis += [self.psi_d_a, self.psi_g_a]
                psis_sh = [functools.reduce(lambda x,y: x+y, ps) for ps in zip(*psis)]  # psis is 4 parameter dims x K x num weights

                # 5h. Gram Schmidt
                psi_orth, psi_norms = self.GramSchmidt(psis_sh)
                psis_temp = list(zip(*psi_orth))
                self.psi_d = [list(el) for el in zip(*psis_temp[:self.num_d])]
                self.psi_g = [list(el) for el in zip(*psis_temp[self.num_d:self.num_d+self.num_g])]
                if self.req_aux:
                    self.psi_d_a = [list(el) for el in zip(*psis_temp[self.num_d+self.num_g:2*self.num_d+self.num_g])]
                    self.psi_g_a = [list(el) for el in zip(*psis_temp[2*self.num_d+self.num_g:])]

                # 5i. Normalize psis
                for k in range(self.K):
                    for i in range(len(self.psi_d[k])):
                        self.psi_d[k][i] /= psi_norms[k]
                        self.psi_d[k][i].detach()
                    for i in range(len(self.psi_g[k])):
                        self.psi_g[k][i] /= psi_norms[k]
                        self.psi_g[k][i].detach()
                if self.req_aux:
                    for k in range(self.K):
                        for i in range(len(self.psi_d_a[k])):
                            self.psi_d_a[k][i] /= psi_norms[k]
                            self.psi_d_a[k][i].detach()
                        for i in range(len(self.psi_g_a[k])):
                            self.psi_g_a[k][i] /= psi_norms[k]
                            self.psi_g_a[k][i].detach()

                # 5j. Update Lyapunov exponents (lambdas)
                new_lam_dts = [np.log(psi_norm.item()) for psi_norm in psi_norms]  # actually equal to lambda*dt
                Ts = [dt*(it - self.m.params['start_lam_it']) for dt in self.delta_ts]
                self.lams = np.array([(lam*T+new_lam_dt)/(T+dt) for lam, new_lam_dt, T, dt in zip(self.lams, new_lam_dts, Ts, self.delta_ts)])
                if np.any(np.isnan(self.lams)):
                    print('Lyapunov Exponent is NaN!')
                    embed()

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

        return self.lams, norm_d.item(), norm_g.item(), V.item(), proj_weights

    @staticmethod
    def GramSchmidt(A):
        # only orthogonalizes, no normalization
        # assumes A is list of vectors
        uu = []
        if len(A) > 1:
            for i in range(len(A)):
                vi = A[i]
                proj = [0*vip for vip in vi]
                for j in range(i):
                    uj = A[j]
                    viuj = sum([torch.sum(vip*ujp) for vip, ujp in zip(vi,uj)])
                    factor = viuj/uu[j]
                    for p in range(len(proj)):
                        proj[p] += factor*uj[p]
                for p in range(len(A[i])):
                    A[i][p] -= proj[p]
                uu += [sum([torch.sum(uip**2) for uip in A[i]])]
        return A, [torch.sqrt(uuk) for uuk in uu]

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
