import os
import psutil
import shutil
import argparse
import datetime
import resource
import pickle
import numpy as np
import torch
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import seaborn as sns

from sklearn.decomposition import IncrementalPCA
from sklearn.metrics.pairwise import pairwise_distances

import sys
sys.path.append('../')

from lyapunov.core import Manager
from lyapunov.utils import gpu_helper, save_weights, flatten_nested, colorline, intersection

from tqdm import tqdm

from IPython import embed


process = psutil.Process(os.getpid())

def parse_params():
    parser = argparse.ArgumentParser(description='GANs in PyTorch')
    parser.add_argument('-dom','--domain', type=str, default='MO8G', help='domain to run', required=False)
    parser.add_argument('-desc','--description', type=str, default='', help='description for the experiment', required=False)
    parser.add_argument('-bs','--batch_size', type=int, default=512, help='batch_size for training', required=False)
    parser.add_argument('-div','--divergence', type=str, default='JS', help='divergence measure, i.e. V, for training', required=False)
    parser.add_argument('-d_lr','--disc_learning_rate', type=float, default=1e-4, help='discriminator learning rate', required=False)
    parser.add_argument('-d_l2','--disc_weight_decay', type=float, default=0., help='discriminator weight decay', required=False)
    parser.add_argument('-d_nh','--disc_n_hidden', type=int, default=128, help='# of hidden units for discriminator', required=False)
    parser.add_argument('-d_nl','--disc_n_layer', type=int, default=1, help='# of hidden layers for discriminator', required=False)
    parser.add_argument('-d_nonlin','--disc_nonlinearity', type=str, default='relu', help='type of nonlinearity for discriminator', required=False)
    parser.add_argument('-d_quad','--disc_quadratic_layer', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to use a quadratic final layer', required=False)
    parser.add_argument('-g_lr','--gen_learning_rate', type=float, default=1e-4, help='generator learning rate', required=False)
    parser.add_argument('-g_l2','--gen_weight_decay', type=float, default=0., help='generator weight decay', required=False)
    parser.add_argument('-g_nh','--gen_n_hidden', type=int, default=128, help='# of hidden units for generator', required=False)
    parser.add_argument('-g_nl','--gen_n_layer', type=int, default=2, help='# of hidden layers for generator', required=False)
    parser.add_argument('-g_nonlin','--gen_nonlinearity', type=str, default='relu', help='type of nonlinearity for generator', required=False)
    parser.add_argument('-betas','--betas', type=float, nargs=2, default=(0.5,0.999), help='beta params for Adam', required=False)
    parser.add_argument('-eps','--epsilon', type=float, default=1e-8, help='epsilon param for Adam', required=False)
    parser.add_argument('-mx_it','--max_iter', type=int, default=100001, help='max # of training iterations', required=False)
    parser.add_argument('-viz_every','--viz_every', type=int, default=1000, help='skip viz_every iterations between plotting current results', required=False)
    parser.add_argument('-series_every','--series_every', type=int, default=25000, help='skip series_every iterations between plotting series plot', required=False)
    parser.add_argument('-w_every','--weights_every', type=int, default=25000, help='skip weights_every iterations between saving weights', required=False)
    parser.add_argument('-n_viz','--n_viz', type=int, default=5120, help='number of samples for series plot', required=False)
    parser.add_argument('-zdim','--z_dim', type=int, default=256, help='dimensionality of p(z) - unit normal', required=False)
    parser.add_argument('-xdim','--x_dim', type=int, default=2, help='dimensionality of p(x) - data distribution', required=False)
    parser.add_argument('-maps','--map_strings', type=str, nargs='+', default=[''], help='string names of optimizers to use for generator and discriminator', required=False)
    parser.add_argument('-gam','--gamma', type=float, default=10., help='gamma parameter for consensus, reg, reg_alt, and cc', required=False)
    parser.add_argument('-gamT','--gammaT', type=float, default=-1e11, help='gamma parameter for JTF in cc algorithm', required=False)
    parser.add_argument('-kap','--kappa', type=float, default=0., help='kappa parameter for F in cc algorithm', required=False)
    parser.add_argument('-K','--K', type=int, default=2, help='number of lyapunov exponents to compute', required=False)
    parser.add_argument('-psi_epsilon','--psi_epsilon', type=float, default=0., help='epsilon to use for finite difference approximation of Jacobian vector product', required=False)
    parser.add_argument('-LE_freq','--LE_freq', type=int, default=5, help='number of steps to wait inbetween computing LEs', required=False)
    parser.add_argument('-LE_batch_mult','--LE_batch_mult', type=int, default=10, help='batch_size multiplier to reduce variance when computing LEs', required=False)
    parser.add_argument('-start_lam_it','--start_lam_it', type=int, default=-1, help='number of steps to wait inbetween computing LEs', required=False)
    parser.add_argument('-freeze_d_its','--freeze_d_its', type=int, nargs=2, default=[-1,-1], help='iteration range for which to freeze the discriminator', required=False)
    parser.add_argument('-freeze_g_its','--freeze_g_its', type=int, nargs=2, default=[-1,-1], help='iteration range for which to freeze the generator', required=False)
    parser.add_argument('-det','--deterministic', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to compute loss always using same samples', required=False)
    parser.add_argument('-saveto','--saveto', type=str, default='', help='path prefix for saving results', required=False)
    parser.add_argument('-gpu','--gpu', type=int, default=-2, help='if/which gpu to use (-1: all, -2: None)', required=False)
    parser.add_argument('-verb','--verbose', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to print progress to stdout', required=False)
    args = vars(parser.parse_args())

    if args['psi_epsilon'] <= 0.:
        args['psi_epsilon'] = 0.1*min(args['disc_learning_rate'],args['gen_learning_rate'])
    if args['start_lam_it'] < 0.:
        args['start_lam_it'] = int(0.9*args['max_iter'])

    if args['domain'] == 'MO8G':
        from examples.domains.synthetic import MOG_Circle as Domain
        from examples.domains.synthetic import Generator, Discriminator
    elif args['domain'] == 'MO25G':
        from examples.domains.synthetic import MOG_Grid as Domain
        from examples.domains.synthetic import Generator, Discriminator
    elif args['domain'] == 'SwissRoll':
        from examples.domains.synthetic import SwissRoll as Domain
        from examples.domains.synthetic import Generator, Discriminator
    elif 'Gaussian' in args['domain']:
        from examples.domains.synthetic import Gaussian as Domain
        if args['domain'][:2] == 'CL':
            from examples.domains.synthetic import Generator_C as Generator
            from examples.domains.synthetic import Discriminator_L as Discriminator
        elif args['domain'][:2] == 'LQ':
            from examples.domains.synthetic import Generator_L as Generator
            from examples.domains.synthetic import Discriminator_Q as Discriminator
    elif args['domain'] == 'MNIST':
        from examples.domains.mnist import MNIST as Domain
        from examples.domains.mnist import Generator, Discriminator
    elif args['domain'] == 'MNIST2':
        from examples.domains.mnist2 import MNIST as Domain
        from examples.domains.mnist2 import Generator, Discriminator
    elif args['domain'] == 'CIFAR10':
        from examples.domains.cifar10 import CIFAR10 as Domain
        from examples.domains.cifar10 import Generator, Discriminator
    else:
        raise NotImplementedError(args['domain'])

    from lyapunov.core import Train
    args['maps'] = []
    for mp in args['map_strings']:
        if mp.lower() == 'consensus':
            from lyapunov.train_ops.consensus import Consensus
            args['maps'] += [Consensus]
        elif mp.lower() == 'rmsprop':
            from lyapunov.train_ops.rmsprop import RMSProp
            args['maps'] += [RMSProp]
        else:
            raise NotImplementedError(mp)
    from lyapunov.train_ops.simgd import SimGD
    args['maps'] += [SimGD]

    if args['saveto'] == '':
        args['saveto'] = 'examples/results/' + args['domain'] + '/' + '-'.join(args['map_strings']) + '/' + args['description']

    if args['description'] == '':
        args['description'] = args['domain'] + '-' + '-'.join(args['map_strings'])
    elif args['description'].isdigit():
        args['description'] = args['domain'] + '-' + '-'.join(args['map_strings']) + '-' + args['description']

    saveto = args['saveto'] + '/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S/{}').format('')
    if not os.path.exists(saveto):
        os.makedirs(saveto)
        os.makedirs(saveto+'/samples')
        os.makedirs(saveto+'/weights')
    shutil.copy(os.path.realpath('lyapunov/run.py'), os.path.join(saveto, 'run.py'))
    shutil.copy(os.path.realpath('lyapunov/core.py'), os.path.join(saveto, 'core.py'))
    for mp in args['map_strings']:
        train_file = mp+'.py'
        shutil.copy(os.path.realpath('lyapunov/train_ops/'+train_file), os.path.join(saveto, train_file))
    with open(saveto+'args.txt', 'w') as file:
        for key, val in args.items():
            file.write('--'+str(key)+' '+str(val)+'\n')
    args['saveto'] = saveto

    cuda_available = torch.cuda.is_available()
    if args['gpu'] >= -1 and cuda_available:
        torch.cuda.device(args['gpu'])
        args['description'] += ' (gpu'+str(torch.cuda.current_device())+')'
    else:
        args['description'] += ' (cpu)'

    # python lyapunov/run.py $(cat examples/args/MO8G/con/00.txt) -dom CLGaussian -xdim 1 -zdim 1 -mx_it 10000 -d_lr 1e-6 -g_lr 1e-6
    
    return Train, Domain, Generator, Discriminator, args


def run_experiment(Train, Domain, Generator, Discriminator, params):
    to_gpu = gpu_helper(params['gpu'])

    data = Domain(dim=params['x_dim'])
    G = Generator(input_dim=params['z_dim'],output_dim=params['x_dim'],n_hidden=params['gen_n_hidden'],
                  n_layer=params['gen_n_layer'],nonlin=params['gen_nonlinearity'])
    D = Discriminator(input_dim=params['x_dim'],n_hidden=params['disc_n_hidden'],n_layer=params['disc_n_layer'],
                      nonlin=params['disc_nonlinearity'],quad=params['disc_quadratic_layer'])
    G.init_weights()
    D.init_weights()
    G = to_gpu(G)
    D = to_gpu(D)

    m = Manager(data, D, G, params, to_gpu)

    train = Train(manager=m)

    fs = []
    frames = []
    np_samples = []
    ds = [] # first gradients 
    gs = []
    les = []
    pws = []
    viz_every = params['viz_every']

    iterations = range(params['max_iter'])
    if params['verbose']:
        iterations = tqdm(iterations,desc=params['description'])

    for i in iterations:
        
        lams, d, g, f, pw = train.train_op(i)
        
        if params['verbose']:
            iterations.set_postfix({'Lambda':lams,'||F_D||^2':d,'||F_G||^2':g,'V':f, 'Mem': process.memory_info().rss})

        fs.append(f)
        ds.append(d)
        gs.append(g)

        if i >= params['start_lam_it']:
            les.append(lams)
            pws.append(pw)
            save_weights(m.D,params['saveto']+'weights/D_'+str(i)+'.pkl')
            save_weights(m.G,params['saveto']+'weights/G_'+str(i)+'.pkl')
            if train.req_aux:
                aux_d = []
                for a in train.aux_d:
                    aux_d += [a.cpu().data.numpy()]
                aux_g = []
                for a in train.aux_g:
                    aux_g += [a.cpu().data.numpy()]
                pickle.dump(aux_d,open(params['saveto']+'weights/D_aux_'+str(i)+'.pkl','wb'))
                pickle.dump(aux_g,open(params['saveto']+'weights/G_aux_'+str(i)+'.pkl','wb'))

        if viz_every > 0 and i % viz_every == 0:
            if params['n_viz'] > 0:
                np.save(params['saveto']+'samples/'+str(i), train.m.get_fake(params['n_viz'], params['z_dim']).cpu().data.numpy())
            data.plot_current(train, params, i)
            if i >= params['start_lam_it']:
                fig = plt.figure()
                plt.plot(np.vstack(les))
                fig.savefig(params['saveto']+'lyapunov_exponents.pdf') 
                plt.close(fig)
            if i >= params['start_lam_it']+1:
                fig = plt.figure()
                z = np.linspace(0, 1, len(pws))
                colorline(*np.split(np.vstack(pws),2,axis=1), z, cmap=plt.get_cmap('Greys'), linewidth=0.2)
                fig.savefig(params['saveto']+'projected_traj.pdf') 
                plt.close(fig)

        if params['weights_every'] > 0 and i % params['weights_every'] == 0:
            save_weights(m.D,params['saveto']+'D_'+str(i)+'.pkl')
            save_weights(m.G,params['saveto']+'G_'+str(i)+'.pkl')
                 

    np.savetxt(params['saveto']+'d_norm.out',np.array(ds))
    np.savetxt(params['saveto']+'g_norm.out',np.array(gs))
    np.savetxt(params['saveto']+'loss.out',np.array(fs))
    np.savetxt(params['saveto']+'les.out',np.vstack(les))
    np.savetxt(params['saveto']+'pws.out',np.vstack(pws))

    print('Plotting gradient norms...')
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(range(len(ds)), ds)
    ax.set_ylabel('Discriminator Gradient L2 Norm')
    ax.set_xlabel('Iteration')
    plt.title('final loss='+str(ds[-1]))
    fig.savefig(params['saveto']+'d_norm.pdf')

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(range(len(gs)), gs)
    ax.set_ylabel('Generator Gradient L2 Norm')
    ax.set_xlabel('Iteration')
    plt.title('final loss='+str(gs[-1]))
    fig.savefig(params['saveto']+'g_norm.pdf')

    print('Plotting loss...')
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(range(len(fs)),np.array(fs))
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    plt.title('final loss='+str(fs[-1]))
    fig.savefig(params['saveto']+'loss.pdf')

    print('Loading weights from saved files...')
    weights = []
    for w_i in range(params['start_lam_it'],params['max_iter']):
        w_D = flatten_nested(pickle.load(open(params['saveto']+'weights/D_'+str(w_i)+'.pkl','rb')))
        w_G = flatten_nested(pickle.load(open(params['saveto']+'weights/G_'+str(w_i)+'.pkl','rb')))
        weights.append(np.hstack([w_D,w_G]))
    weights = np.vstack(weights)

    print('Plotting PCA of trajectory...')
    ipca = IncrementalPCA(n_components=2, batch_size=10)
    X_ipca = ipca.fit_transform(weights)
    fig, ax = plt.subplots()
    path = mpath.Path(X_ipca)
    verts = path.interpolated(steps=1).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, 1, len(x))
    colorline(x, y, z, cmap=plt.get_cmap('Greys'), linewidth=0.2)
    d_rng = intersection(params['freeze_d_its'], (params['start_lam_it'],params['max_iter']-1), sub=params['start_lam_it'])
    if d_rng is not None:
        plt.plot(X_ipca[d_rng,0], X_ipca[d_rng, 1], 'b-', lw=0.2)
    g_rng = intersection(params['freeze_g_its'], (params['start_lam_it'],params['max_iter']-1), sub=params['start_lam_it'])
    if g_rng is not None:
        plt.plot(X_ipca[g_rng,0], X_ipca[g_rng, 1], 'r-', lw=0.2)
    if d_rng is not None and g_rng is not None:
        both_rng = intersection(list(d_rng), list(g_rng))
        plt.plot(X_ipca[both_rng,0], X_ipca[both_rng, 1], 'g-', lw=0.2)
    ax.set_xlim([X_ipca[:,0].min(), X_ipca[:,0].max()])
    ax.set_ylim([X_ipca[:,1].min(), X_ipca[:,1].max()])
    plt.title('p2px='+str(np.ptp(x))+', p2py='+str(np.ptp(y)))
    fig.savefig(params['saveto']+'weights_pca.pdf')
    plt.close(fig)

    print('Plotting PCA of normalized trajectory...')
    ipca2 = IncrementalPCA(n_components=2, batch_size=10)
    weights_normalized = (weights - weights.min(axis=0))/(np.ptp(weights, axis=0)+1e-10)
    X_ipca2 = ipca2.fit_transform(weights_normalized)
    fig, ax = plt.subplots()
    path2 = mpath.Path(X_ipca2)
    verts2 = path2.interpolated(steps=1).vertices
    x2, y2 = verts2[:, 0], verts2[:, 1]
    z2 = np.linspace(0, 1, len(x2))
    colorline(x2, y2, z2, cmap=plt.get_cmap('Greys'), linewidth=0.2)
    plt.title('p2px='+str(np.ptp(x2))+', p2py='+str(np.ptp(y2)))
    fig.savefig(params['saveto']+'weights_pca2.pdf')
    plt.close(fig)

    print('Plotting norm of weights over trajectory...')
    w_norms = np.linalg.norm(weights, axis=1)
    fig = plt.figure()
    plt.plot(w_norms)
    plt.title('p2p='+str(np.ptp(w_norms)))
    fig.savefig(params['saveto']+'weight_norms.pdf')
    plt.close(fig)

    print('Plotting distance of weights from mean over trajectory...')
    w_mean_norms = np.linalg.norm(weights-weights.mean(axis=0), axis=1)
    fig = plt.figure()
    plt.plot(w_mean_norms)
    plt.title('p2p='+str(np.ptp(w_mean_norms)))
    fig.savefig(params['saveto']+'weight_mean_norms.pdf')
    plt.close(fig)

    print('Plotting distance of weights from closest vector over trajectory...')
    D = pairwise_distances(weights)
    closest = weights[D.sum(axis=1).argmin()]
    w_closest_norms = np.linalg.norm(weights-closest, axis=1)
    fig = plt.figure()
    plt.plot(w_closest_norms)
    plt.title('p2p='+str(np.ptp(w_closest_norms)))
    fig.savefig(params['saveto']+'weight_closest_norms.pdf')
    plt.close(fig)

    print('Plotting sample series over epochs...')
    if params['n_viz'] > 0:
        np_samples = []
        for viz_i in range(0,params['max_iter'],viz_every):
            np_samples.append(np.load(params['saveto']+'samples/'+str(viz_i)+'.npy'))
        data.plot_series(np_samples, params)

    print('Complete.')


if __name__ == '__main__':
    Train, Domain, Generator, Discriminator, params = parse_params()
    run_experiment(Train, Domain, Generator, Discriminator, params)
