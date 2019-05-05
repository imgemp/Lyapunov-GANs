import argparse
import pickle
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mpath

from sklearn.decomposition import IncrementalPCA
from sklearn.metrics.pairwise import pairwise_distances

import sys
sys.path.append('../')

from lyapunov.utils import flatten_nested, colorline, intersection, shift_range


from IPython import embed


def parse_params():
    parser = argparse.ArgumentParser(description='GANs in PyTorch')
    parser.add_argument('-saveto','--saveto', type=str, default='', help='path prefix for saving results', required=False)
    parser.add_argument('-max_iter','--max_iter', type=int, default=-1, help='overwrite max_iter', required=False)
    parser.add_argument('-replt_rt','--replt_rt', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to replot the lyapunov exponents and pca trajectories (runtime plots)', required=False)
    args = vars(parser.parse_args())
    return args['saveto'], args['max_iter'], args['replt_rt']


def load_args(filepath, max_iter, replt_rt=False):
    params = {}
    with open(filepath+'args.txt', 'r') as f:
        keys, vals = [], []
        for line in f:
            elements = line.strip('\n').split(' ')
            key, val = elements[0], elements[1:]
            key = key.strip('--')
            keys += [key]
            vals += [val]
        temp = dict(zip(keys,vals))
    params['saveto'] = filepath
    params['start_lam_it'] = int(temp['start_lam_it'][0])
    params['freeze_d_its'] = [int(it.strip(',').strip('[').strip(']')) for it in temp['freeze_d_its']]
    params['freeze_g_its'] = [int(it.strip(',').strip('[').strip(']')) for it in temp['freeze_g_its']]
    if max_iter > -1:
        params['max_iter'] = max_iter
    else:
        params['max_iter'] = int(temp['max_iter'][0])
    params['weights_every'] = int(temp['weights_every'][0])
    params['n_viz'] = int(temp['n_viz'][0])
    params['viz_every'] = int(temp['viz_every'][0])
    params['x_dim'] = int(temp['x_dim'][0])
    params['domain'] = temp['domain'][0]
    params['batch_size'] = int(temp['batch_size'][0])

    if params['domain'] == 'MO8G':
        from examples.domains.synthetic import MOG_Circle as Domain
    elif params['domain'] == 'MO25G':
        from examples.domains.synthetic import MOG_Grid as Domain
    elif params['domain'] == 'SwissRoll':
        from examples.domains.synthetic import SwissRoll as Domain
    elif 'Gaussian' in params['domain']:
        from examples.domains.synthetic import Gaussian as Domain
    elif params['domain'] == 'MNIST':
        from examples.domains.mnist import MNIST as Domain
    elif params['domain'] == 'MNIST2':
        from examples.domains.mnist2 import MNIST as Domain
    elif params['domain'] == 'CIFAR10':
        from examples.domains.cifar10 import CIFAR10 as Domain
    else:
        raise NotImplementedError(params['domain'])

    data = Domain(dim=params['x_dim'], batch_size=params['batch_size'])

    return data, params, replt_rt

def plot_les(les, params, fontsize=18):
    fig = plt.figure()
    plt.plot(np.vstack(les))
    mn, mx = np.min(les), np.max(les)
    if params['freeze_d_its'][0] != params['freeze_d_its'][1]:
        plt.plot([params['freeze_d_its'][0]-params['start_lam_it']]*2,[mn,mx], '--', color='dodgerblue')
        plt.plot([params['freeze_d_its'][1]-params['start_lam_it']]*2,[mn,mx], '--', color='dodgerblue')
    if params['freeze_g_its'][0] != params['freeze_g_its'][1]:
        plt.plot([params['freeze_g_its'][0]-params['start_lam_it']]*2,[mn,mx], '--', color='r')
        plt.plot([params['freeze_g_its'][1]-params['start_lam_it']]*2,[mn,mx], '--', color='r')
    le_strs = ['{num:.3{c}}'.format(num=le, c='e' if (abs(le) < 1e-3 or abs(le) > 1e3) else 'f') for le in [np.min(les[-1]),np.max(les[-1])]]
    plt.title('LE range = ({},{})'.format(*le_strs), fontsize=fontsize)
    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel(r'Top-2 Lyapunov Exponents ($\Lambda_{1,2}$)', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    locs, _ = plt.xticks()
    if locs[-1] >= 10000:
        newlocs = [loc for loc in locs if loc >= 0 and loc < len(les)]
        xlabels = [str(int(loc)//1000)+'k' for loc in newlocs]
        plt.xticks(newlocs, xlabels)
    plt.tight_layout()
    fig.savefig(params['saveto']+'lyapunov_exponents.pdf')
    plt.close(fig)

def plot_traj(pws, params, fontsize=18):
    fig, ax = plt.subplots(1)
    z = np.linspace(0, 1, len(pws))
    colorline(*np.split(np.vstack(pws),2,axis=1), z, cmap=plt.get_cmap('Greys'), linewidth=0.2)
    plt.title(r'Weights Trajectory Projected onto $\psi$', fontsize=fontsize)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.tight_layout()
    fig.savefig(params['saveto']+'projected_traj.pdf')
    plt.close(fig)

def post_eval(data, params, replot_runtimeplots=False):
    fontsize = 18

    if replot_runtimeplots:
        print('Plotting runtime plots...')
        try:
            les = np.loadtxt(params['saveto']+'les.out')
            plot_les(les, params, fontsize)
        except:
            print('LE plotting failed.')

        try:
            pws = np.loadtxt(params['saveto']+'pws.out')
            plot_traj(pws, params, fontsize)
        except:
            print('Projected trajectory plotting failed.')

    d_rng = intersection(params['freeze_d_its'], (params['start_lam_it'],params['max_iter']-1))
    g_rng = intersection(params['freeze_g_its'], (params['start_lam_it'],params['max_iter']-1))
    if d_rng is not None and g_rng is not None:
        both_rng = intersection(list(d_rng), list(g_rng))
    else:
        both_rng = None

    print('Plotting gradient norms...')
    try:
        ds = np.loadtxt(params['saveto']+'d_norm.out')
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.plot(range(len(ds)), ds, 'k-')
        if d_rng is not None: plt.plot(d_rng, ds[d_rng], '-', color='dodgerblue')
        if g_rng is not None: plt.plot(g_rng, ds[g_rng], 'r-')
        if both_rng is not None: plt.plot(both_rng, ds[both_rng], '-', color='lime')
        ax.set_ylabel('Discriminator Gradient L2 Norm', fontsize=fontsize)
        ax.set_xlabel('Iteration', fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.title('Final Norm: {:.3e}'.format(ds[-1]), fontsize=fontsize)
        locs, _ = plt.xticks()
        if locs[-1] >= 10000:
            newlocs = [loc for loc in locs if loc >= 0 and loc < len(ds)]
            xlabels = [str(int(loc)//1000)+'k' for loc in newlocs]
            plt.xticks(newlocs, xlabels)
        plt.tight_layout()
        fig.savefig(params['saveto']+'d_norm.pdf')
    except:
        print('d_norm.out not found.')

    try:
        gs = np.loadtxt(params['saveto']+'g_norm.out')
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.plot(range(len(gs)), gs, 'k-')
        if d_rng is not None: plt.plot(d_rng, gs[d_rng], '-', color='dodgerblue')
        if g_rng is not None: plt.plot(g_rng, gs[g_rng], 'r-')
        if both_rng is not None: plt.plot(both_rng, gs[both_rng], '-', color='lime')
        ax.set_ylabel('Generator Gradient L2 Norm', fontsize=fontsize)
        ax.set_xlabel('Iteration', fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.title('Final Norm: {:.3e}'.format(gs[-1]), fontsize=fontsize)
        locs, _ = plt.xticks()
        if locs[-1] >= 10000:
            newlocs = [loc for loc in locs if loc >= 0 and loc < len(gs)]
            xlabels = [str(int(loc)//1000)+'k' for loc in newlocs]
            plt.xticks(newlocs, xlabels)
        plt.tight_layout()
        fig.savefig(params['saveto']+'g_norm.pdf')
    except:
        print('g_norm.out not found.')

    print('Plotting loss...')
    try:
        fs = np.loadtxt(params['saveto']+'loss.out')
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.plot(range(len(fs)),np.array(fs), 'k-')
        if d_rng is not None: plt.plot(d_rng, fs[d_rng], '-', color='dodgerblue')
        if g_rng is not None: plt.plot(g_rng, fs[g_rng], 'r-')
        if both_rng is not None: plt.plot(both_rng, fs[both_rng], '-', color='lime')
        ax.set_ylabel('Minimax Loss', fontsize=fontsize)
        ax.set_xlabel('Iteration', fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.title('Final Loss: {:.3e}'.format(fs[-1]), fontsize=fontsize)
        locs, _ = plt.xticks()
        if locs[-1] >= 10000:
            newlocs = [loc for loc in locs if loc >= 0 and loc < len(fs)]
            xlabels = [str(int(loc)//1000)+'k' for loc in newlocs]
            plt.xticks(newlocs, xlabels)
        plt.tight_layout()
        fig.savefig(params['saveto']+'loss.pdf')
    except:
        print('loss.out not found.')

    print('Loading weights from saved files...')
    weights = []
    for w_i in range(params['start_lam_it'],params['max_iter'],params['weights_every']):
        w_D = flatten_nested(pickle.load(open(params['saveto']+'weights/D_'+str(w_i)+'.pkl','rb')))
        w_G = flatten_nested(pickle.load(open(params['saveto']+'weights/G_'+str(w_i)+'.pkl','rb')))
        weights.append(np.hstack([w_D,w_G]))
    weights = np.vstack(weights)

    d_rng = shift_range(d_rng, shift=-params['start_lam_it'], keep_every=params['weights_every'])
    g_rng = shift_range(g_rng, shift=-params['start_lam_it'], keep_every=params['weights_every'])
    both_rng = shift_range(both_rng, shift=-params['start_lam_it'], keep_every=params['weights_every'])

    print('Plotting PCA of trajectory...')
    ipca = IncrementalPCA(n_components=2, batch_size=10)
    X_ipca = ipca.fit_transform(weights)
    fig, ax = plt.subplots()
    path = mpath.Path(X_ipca)
    verts = path.interpolated(steps=1).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, 1, len(x))
    colorline(x, y, z, cmap=plt.get_cmap('Greys'), linewidth=1.0)
    if d_rng is not None: plt.plot(X_ipca[d_rng,0], X_ipca[d_rng, 1], '-', color='dodgerblue', lw=0.5)
    if g_rng is not None: plt.plot(X_ipca[g_rng,0], X_ipca[g_rng, 1], 'r-', lw=0.5)
    if both_rng is not None: plt.plot(X_ipca[both_rng,0], X_ipca[both_rng, 1], '-', color='lime', lw=0.5)
    ax.set_xlim([X_ipca[:,0].min(), X_ipca[:,0].max()])
    ax.set_ylim([X_ipca[:,1].min(), X_ipca[:,1].max()])
    plt.title('Weights Trajectory Projected onto Top-2 PCs\n'+r'($p2p_x,p2p_y$)'+' = ({:.3f},{:.3f})'.format(np.ptp(x),np.ptp(y)), fontsize=fontsize)
    plt.tick_params(axis='both', which='major', bottom=False, top=False, left=False, right=False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    fig.tight_layout()
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
    colorline(x2, y2, z2, cmap=plt.get_cmap('Greys'), linewidth=1.0)
    if d_rng is not None: plt.plot(X_ipca2[d_rng,0], X_ipca2[d_rng, 1], '-', color='dodgerblue', lw=0.5)
    if g_rng is not None: plt.plot(X_ipca2[g_rng,0], X_ipca2[g_rng, 1], 'r-', lw=0.5)
    if both_rng is not None: plt.plot(X_ipca2[both_rng,0], X_ipca2[both_rng, 1], '-', color='lime', lw=0.5)
    plt.title('Normalized Weights Trajectory\nProjected onto Top-2 PCs\n'+r'($p2p_x,p2p_y$)'+' = ({:.3f},{:.3f})'.format(np.ptp(x2),np.ptp(y2)), fontsize=fontsize)
    plt.tick_params(axis='both', which='major', bottom=False, top=False, left=False, right=False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    fig.tight_layout()
    fig.savefig(params['saveto']+'weights_pca2.pdf')
    plt.close(fig)

    print('Plotting norm of weights over trajectory...')
    w_norms = np.linalg.norm(weights, axis=1)
    fig = plt.figure()
    plt.plot(range(len(w_norms)), w_norms, 'k-')
    if d_rng is not None: plt.plot(d_rng, w_norms[d_rng], '-', color='dodgerblue')
    if g_rng is not None: plt.plot(g_rng, w_norms[g_rng], 'r-')
    if both_rng is not None: plt.plot(both_rng, w_norms[both_rng], '-', color='lime')
    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel(r'Norm of Weights ($||w||$)', fontsize=fontsize)
    plt.title('Norm of Weights Over Trajectory\n'+r'($p2p$'+'={:.3f})'.format(np.ptp(w_norms)), fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    locs, _ = plt.xticks()
    if locs[-1] >= 10000:
        newlocs = [loc for loc in locs if loc >= 0 and loc < len(w_norms)]
        xlabels = [str(int(loc)//1000)+'k' for loc in newlocs]
        plt.xticks(newlocs, xlabels)
    plt.tight_layout()
    fig.savefig(params['saveto']+'weight_norms.pdf')
    plt.close(fig)

    print('Plotting distance of weights from mean over trajectory...')
    weights_mean = weights.mean(axis=0)
    w_mean_norms = np.linalg.norm(weights-weights_mean, axis=1)
    fig = plt.figure()
    plt.plot(range(len(w_mean_norms)), w_mean_norms, 'k-')
    if d_rng is not None: plt.plot(d_rng, w_mean_norms[d_rng], '-', color='dodgerblue')
    if g_rng is not None: plt.plot(g_rng, w_mean_norms[g_rng], 'r-')
    if both_rng is not None: plt.plot(both_rng, w_mean_norms[both_rng], '-', color='lime')
    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel(r'Norm of Weights ($||w||$)', fontsize=fontsize)
    plt.title('Norm of Weights Over Trajectory\n'+r'($p2p$'+'={:.3f})'.format(np.ptp(w_mean_norms)), fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    locs, _ = plt.xticks()
    if locs[-1] >= 10000:
        newlocs = [loc for loc in locs if loc >= 0 and loc < len(w_mean_norms)]
        xlabels = [str(int(loc)//1000)+'k' for loc in newlocs]
        plt.xticks(newlocs, xlabels)
    plt.tight_layout()
    fig.savefig(params['saveto']+'weight_mean_norms.pdf')
    plt.close(fig)

    print('Plotting angular distance of weights from mean over trajectory...')
    w_mean_angles = 180/np.pi*np.arccos(np.sum(weights*weights_mean, axis=1)/w_norms/np.linalg.norm(weights_mean))
    fig = plt.figure()
    plt.plot(range(len(w_mean_angles)), w_mean_angles, 'k-')
    if d_rng is not None: plt.plot(d_rng, w_mean_angles[d_rng], '-', color='dodgerblue')
    if g_rng is not None: plt.plot(g_rng, w_mean_angles[g_rng], 'r-')
    if both_rng is not None: plt.plot(both_rng, w_mean_angles[both_rng], '-', color='lime')
    plt.title('Angular Deviation of Weights\nfrom Mean Over Trajectory\n'+r'($p2p$'+'={:.3f})'.format(np.ptp(w_mean_angles)), fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.ylabel('Angles in degrees', fontsize=fontsize)
    plt.xlabel('Iteration', fontsize=fontsize)
    locs, _ = plt.xticks()
    if locs[-1] >= 10000:
        newlocs = [loc for loc in locs if loc >= 0 and loc < len(w_mean_angles)]
        xlabels = [str(int(loc)//1000)+'k' for loc in newlocs]
        plt.xticks(newlocs, xlabels)
    plt.tight_layout()
    fig.savefig(params['saveto']+'weight_mean_angles.pdf')
    plt.close(fig)

    print('Plotting distance of weights from closest vector over trajectory...')
    D = pairwise_distances(weights)
    closest = weights[D.sum(axis=1).argmin()]
    w_closest_norms = np.linalg.norm(weights-closest, axis=1)
    fig = plt.figure()
    plt.plot(range(len(w_closest_norms)), w_closest_norms, 'k-')
    if d_rng is not None: plt.plot(d_rng, w_closest_norms[d_rng], '-', color='dodgerblue')
    if g_rng is not None: plt.plot(g_rng, w_closest_norms[g_rng], 'r-')
    if both_rng is not None: plt.plot(both_rng, w_closest_norms[both_rng], '-', color='lime')
    plt.title('Norm of Weights Over Trajectory\n'+r'($p2p$'+'={:.3f})'.format(np.ptp(w_closest_norms)), fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel(r'Norm of Weights ($||w||$)', fontsize=fontsize)
    locs, _ = plt.xticks()
    if locs[-1] >= 10000:
        newlocs = [loc for loc in locs if loc >= 0 and loc < len(w_closest_norms)]
        xlabels = [str(int(loc)//1000)+'k' for loc in newlocs]
        plt.xticks(newlocs, xlabels)
    plt.tight_layout()
    fig.savefig(params['saveto']+'weight_closest_norms.pdf')
    plt.close(fig)

    print('Plotting sample series over epochs...')
    if params['n_viz'] > 0:
        np_samples = []
        for viz_i in range(0,params['max_iter'],params['viz_every']):
            np_samples.append(np.load(params['saveto']+'samples/'+str(viz_i)+'.npy'))
        data.plot_series(np_samples, params)

    print('Complete.')


if __name__ == '__main__':
    saveto, max_iter, replt_rt = parse_params()
    post_eval(*load_args(saveto, max_iter, replt_rt))
