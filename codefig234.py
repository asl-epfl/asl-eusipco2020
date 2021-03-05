"""
This code can be used to generate simulations similar to Figs. 2, 3 and 4 in the following paper:
Virginia Bordignon, Vincenzo Matta, and Ali H. Sayed,  "Adaptation in online social learning,''  Proc. EUSIPCO, pp. 1-5, Amsterdam, The Netherlands, August 2020.

Please note that the code is not generally perfected for performance, but is rather meant to illustrate certain results from the paper. The code is provided as-is without guarantees.

July 2020 (Author: Virginia Bordignon)
"""
import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from functions import *
#%%
mpl.style.use('seaborn-deep')
mpl.rcParams['text.latex.preamble']= [r'\usepackage{amsmath}', r'\usepackage{amssymb}', r'\usepackage{bm}']
plt.rcParams.update({'text.usetex': True})
mpl.rcParams['font.family'] = 'serif'
#%%
FIG_PATH = 'figs/'
if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)
# %%
N=10
M=3
np.random.seed(0)
# %%
################################ Build Network Topology ################################
G = np.random.choice([0.0, 1.0], size=(N, N), p=[0.5, 0.5])
G = G + np.eye(N)
G = (G > 0) * 1.0

lamb = .5
A = np.zeros((N, N))
A = G / G.sum(axis = 0)
# %%
Gr = nx.from_numpy_array(A)
pos = nx.kamada_kawai_layout(Gr)
# %%
f, ax = plt.subplots(1, 1, figsize=(5, 2.5))
plt.axis('off')
plt.xlim([-1.1, 1.15])
plt.ylim([-.92, 1.12])
nx.draw_networkx_nodes(Gr, pos = pos, node_color = 'C1',nodelist = [0],node_size = 1000, edgecolors = 'k', linewidths = .5)
nx.draw_networkx_nodes(Gr, pos = pos, node_color = 'C5',nodelist = range(1, N), node_size = 1000, edgecolors = 'k', linewidths = .5)
nx.draw_networkx_labels(Gr, pos, {i: i+1 for i in range(N)}, font_size = 18, font_color = 'black', alpha = 1)
nx.draw_networkx_edges(Gr, pos = pos, node_size = 1000, alpha=1, arrowsize = 6, width = 1)
plt.tight_layout()
plt.savefig(FIG_PATH + 'fig2.pdf', bbox_inches = 'tight', pad_inches = 0)
#%%
################################ Run Social Learning ################################
theta = np.arange(1,4) * 0.5
b = 1
x = np.linspace(-10, 10, 1000)
dt = (max(x) - min(x)) / len(x)
N_ITER = 10000
# %%
L0 = laplace(x, theta[0], b)
L1 = laplace(x, theta[1], b)
L2 = laplace(x, theta[2], b)
L = np.array([L0, L1, L2])
# %%
np.random.seed(0)
mu_0 = np.random.rand(N, M)
mu_0 = mu_0 / np.sum(mu_0, axis = 1)[:, None]
delta = 0.1
csi = []
for l in range(0, N):
    csi.append(np.hstack([np.random.laplace(theta[0], b, size = 200),np.random.laplace(theta[2], b, size= N_ITER - 200)]))
csi = np.array(csi)

_, pv = np.linalg.eig(A)
pv = np.real(pv[:, 0] / sum(pv[:, 0]))
dklv = np.array([0] * 3+[DKL(L0, L2, dt)] * 3+[DKL(L0, L1, dt)] * 4)
ave1 = pv.dot(dklv)
dklv = np.array([DKL(L0, L2, dt)] * 3+[DKL(L0, L2, dt)] * 3+[0] * 4)
ave2 = pv.dot(dklv)
Ave = np.array([ave1, ave2])
cave = .5 * np.array([[(pv ** 2).dot(np.array([Cave(L, 0, 0, dt)] * 3 + [Cave(L, 2, 2, dt)] * 3 + [Cave(L, 1, 1, dt)] * 4)), (pv ** 2).dot(np.array([Cave(L, 0, 2, dt)] * 3 + [Cave(L, 2, 2, dt)] * 3 + [Cave(L, 1, 0, dt)]*4))],
[(pv ** 2).dot(np.array([Cave(L, 2, 0, dt)] * 3 + [Cave(L, 2, 2, dt)] * 3 + [Cave(L, 0, 1, dt)]*4)), (pv ** 2).dot(np.array([Cave(L, 2, 2, dt)] * 3 + [Cave(L, 2, 2, dt)] * 3 + [Cave(L, 0, 0, dt)] * 4))]])
#%%
N_MC = 100
N_ITER = 10000
delta = [1, 0.1, 0.01, 0.001]
MU_delta = []
for d in delta:
    MU_mc1 = []
    for i in range(N_MC):
        csi = []
        for l in range(0, N):
            csi.append(np.random.laplace(theta[0], b, size= N_ITER))
        csi=np.array(csi)
        MU_mc1.append(asl_l(mu_0, csi, A, N_ITER, theta, b, N, d, adaptive = True))
    MU_delta.append(MU_mc1)
# %%
data = np.array([[MU_delta[j][i][-1][0] for i in range(N_MC)] for j in range(len(delta))])
#%%
f, ax = plt.subplots(2, 2, figsize=(8, 6))
for di, d in enumerate(delta):
    bv1 = np.log(data[di][:,0] / data[di][:, 1])
    bv2 = np.log(data[di][:,0] / data[di][:, 2])
    Lbv = np.array([bv1, bv2])
    Cdelta = np.cov(Lbv)
    Avedelta = np.mean(Lbv, axis=1)

    confidence_ellipse(Lbv[0], Lbv[1], ax[di//2, di%2], n_std = 1, edgecolor = 'C0', linewidth = 2, linestyle = 'dashed')
    h2 = confidence_ellipse(Lbv[0], Lbv[1], ax[di//2, di%2], n_std = 2, edgecolor = 'C0', linewidth = 2, linestyle = 'dashed')
    gaussian_ellipse(d*cave, Ave, ax[di//2, di%2], n_std = 1, edgecolor = 'C2', linewidth = 2, linestyle = 'dotted')
    h3 = gaussian_ellipse(d*cave, Ave, ax[di//2, di%2], n_std = 2, edgecolor = 'C2', linewidth = 2, linestyle = 'dotted')

    h1 = ax[di//2, di%2].scatter(bv1, bv2, color = 'C5')
    ax[di//2, di%2].scatter(Avedelta[0], Avedelta[1], color = 'C0', marker = 'o', facecolors = 'None', s = 100, linewidth = 2)
    ax[di//2, di%2].scatter(ave1, ave2, color = 'C2', marker = '+', s = 100, linewidth = 2)
    ax[di//2, di%2].tick_params(axis = 'both', which = 'major', labelsize = 18)
    ax[di//2, di%2].set_xlabel(r'$\bm{\lambda}^\delta_{1}(\theta=2)$', fontsize = 18)
    ax[di//2, di%2].set_ylabel(r'$\bm{\lambda}^\delta_{1}(\theta=3)$', fontsize = 18)
    ax[di//2, di%2].set_title(r'$\delta={}$'.format(d), fontsize = 20)
    if di == 0:
        ax[di//2, di%2].set_xlim(Ave[0] - .6,Ave[0] + .6)
        ax[di//2, di%2].set_ylim(Ave[1] - 1,Ave[1] + 1)

    if di == 1:
        ax[di//2, di%2].set_xlim(Ave[0] - .11, Ave[0] + .11)
        ax[di//2, di%2].set_ylim(Ave[1] - .15, Ave[1] + .2)

    if di == 2:
        ax[di//2, di%2].set_xlim(Ave[0] - .03, Ave[0] + .03)
        ax[di//2, di%2].set_ylim(Ave[1] - .05, Ave[1] + .05)

    if di == 3:
        ax[di//2, di%2].set_xlim(Ave[0] - .01, Ave[0] + .01)
        ax[di//2, di%2].set_ylim(Ave[1] - .018, Ave[1] + .018)
f.legend([h1, h2, h3], ['Data \nsamples', 'Empirical \nGaussian distribution', 'Limiting \nGaussian distribution'],ncol = 3, loc = 'center', bbox_to_anchor = (0.5, 0.06), fontsize = 17, handlelength = 1)
f.tight_layout(rect = (0,0.1,1,1))
f.savefig(FIG_PATH + 'fig4.pdf', bbox_inches = 'tight')
#%%
np.random.seed(0)
mu_0 = np.random.rand(N, M)
mu_0 = mu_0 / np.sum(mu_0, axis = 1)[:, None]
N_ITER = 10000
delta = np.logspace(-3, 0, 50)
#%%
MU_mc = []
for di in delta:
    csi = []
    for l in range(0, N):
        csi.append(np.random.laplace(theta[0], b, size = N_ITER))
    csi = np.array(csi)
    MU_mc.append(asl_l(mu_0, csi, A, N_ITER, theta, b, N, di))
#%%
beliefv = np.array([MU_mc[i][-1][0] for i in range(len(MU_mc))])
beliefv9 = np.array([MU_mc[i][-1][9] for i in range(len(MU_mc))])

logbv1 = np.log(beliefv[:,0] / beliefv[:,1])
logbv2 = np.log(beliefv[:,0] / beliefv[:,2])

f, ax = plt.subplots(1, 1, figsize = (6, 2.5))
ax.set_xlim(1e-3, 1)
ax.set_ylim(-0.3, .65)
ax.set_xscale('log')
ax.invert_xaxis()
ax.plot(delta, ave1 * np.ones(len(delta)), color = 'k', linewidth = 2, linestyle = ':')
ax.plot(delta, ave2 * np.ones(len(delta)), color = 'k', linewidth = 2, linestyle = ':')
h1 = ax.plot(delta, logbv1, linewidth = 2, color = 'C1', linestyle = '-', marker = 'o')
h2 = ax.plot(delta, logbv2, linewidth = 2, color = 'C2', linestyle = '-', marker = 'o')
ax.legend([h1[0], h2[0]],[r'$\theta=2$', r'$\theta=3$'], ncol = 2, fontsize = 16, handlelength = 1, loc = 'lower right')
ax.annotate(r'${\sf m}_{\sf ave}(\theta)=%0.2f$' % ave1, (.65, ave1 - 0.15), xycoords=('axes fraction', 'data'), color = 'k', fontsize = 16)
ax.annotate(r'${\sf m}_{\sf ave}(\theta)=%0.2f$' % ave2, (.65, ave2 + 0.05), xycoords=('axes fraction', 'data'), color = 'k', fontsize = 16)
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
ax.set_xlabel(r'$\delta$', fontsize = 16)
ax.set_ylabel(r'$\bm{\lambda}^{\delta}_1(\theta)$', fontsize = 16)
f.savefig(FIG_PATH + 'fig3.pdf', bbox_inches = 'tight')
