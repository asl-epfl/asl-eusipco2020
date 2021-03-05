"""
This code can be used to generate simulations similar to Fig. 1 in the following paper:
Virginia Bordignon, Vincenzo Matta, and Ali H. Sayed,  "Adaptation in online social learning,''  Proc. EUSIPCO, pp. 1-5, Amsterdam, The Netherlands, August 2020.

Please note that the code is not generally perfected for performance, but is rather meant to illustrate certain results from the paper. The code is provided as-is without guarantees.

July 2020 (Author: Virginia Bordignon)
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import os
from functions import *
#%%
getcontext().prec = 200
mpl.style.use('seaborn-deep')
plt.rcParams.update({'text.usetex': True})
mpl.rcParams['font.family'] = 'serif'
#%%
FIG_PATH = 'figs/'
if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)
#%%
N = 10
M = 3
np.random.seed(0)
# %%
################################ Build Network Topology ################################
G = np.random.choice([0.0,1.0], size=(N, N), p = [0.5,0.5])
G = G + np.eye(N)
G = (G > 0) * 1.0
lamb = .5

A = np.zeros((N, N))
A = G / G.sum(axis = 0)
A_dec = np.array([[Decimal(x) for x in y] for y in A])
#%%
################################ Run Social Learning ################################
np.random.seed(20)
N_ITER = 600
theta = np.array([1., 2., 3.]) * .6
thetadec = decimal_array(theta)
var = 1
vardec = Decimal(var)
x = np.linspace(-10, 10, 1000)
x_dec = decimal_array(x)
dt = (max(x) - min(x)) / len(x)
dtdec = (max(x_dec) - min(x_dec)) / len(x_dec)
# %%
mu_0 = np.random.rand(N, M)
mu_0 = mu_0 / np.sum(mu_0, axis = 1)[:, None]
mu_0dec = decimal_array(mu_0)
delta = .1
# %%
csi=[]
for l in range(0, N):
    csi.append(np.hstack([theta[0] + np.sqrt(var) * np.random.randn(200), theta[2] + np.sqrt(var) * np.random.randn(N_ITER - 200)]))
csi = np.array(csi)
csidec = decimal_array(csi)
# %%
MU_adapt = asl(mu_0, csi, A, N_ITER, theta, var, N, delta)
MU = sl(mu_0dec, csidec, A_dec, N_ITER, thetadec, vardec, N)
# %%
dec_sl = np.argmax(np.array(MU), axis = 2)[:,0]
dec_asl = np.argmax(np.array(MU_adapt), axis = 2)[:,0]
# %%
fig, ax = plt.subplots(2, 2, figsize = (6, 4.5), gridspec_kw = {'height_ratios': [1, 1]})
plot_agent_dec(MU, N_ITER, 0, ax[0, 0])
ax[0, 0].set_ylabel('Belief \nof Agent 1', fontsize = 16)
ax[0, 0].set_xlabel('Iteration', fontsize = 16)
ax[0, 0].set_xticks([0, 200, 400, 600])
ax[1, 0].scatter(np.argwhere(dec_sl == 0)[:, 0], np.ones(len(np.argwhere(dec_sl == 0)[:, 0])), s = 20, marker = '.', color = 'C0')
ax[1, 0].scatter(np.argwhere(dec_sl == 1)[:, 0], 2 * np.ones(len(np.argwhere(dec_sl == 1)[:, 0])), s = 20, marker = '.', color = 'C1')
ax[1, 0].scatter(np.argwhere(dec_sl == 2)[:, 0], 3 * np.ones(len(np.argwhere(dec_sl == 2)[:, 0])), s = 20, marker = '.', color = 'C2')
ax[1, 0].set_xlim(0, N_ITER)
ax[1, 0].set_ylabel('Opinion \nof Agent 1', fontsize = 16)
ax[1, 0].set_xlabel('Iteration', fontsize = 16)
ax[0, 0].set_title('Social Learning', fontsize = 16)
ax[1, 0].set_ylim(0.5, 3.5)
ax[1, 0].yaxis.grid()
ax[1, 0].set_axisbelow(True)
ax[1, 0].tick_params(axis = 'x', which = 'major', labelsize = 16)
ax[1, 0].set_yticks([1,2,3])
ax[1, 0].set_xticks([0, 200, 400, 600])
ax[1, 0].set_yticklabels(['S','C','R'], size = 14)
plot_agent(MU_adapt, N_ITER, 0, ax[0,1])
ax[0, 1].set_xlabel('Iteration', fontsize = 16)
ax[0, 1].set_xticks([0,200,400,600])
ax[1, 1].scatter(np.argwhere(dec_asl == 0) [:, 0], np.ones(len(np.argwhere(dec_asl == 0)[:, 0])), s = 20, marker = '.', color ='C0')
ax[1, 1].scatter(np.argwhere(dec_asl == 1)[:, 0], 2 * np.ones(len(np.argwhere(dec_asl == 1)[:, 0])), s = 20, marker = '.', color = 'C1')
ax[1, 1].scatter(np.argwhere(dec_asl == 2)[:, 0], 3 * np.ones(len(np.argwhere(dec_asl == 2)[:, 0])), s = 20, marker = '.', color = 'C2')
ax[1, 1].set_xlim(0, N_ITER)
ax[0, 1].set_ylabel('')
ax[1, 1].set_xlabel('Iteration', fontsize = 16)
ax[0, 1].set_title('Adaptive Social Learning', fontsize = 16)
ax[1,  1].set_ylim(0.5, 3.5)
ax[0, 1].set_ylim(0.1, .6)
ax[1, 1].yaxis.grid()
ax[1, 1].set_axisbelow(True)
ax[1, 1].tick_params(axis = 'x', which = 'major', labelsize = 16)
ax[1, 1].set_yticks([1, 2, 3])
ax[1, 1].set_xticks([0, 200, 400, 600])
ax[1, 1].set_yticklabels(['S', 'C', 'R'], size = 14)
fig.legend([r'Sunny', r'Cloudy', r'Rainy'], ncol = M, loc = 'center', bbox_to_anchor = (0.5, 0.05), fontsize = 16, handlelength = 1)
fig.tight_layout(rect = (0, 0.05, 1, 1))
fig.savefig(FIG_PATH + 'fig1.pdf', bbox_inches = 'tight')
