import numpy as np
from decimal import *
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def gaussian_dec(x, m, var):
    '''
    Computes the Gaussian pdf value (Decimal type) at x.
    x: value at which the pdf is computed (Decimal type)
    m: mean (Decimal type)
    var: variance
    '''
    p = np.exp(-(x-m)**Decimal(2)/(Decimal(2)*Decimal(var)))/(np.sqrt(Decimal(2)*Decimal(np.pi)*Decimal(var)))
    return p

def gaussian(x, m, var):
    '''
    Computes the Gaussian pdf value at x.
    x: value at which the pdf is computed (Decimal type)
    m: mean
    var: variance
    '''
    p = np.exp(-(x-m)**2/(2*var))/(np.sqrt(2*np.pi*var))
    return p

def laplace(x, m, b):
    '''
    Computes the Laplace pdf value at x.
    x: value at which the pdf is computed (Decimal type)
    m: mean
    b: scale parameter
    '''
    p = np.exp(-np.abs((x-m)/b))/(2*b)
    return p

def bayesian_update(L, mu):
    '''
    Computes the Bayesian update.
    L: likelihoods matrix
    mu: beliefs matrix
    '''
    aux = L*mu
    bu = aux/aux.sum(axis = 1)[:, None]
    return bu

def asl_bayesian_update(L, mu, delta):
    '''
    Computes the adaptive Bayesian update.
    L: likelihoods matrix
    mu: beliefs matrix
    delta: step size
    '''
    aux = L**(delta)*mu**(1-delta)
    bu = aux/aux.sum(axis = 1)[:, None]
    return bu

def DKL(m,n,dx):
    '''
    Computes the KL divergence between m and n.
    m: true distribution in vector form
    n: second distribution in vector form
    dx : sample size
    '''
    mn=m/n
    mnlog= np.log(mn)
    return np.sum(m*dx*mnlog)

def Cave(L, t1, t2, dt):
    '''Computes the covariance between the log likelihood ratios for two hypotheses.
    L: likelihood Functions
    t1: hypothesis 1
    t2: hypothesis 2
    dt: step size
    '''
    aux=L[0]/L[t1]
    auxlog= np.log(aux)
    aux1=L[0]/L[t2]
    auxlog1= np.log(aux1)
    return np.sum(L[0]*dt*auxlog1*auxlog)-DKL(L[0], L[t1], dt)*DKL(L[0], L[t2], dt)

def decimal_array(arr):
    '''
    Converts an array to an array of Decimal objects.
    arr: array to be converted
    '''
    if len(arr.shape)==1:
        return np.array([Decimal(y) for y in arr])
    else:
        return np.array([[Decimal(x) for x in y] for y in arr])

def float_array(arr):
    '''
    Converts an array to an array of float objects.
    arr: array to be converted
    '''
    if len(arr.shape)==1:
        return np.array([float(y) for y in arr])
    else:
        return np.array([[float(x) for x in y] for y in arr])

def asl(mu_0, csi, A, N_ITER, theta, var, N, delta = 0):
    '''
    Executes the adaptive social learning algorithm with Gaussian likelihoods.
    mu_0: initial beliefs
    csi: observations
    A: Combination matrix
    N_ITER: number of iterations
    theta: vector of means for the Gaussian likelihoods
    var: variance of Gaussian likelihoods
    N: number of agents
    delta: step size
    '''
    mu = mu_0.copy()
    MU = [mu]
    for i in range(N_ITER):
        L_i = np.array([gaussian(csi[:,i], t, var) for t in theta]).T
        L_i[:N//3, 1]= L_i[:N//3, 0]
        L_i[N//3:2*N//3, 1]= L_i[N//3:2*N//3, 2]
        L_i[2*N//3:, 2]= L_i[2*N//3:, 0]
        psi = asl_bayesian_update(L_i, mu, delta)
        decpsi = np.log(psi)
        mu = np.exp((A.T).dot(decpsi))/np.sum(np.exp((A.T).dot(decpsi)),axis =1)[:,None]
        MU.append(mu)
    return MU

def asl_l(mu_0, csi, A, N_ITER, theta, b, N, delta = 0, adaptive = True):
    '''
    Executes the adaptive social learning algorithm with Laplace likelihoods.
    mu_0: initial beliefs
    csi: observations
    A: Combination matrix
    N_ITER: number of iterations
    theta: vector of means for the Laplace likelihoods
    b: scale parameter of the Laplace likelihoods
    N: number of agents
    delta: step size
    '''
    mu = mu_0.copy()
    MU = [mu]
    for i in range(N_ITER):
        L_i = np.array([laplace(csi[:,i], t, b) for t in theta]).T
        L_i[:N//3, 1]= L_i[:N//3, 0]
        L_i[N//3:2*N//3, 1]= L_i[N//3:2*N//3, 2]
        L_i[2*N//3:, 2]= L_i[2*N//3:, 0]
        if adaptive:
            psi = asl_bayesian_update(L_i, mu, delta)
        else:
            psi = bayesian_update(L_i, mu)
        decpsi = np.log(psi)
        mu = np.exp((A.T).dot(decpsi))/np.sum(np.exp((A.T).dot(decpsi)),axis =1)[:,None]
        MU.append(mu)
    return MU


def sl(mu_0, csi, A, N_ITER, thetadec, vardec, N):
    '''
    Executes the social learning algorithm with Gaussian likelihoods.
    mu_0: initial beliefs
    csi: observations
    A: Combination matrix
    N_ITER: number of iterations
    thetadec: vector of means for the Gaussian likelihoods (Decimal type)
    vardec: variance of Gaussian likelihoods (Decimal type)
    N: number of agents
    '''
    mu = mu_0.copy()
    MU = [mu]
    for i in range(N_ITER):
        L_i = np.array([gaussian_dec(csi[:,i], t, vardec) for t in thetadec]).T
        L_i[:N//3, 1]= L_i[:N//3, 0]
        L_i[N//3:2*N//3, 1]= L_i[N//3:2*N//3, 2]
        L_i[2*N//3:, 2]= L_i[2*N//3:, 0]
        psi = bayesian_update(L_i, mu)
        decpsi = np.array([[x.ln() for x in y] for y in psi])
        mu = np.exp((A.T).dot(decpsi))/np.sum(np.exp((A.T).dot(decpsi)),axis =1)[:,None]
        MU.append(mu)
    return MU

def plot_agent(MU, N_ITER, ag=0, ax=[]):
    '''
    Plot the evolution of beliefs for one agent.
    MU: beliefs evolution for all agents
    N_ITER: number of iterations
    ag: chosen agent
    ax: axis specification
    '''
    vec = np.array([MU[k][ag] for k in range(len(MU))])
    d = np.zeros(2)
    if abs(vec[-1,1] - vec[-1,0])< 0.05:
        d[0] = .1
    if abs(vec[-1,2] - vec[-1,0])  or  abs(vec[-1,2] - vec[-1,1]) < .05:
        d[1] = .1

    if ax:
        ax.plot(vec[:,0], linewidth='2', color = 'C0', label=r'$\theta=1$', alpha=0.8)
        ax.plot(vec[:,1], linewidth='2', color = 'C1', label = r'$\theta=2$', alpha=0.8)
        ax.plot(vec[:,2], linewidth='2', color = 'C2', label = r'$\theta=3$', alpha=0.8)
        ax.set_xlim([0,N_ITER])
        ax.set_ylim([-0.1,1.1])
        ax.set_xlabel(r'$i$', fontsize = 18)
        ax.set_ylabel(r'$\bm{{\mu}}_{{{l},i}}(\theta)$'.format(l=1), fontsize = 18)
        ax.tick_params(axis='both', which='major', labelsize=18)
    else:
        plt.figure(figsize=(5,3))
        plt.plot(vec[:,0], linewidth='2', color = 'C0', label=r'$\theta=1$', alpha=0.8)
        plt.plot(vec[:,1], linewidth='2', color = 'C1', label = r'$\theta=2$', alpha=0.8)
        plt.plot(vec[:,2], linewidth='2', color = 'C2', label = r'$\theta=3$', alpha=0.8)
        plt.xlim([0,N_ITER])
        plt.ylim([-0.1,1.1])
        plt.xlabel(r'$i$', fontsize = 18)
        plt.ylabel(r'$\bm{{\mu}}_{{{l},i}}(\theta)$'.format(l=1), fontsize = 18)
        plt.figlegend(ncol = M,loc = 1, fontsize = 18, bbox_to_anchor=(0.44, -.33, 0.5, 0.5))
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tight_layout()

def plot_agent_dec(MU, N_ITER, ag=0, ax=[]):
    '''
    Plot the evolution of beliefs for one agent using Decimal values.
    MU: beliefs evolution for all agents
    N_ITER: number of iterations
    ag: chosen agent
    ax: axis specification
    '''
    vec = np.array([MU[k][ag] for k in range(len(MU))])
    d = np.array([Decimal(0), Decimal(0)])
    if abs(vec[-1,1] - vec[-1,0])< Decimal(.05):
        d[0] = Decimal(.1)
    if abs(vec[-1,2] - vec[-1,0])  or  abs(vec[-1,2] - vec[-1,1]) < Decimal(.05):
        d[1] = Decimal(.1)
    if ax:
        ax.plot(vec[:,0], linewidth='2', color = 'C0', label=r'$\theta=1$', alpha=0.8)
        ax.plot(vec[:,1], linewidth='2', color = 'C1', label = r'$\theta=2$', alpha=0.8)
        ax.plot(vec[:,2], linewidth='2', color = 'C2', label = r'$\theta=3$', alpha=0.8)
        ax.set_xlim([0,N_ITER])
        ax.set_ylim([-0.1,1.1])
        ax.set_xlabel(r'$i$', fontsize = 18)
        ax.set_ylabel(r'$\bm{{\mu}}_{{{l},i}}(\theta)$'.format(l=1), fontsize = 18)
        ax.tick_params(axis='both', which='major', labelsize=18)
    else:
        plt.figure(figsize=(5,3))
        plt.plot(vec[:,0], linewidth='2', color = 'C0', label=r'$\theta=1$', alpha=0.8)
        plt.plot(vec[:,1], linewidth='2', color = 'C1', label = r'$\theta=2$', alpha=0.8)
        plt.plot(vec[:,2], linewidth='2', color = 'C2', label = r'$\theta=3$', alpha=0.8)
        plt.xlim([0,N_ITER])
        plt.ylim([-0.1,1.1])
        plt.xlabel(r'$i$', fontsize = 18)
        plt.ylabel(r'$\bm{{\mu}}_{{{l},i}}(\theta)$'.format(l=1), fontsize = 18)
        plt.figlegend(ncol = M,loc = 1, fontsize = 18, bbox_to_anchor=(0.44, -.33, 0.5, 0.5))
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tight_layout()

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    '''
    Create a plot of the covariance confidence ellipse of *x* and *y*. Inspired on the code in:
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    '''
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def gaussian_ellipse(C, Ave, ax, n_std=3.0, facecolor='none', **kwargs):
    '''
    Create a plot of the covariance confidence ellipse relative to the Covariance matrix C.
    C: covariance matrix
    Ave: average vector
    n_std: number of standard deviation ellipses
    '''
    cov = C
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = Ave[0]
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = Ave[1]
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
