import os,sys
import importlib
file_name =  os.path.splitext(os.path.basename(sys.argv[0]))[0]
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('../..')
wd = os.getcwd() # Defines the working directory

sys.path.append('../models')
sys.path.append('../lib')

import numpy as np
from ML import Coalescence
from CAML import CAML

#Files locations


fig_path = '../../results/fig/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

parameters_file_name = sys.argv[1]

#Parameters

param_module = importlib.import_module(parameters_file_name)

radius = param_module.radius
alpha_range = param_module.alpha_range
x = param_module.x
k = param_module.k

BR_folder = param_module.BR_folder
SC_folder = param_module.SC_folder
CASC_folder = param_module.CASC_folder

save_path_BR = '../../results/' + BR_folder + '/'
save_path_SC = '../../results/' + SC_folder + '/'
save_path_CASC = '../../results/' + CASC_folder + '/'

from compute_probas import probs,prob_fun
from concatenator import concatenate_sim
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad
from time import time

def dist(A,B,d):
    """ 
    A and B are 2-dimensional matrices corresponding to the probability functions first dimension is the mass discretization and second time
    times is a 1-dimensional vector of size equal to A.shape[1]
    d is the order of the distance on the mass dimension
    """

    Nx,Nt = A.shape # len(times) = Nt 
    res = np.zeros([Nt])
    D1 = np.abs(A-B)**d
    D2 = 1/Nx*np.sum(D1, axis = 0)**(1/d)
    maxi = 0
    for nt in range(Nt):
        maxi = max(maxi, D2[nt])
        res[nt] = maxi
    return res

Nx = 100
masses = np.linspace(0.1,1,Nx)
d = 1
k = 2
for i,alpha in enumerate(alpha_range):
    # Browninan
    print("Loading Brownian samples \n ...parameters: alpha = %.3f, beta = %.3f"%(alpha,0))
    t0 = time()
    save_path_i_BR = save_path_BR + "alpha_beta_%.3f_%.3f"%(alpha,0)
    sample_sizes_BR, sample_times_BR = concatenate_sim(save_path_i_BR)

    n_samples,n_clusters = sample_times_BR.shape
    print("...done in %.3f s"%(time()- t0))

    time_range_BR = np.linspace(0,np.mean(sample_times_BR[:,-1])/4,200)
    
    time_range =  time_range_BR *(n_clusters)**(-alpha) *1/(-np.log(2*radius) + np.log(2)) #To be able to compare BR with SC and CASC we divide be the costant in front of the kernel.
    print("Computing branching probability")
    t0 = time()
    branching_proba_BR = prob_fun(sample_sizes_BR,sample_times_BR,time_range_BR,masses,k)
    print("...done in %.3f s"%(time()- t0))
    # SC
    print("Loading SC samples \n ...parameters: alpha = %.3f"%(alpha))
    t0 = time()
    save_path_i_SC = save_path_SC + 'alpha_%.3f'%(alpha)
    sample_sizes_SC, sample_times_SC = concatenate_sim(save_path_i_SC)
    print("...done in %.3f s"%(time()- t0))

    print("Computing branching probability")
    t0 = time()
    branching_proba_SC = prob_fun(sample_sizes_SC,sample_times_SC,time_range,masses,k)
    print("...done in %.3f s"%(time()- t0))

    # CASC
    print("Loading CASC samples \n ...parameters: alpha = %.3f"%(alpha))
    t0 = time()
    save_path_i_CASC = save_path_CASC + 'alpha_%.3f'%(alpha)
    sample_sizes_CASC, sample_times_CASC = concatenate_sim(save_path_i_CASC)
    print("...done in %.3f s"%(time()- t0))
    
    print("Computing branching probability")
    t0 = time()
    branching_proba_CASC = prob_fun(sample_sizes_CASC,sample_times_CASC,time_range,masses,k)
    print("...done in %.3f s"%(time()- t0))
    # #Plot
    # plt.figure(dpi = 300)
    # plt.plot(time_range,branching_proba_BR[28,:])
    # plt.plot(time_range,branching_proba_SC[28,:])
    # plt.plot(time_range,branching_proba_CASC[28,:])
    # plt.savefig(fig_path+parameters_file_name+'_%.3f_plot.png'%(alpha))
    # Distances
    plt.figure(dpi = 300)
    # BR-SC
    print("Computing distances between BR and SC")
    t0 = time()
    dists = dist(branching_proba_BR,branching_proba_SC,d)
    plt.plot(time_range, dists, label = r'distance $d_{%d}(p_{BR},p_{SC})$'%(d))
    print("...done in %.3f s"%(time()- t0))

    # BR-CASC
    print("Computing distances between BR and CASC")
    t0 = time()
    dists = dist(branching_proba_BR,branching_proba_CASC,d)
    plt.plot(time_range, dists, label = r'distance $d_{%d}(p_{BR},p_{CASC})$'%(d))
    print("...done in %.3f s"%(time()- t0))

    # SC-CASC
    print("Computing distances between SC and CASC")
    t0 = time()
    dists = dist(branching_proba_CASC,branching_proba_SC,d)
    plt.plot(time_range, dists, label = r'distance $d_{%d}(p_{SC},p_{CASC})$'%(d))
    print("...done in %.3f s"%(time()- t0))

    plt.legend()
    plt.savefig(fig_path+parameters_file_name+'_%.3f_dist.png'%(alpha))