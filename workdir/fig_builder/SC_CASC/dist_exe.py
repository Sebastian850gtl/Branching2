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
# parameters_file_name = "mono_nc100_r001_eps05"
# #Parameters

param_module = importlib.import_module(parameters_file_name)

alpha_range = param_module.alpha_range
k = param_module.k

SC_folder = param_module.SC_folder
CASC_folder = param_module.CASC_folder


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


use_saved_branching_proba = 1
Nx = 100  # Used only if use_saved_branching_proba = 0

Nt = 300 


masses = np.linspace(0.1,1,Nx) # Used only if use_saved_branching_proba = 0
d = 1
k = 2

time_range = time_range[:Nt]

plt.figure(dpi = 300)
plt.xlabel(r"Time horizon $T$")


for i,alpha in enumerate(alpha_range):
    # SC
    save_path_i_SC = save_path_SC + 'alpha_%.3f'%(alpha)
    save_file = save_path_i_SC + "_pb.npy"
    save_times = save_path_i_SC + "_times_range.npy"
    if use_saved_branching_proba and os.path.exists(save_file) and os.path.exists(save_times):
        print("Loading saved branching probability")
        t0 = time()
        branching_proba_SC = np.load(save_file)
        time_range = np.load(save_times)
        print("...done in %.3f s"%(time()- t0))
    else:
        print("Loading SC samples \n ...parameters: alpha = %.3f"%(alpha))
        t0 = time()
        sample_sizes_SC, sample_times_SC = concatenate_sim(save_path_i_SC)
        print("...done in %.3f s"%(time()- t0))

        print("Computing branching probability")
        t0 = time()
        branching_proba_SC = prob_fun(sample_sizes_SC,sample_times_SC,time_range,masses,k)

        time_range_BR = np.linspace(0,np.mean(sample_times_SC[:,-1]),Nt)
        print("...done in %.3f s"%(time()- t0))

        np.save(save_file, branching_proba_SC)
        np.save(save_times,time_range)

    # CASC
    save_path_i_CASC = save_path_CASC + 'alpha_%.3f'%(alpha)
    save_file = save_path_i_CASC + "_pb.npy"
    if use_saved_branching_proba and os.path.exists(save_file):
        print("Loading saved branching probability")
        t0 = time()
        branching_proba_CASC = np.load(save_file)
        print("...done in %.3f s"%(time()- t0))
    else:
        print("Loading CASC samples \n ...parameters: alpha = %.3f"%(alpha))
        t0 = time()
        sample_sizes_CASC, sample_times_CASC = concatenate_sim(save_path_i_CASC)
        print("...done in %.3f s"%(time()- t0))
        
        print("Computing branching probability")
        t0 = time()
        branching_proba_CASC = prob_fun(sample_sizes_CASC,sample_times_CASC,time_range,masses,k)
        print("...done in %.3f s"%(time()- t0))
        np.save(save_file, branching_proba_CASC)
    