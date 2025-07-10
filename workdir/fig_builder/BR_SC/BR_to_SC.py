
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


fig_path = '../../results/fig/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

from compute_probas import probs,prob_fun
from concatenator import concatenate_sim
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad
from time import time


radius = 0.001
alpha_range = [0,1/3,2/3,1]
#beta_range = [0,1/3,1/2,1]


BR_folders = ['BR_mono_nc50_r1_eps05','BR_mono_nc50_r07_eps05','BR_mono_nc50_r03_eps05','BR_mono_nc50_r01_eps05']
SC_folder = 'SC_mono_nc50'

save_path_SC = '../../results/' + SC_folder + '/'


Nx, Nt = 100, 300 
mass_min,mass_max = 0.1, 1/2
masses = np.linspace(mass_min,mass_max,Nx) # Used only if use_saved_branching_proba = 0
d = 1
k = 2

# Computing branching probabilities
use_saved_branching_proba = 1
for i,alpha in enumerate(alpha_range):
    # SC
    save_path_i_SC = save_path_SC + 'alpha_%.3f'%(alpha)
    save_file = save_path_i_SC + "_pb.npy"
    if use_saved_branching_proba and os.path.exists(save_file):
        print("Loading saved branching probability")
        t0 = time()
        branching_proba_SC = np.load(save_file)
        print("...done in %.3f s"%(time()- t0))
    else:
        print("Loading SC samples \n ...parameters: alpha = %.3f"%(alpha))
        t0 = time()
        sample_sizes_SC, sample_times_SC = concatenate_sim(save_path_i_SC)
        print("...done in %.3f s"%(time()- t0))

        print("Computing branching probability")
        t0 = time()
        branching_proba_SC = prob_fun(sample_sizes_SC,sample_times_SC,time_range,masses,k)
        print("...done in %.3f s"%(time()- t0))

        np.save(save_file, branching_proba_SC)
    
    for tag,radius,BR_folder in enumerate(zip([0.1,0.07,0.03,0.01],BR_folders)):
        save_path_BR = '../../results/' + BR_folder + '/'
        # Browninan
        save_path_i_BR = save_path_BR + "alpha_beta_%.3f_%.3f"%(alpha,0)
        save_file = save_path_i_BR + "_pb.npy"
        save_times = save_path_i_BR + "_times_range.npy"
        if use_saved_branching_proba and os.path.exists(save_file) and os.path.exists(save_times):
            print("Loading saved branching probability")
            t0 = time()
            branching_proba_BR = np.load(save_file)
            time_range = np.load(save_times)
            print("...done in %.3f s"%(time()- t0))
        else:
            print("Loading Brownian samples \n ...parameters: alpha = %.3f, beta = %.3f"%(alpha,0))
            t0 = time()

            sample_sizes_BR, sample_times_BR = concatenate_sim(save_path_i_BR)

            n_samples,n_clusters = sample_times_BR.shape
            print("...done in %.3f s"%(time()- t0))

            time_range_BR = np.linspace(0,np.mean(sample_times_BR[:,-1])*3,Nt)
            
            time_range =  time_range_BR *(n_clusters)**(-alpha) *1/(-np.log(2*radius) + np.log(2)) #To be able to compare BR with SC and CASC we divide be the costant in front of the kernel.
            print("Computing branching probability")
            t0 = time()

            branching_proba_BR = prob_fun(sample_sizes_BR,sample_times_BR,time_range_BR,masses,k)
            print("...done in %.3f s"%(time()- t0))

            np.save(save_file, branching_proba_BR)
            np.save(save_times,time_range)

# Plotting branching probas side_side

x = 0.28


# Distances