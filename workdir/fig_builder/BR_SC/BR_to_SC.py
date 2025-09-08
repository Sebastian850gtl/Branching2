
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
    save_times = save_path_i_SC + "_times_range.npy"

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

        time_range = np.linspace(0,np.mean(sample_times_SC[:,-1])*3,Nt)
        print("Computing branching probability")
        t0 = time()
        branching_proba_SC = prob_fun(sample_sizes_SC,sample_times_SC,time_range,masses,k)

        
        print("...done in %.3f s"%(time()- t0))

        np.save(save_file, branching_proba_SC)
        np.save(save_times,time_range)

    for radius,BR_folder in zip([0.1,0.07,0.03,0.01],BR_folders):
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

            time_range_BR = time_range * (n_clusters)**(alpha) *(-np.log(2*radius) + np.log(2))
            
            #time_range =  time_range_BR *(n_clusters)**(-alpha) *1/(-np.log(2*radius) + np.log(2)) #To be able to compare BR with SC and CASC we divide be the costant in front of the kernel.
            print("Computing branching probability")
            t0 = time()

            branching_proba_BR = prob_fun(sample_sizes_BR,sample_times_BR,time_range_BR,masses,k)
            print("...done in %.3f s"%(time()- t0))

            np.save(save_file, branching_proba_BR)
            np.save(save_times,time_range)

# Plotting branching probas side_side

for x in [0.15,0.28,0.37,0.45]:
    int_mass = int((x - mass_min) * Nx/(mass_max - mass_min))
    for i,alpha in enumerate(alpha_range):
        plt.figure(dpi = 300)

        save_path_i_SC = save_path_SC + 'alpha_%.3f'%(alpha)
        save_file = save_path_i_SC + "_pb.npy"
        save_times = save_path_i_SC + "_times_range.npy"
        print("Loading saved branching probability")
        t0 = time()
        branching_proba_SC = np.load(save_file)
        print("...done in %.3f s"%(time()- t0))
        time_range = np.load(save_times)

        plt.plot(time_range,branching_proba_SC[int_mass,:],label = "stochastic coalescence")
        for radius,BR_folder in zip([0.1,0.07,0.03,0.01],BR_folders):
            save_path_BR = '../../results/' + BR_folder + '/'
            # Browninan
            save_path_i_BR = save_path_BR + "alpha_beta_%.3f_%.3f"%(alpha,0)
            save_file = save_path_i_BR + "_pb.npy"
            save_times = save_path_i_BR + "_times_range.npy"
            print("Loading saved branching probability")
            t0 = time()
            branching_proba_BR = np.load(save_file)
            time_range = np.load(save_times)
            print("...done in %.3f s"%(time()- t0))

            plt.plot(time_range,branching_proba_BR[int_mass,:],label = r"Brownian coalescence $r_0 = %.2f"%(radius))
        plt.legend()
        plt.savefig(fig_path+"BR_to_SC"+'_%.3f_%.3f_fig.png'%(x,alpha))
#plt.show()
# Distances
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

for i,alpha in enumerate(alpha_range):
    

    save_path_i_SC = save_path_SC + 'alpha_%.3f'%(alpha)
    save_file = save_path_i_SC + "_pb.npy"
    save_times = save_path_i_SC + "_times_range.npy"
    print("Loading saved branching probability")
    t0 = time()
    branching_proba_SC = np.load(save_file)
    print("...done in %.3f s"%(time()- t0))
    time_range = np.load(save_times)
    radiuses = [0.1,0.07,0.03,0.01]

    dists = []
    for radius,BR_folder in zip(radiuses,BR_folders):
        save_path_BR = '../../results/' + BR_folder + '/'
        # Browninan
        save_path_i_BR = save_path_BR + "alpha_beta_%.3f_%.3f"%(alpha,0)
        save_file = save_path_i_BR + "_pb.npy"
        save_times = save_path_i_BR + "_times_range.npy"
        print("Loading saved branching probability")
        t0 = time()
        branching_proba_BR = np.load(save_file)
        time_range = np.load(save_times)
        print("...done in %.3f s"%(time()- t0))

        dist_radius = dist(branching_proba_BR,branching_proba_SC,1)
        dist_radius_1 = dist_radius[-1]
        print("Distance for radius %.3f : %.6f"%(radius,dist_radius_1))
        dists.append(dist_radius_1)

    plt.figure(dpi = 300)
    #plt.ylabel("distances")
    plt.ylabel(r"r_0")
    plt.scatter(radiuses,dists)        
    plt.legend()
    plt.savefig(fig_path+"BR_to_SC_dists"+'_%.3f_fig.png'%(alpha))