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
beta_range = param_module.beta_range
x = param_module.x
k = param_module.k

BR_folder = param_module.BR_folder


save_path_BR = '../../results/' + BR_folder + '/'


from compute_probas import probs
from concatenator import concatenate_sim
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm, gamma
from scipy.special import gamma as Gamma


for i,alpha in enumerate(alpha_range):
    plt.figure(dpi = 200)
    plt.title(r"$x = %.3f$ and $\alpha = %.3f$"%(x,alpha))
    plt.xlabel("Time")
    plt.ylabel("Branching probability")

    for j,beta in enumerate(beta_range):
        #Browninan
        print("Computing probas for BR, parameters : alpha = %.3f, beta = %.3f"%(alpha,beta))
        save_path_i_BR = save_path_BR + "alpha_beta_%.3f_%.3f"%(alpha,beta)
        sample_sizes_BR, sample_times_BR = concatenate_sim(save_path_i_BR)

        n_samples, n_clusters = sample_times_BR.shape
        n_samples,n_clusters = sample_times_BR.shape
        time_range_BR = np.linspace(0,3.5*np.mean(sample_times_BR[:,-1]),200)

        probies = probs(sample_sizes_BR,sample_times_BR,time_range_BR,k,x)
        #print(probies)
        time_range =  time_range_BR *(n_clusters)**(-alpha) *1/(-np.log(2*radius*n_clusters**beta) + np.log(2)) #To be able to compare BR with SC and CASC we divide be the costant in front of the kernel.
        plt.plot(time_range,probies, label = r"$\beta = %.3f$"%(beta))

    
    plt.legend()
    plt.savefig(fig_path+parameters_file_name+'_%.3f_fig.png'%(alpha))