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

from compute_probas import probs
from concatenator import concatenate_sim
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad

for i,alpha in enumerate(alpha_range):
    plt.figure(dpi = 300)
    plt.title(r"$x = %.3f$ and $\alpha = %.3f$"%(x,alpha))
    #Browninan
    print("Computing probas for BR, parameters : alpha = %.3f, beta = %.3f"%(alpha,0))
    save_path_i_BR = save_path_BR + "alpha_beta_%.3f_%.3f"%(alpha,0)
    sample_sizes_BR, sample_times_BR = concatenate_sim(save_path_i_BR)

    n_samples, n_clusters = sample_times_BR.shape
    n_samples,n_clusters = sample_times_BR.shape
    print(n_clusters)
    print(sample_sizes_BR[0,-1,:])
    print(np.sum(sample_sizes_BR[:,-1,:])/n_samples)
    time_range_BR = np.linspace(0,3.5*np.mean(sample_times_BR[:,-1]),200)

    probies = probs(sample_sizes_BR,sample_times_BR,time_range_BR,k,x)
    #print(probies)
    time_range =  time_range_BR *(n_clusters)**(-alpha) *1/(-np.log(2*radius) + np.log(2))
    plt.plot(time_range,probies, label = r"BR $\beta = 0$")

    #print("Computing probas for BR, parameters : alpha = %.3f, beta = %.3f"%(alpha,0.5))
    #save_path_i_BR = save_path_BR + "alpha_beta_%.3f_%.3f"%(alpha,0.5)
    #sample_sizes_BR, sample_times_BR = concatenate_sim(save_path_i_BR)
    #n_samples,n_clusters = sample_times_BR.shape
    #time_range_BR = np.linspace(0,3*np.mean(sample_times_BR[:,-1]),200)
    #time_range = time_range_BR *(n_clusters)**(-alpha)/(-np.log(2*radius) + np.log(2))
    #probies = probs(sample_sizes_BR,sample_times_BR,time_range_BR,k,x)
    #plt.plot(time_range,probies, label = r"BR, $\beta = 0.5$")

     #ML
    print("Computing probas for ML, parameters : alpha = %.3f"%(alpha) )
    save_path_i_ML = save_path_SC + 'alpha_%.3f'%(alpha)
    sample_sizes_ML, sample_times_ML = concatenate_sim(save_path_i_ML)
    n_samples,n_clusters = sample_times_ML.shape

    #time_range = np.linspace(0,3*np.mean(sample_times_ML[:,-1]),100)
    probies = probs(sample_sizes_ML,sample_times_ML,time_range,k,x)
    plt.plot(time_range,probies, label = r"SC")

    # CAML
    print("Computing probas for CAML, parameters : alpha = %.3f"%(alpha) )
    save_path_i_CAML = save_path_CASC + 'alpha_%.3f'%(alpha)
    sample_sizes_CAML, sample_times_CAML = concatenate_sim(save_path_i_CAML)
    n_samples,n_clusters = sample_times_CAML.shape
    
    probies = probs(sample_sizes_CAML,sample_times_CAML,time_range,k,x)
    plt.plot(time_range,probies, label = r"CASC")

    plt.legend()
    plt.savefig(fig_path+parameters_file_name+'_'+str(i)+'_fig.png')