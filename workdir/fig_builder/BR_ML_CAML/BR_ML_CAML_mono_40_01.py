import os,sys

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

save_path_BR = '../../results/BR_mono_40_01/'
save_path_ML = '../../results/ML_mono_40/'
save_path_CAML = '../../results/CAML_40/'
#Parameters
#alpha_range = [0,1/3,2/3,1]
radius = 0.01
alpha_range = [0,1/3,2/3,1]
# Plots
x = 0.45
k = 2
from compute_probas import probs
from concatenator import concatenate_sim
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad

for i,alpha in enumerate(alpha_range):
    plt.figure(dpi = 300)
    #Browninan
    print("Computing probas for BR, parameters : alpha =" + str((alpha)))
    save_path_i_BR = save_path_BR + "alpha_beta_"+  str(i) + '_0'
    sample_sizes_BR, sample_times_BR = concatenate_sim(save_path_i_BR)
    n_samples,n_clusters = sample_times_BR.shape
    time_range_BR = np.linspace(0,3*np.mean(sample_times_BR[:,-1]),200)

    time_range = 2/(1+alpha)*time_range_BR *(n_clusters)**(-alpha)/(-np.log(2*radius) + np.log(2))
    probies = probs(sample_sizes_BR,sample_times_BR,time_range_BR,k,x)
    plt.plot(time_range,probies, label = r"BR")

    # ML
    print("Computing probas for ML, parameters : alpha =" + str((alpha)))
    save_path_i_ML = save_path_ML + 'ML_' + str(i) 
    sample_sizes_ML, sample_times_ML = concatenate_sim(save_path_i_ML)
    n_samples,n_clusters = sample_times_ML.shape

    time_range = np.linspace(0,3*np.mean(sample_times_ML[:,-1]),200)
    probies = probs(sample_sizes_ML,sample_times_ML,time_range,k,x)
    plt.plot(time_range,probies, label = r"ML")

    # CAML
    print("Computing probas for CAML, parameters : alpha =" + str((alpha)))
    save_path_i_CAML = save_path_CAML + 'CAML_' + str(i) 
    sample_sizes_CAML, sample_times_CAML = concatenate_sim(save_path_i_CAML)
    n_samples,n_clusters = sample_times_CAML.shape
    
    probies = probs(sample_sizes_CAML,sample_times_CAML,time_range,k,x)
    plt.plot(time_range,probies, label = r"CAML")

    plt.legend()
    plt.savefig(fig_path+file_name+'_'+str(i)+'_fig.png')