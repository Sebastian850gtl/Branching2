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

save_path_BR = '../../results/BR_mono_100_01/'
save_path_ML = '../../results/ML_mono_100/'
save_path_CAML = '../../results/CAML_100/'
#Parameters
n_clusters = 100
alpha_range = [0,1/3,2/3,1]
# Initial distribution 
monodisperse = np.ones([n_clusters])/n_clusters

# Plots
x = 0.3
k = 2
from compute_probas import probs
from concatenator import concatenate_sim
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad

for i,alpha in enumerate(alpha_range):
    plt.figure(dpi = 300)
    #Browninan
    save_path_i_BR = save_path_BR + "alpha_beta_"+  str(i) + '_0'
    sample_sizes_BR, sample_times_BR = concatenate_sim(save_path_i_BR)
    n_samples,n_clusters = sample_times_BR.shape
    time_range_BR = np.linspace(0,3*np.mean(sample_times_BR[:,-1]),200)

    time_range = time_range_BR * 2 * n_clusters/(-np.log(0.02) + 3*np.log(2)-3/2)
    print("Computing probas for BR, parameters : alpha =" + str((alpha)))
    print("Number of samples :"+str(n_samples))
    probies = probs(sample_sizes_BR,sample_times_BR,time_range_BR,k,x)
    plt.plot(time_range,probies, label = r"BR")

    # ML
    save_path_i_ML = save_path_ML + 'ML_' + str(i) 
    sample_sizes_ML, sample_times_ML = concatenate_sim(save_path_i_ML)
    n_samples,n_clusters = sample_times_ML.shape
    time_range = np.linspace(0,3*np.mean(sample_times_ML[:,-1]),200)
    print("Computing probas for ML, parameters : alpha =" + str((alpha)))
    print("Number of samples :"+str(n_samples))
    probies = probs(sample_sizes_ML,sample_times_ML,time_range,k,x)
    plt.plot(time_range,probies, label = r"ML")

    # CAML
    save_path_i_CAML = save_path_CAML + 'CAML_' + str(i) 
    sample_sizes_CAML, sample_times_CAML = concatenate_sim(save_path_i_CAML)
    n_samples,n_clusters = sample_times_CAML.shape
    print("Computing probas for CAML, parameters : alpha =" + str((alpha)))
    print("Number of samples :"+str(n_samples))
    probies = probs(sample_sizes_CAML,sample_times_CAML,time_range,k,x)
    plt.plot(time_range,probies, label = r"CAML")

    plt.legend()
    plt.savefig(fig_path+file_name+'_'+str(i)+'_fig.png')