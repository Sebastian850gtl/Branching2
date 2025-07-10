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

#Files locations


fig_path = '../../results/fig/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)


#Parameters

# SC_folder = param_module.SC_folder
# CASC_folder = param_module.CASC_folder

# save_path_SC = '../../results/' + SC_folder + '/'
# save_path_CASC = '../../results/' + CASC_folder + '/'

from compute_probas import probs
from concatenator import concatenate_sim
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
from scipy.special import gamma as Gamma


alpha = 1/3
    
save_path_i_ML = save_path_SC + 'alpha_%.3f'%(alpha)
sample_sizes_ML, sample_times_ML = concatenate_sim(save_path_i_ML)
n_samples,n_clusters = sample_times_ML.shape

time_range = np.linspace(0,3*np.mean(),100)
probies = probs(sample_sizes_ML,sample_times_ML,time_range,k,x)
plt.plot(time_range,probies, label = r"SC")