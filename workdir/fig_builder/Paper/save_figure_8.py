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

save_path = '../../results/figure8/'

from compute_probas import probs,state
from concatenator import concatenate_sim
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm, gamma
from scipy.special import gamma as Gamma

Ns = [26,28,30,32,34]
T = 12


for N in Ns:
    BR_N_folder = 'BR_mono_nc' + str(N) + '_r001_eps05'
    save_path_BR = '../../results/' + BR_N_folder + '/'

    save_path_i_BR = save_path_BR + "alpha_beta_%.3f_%.3f"%(0,1/2)

    sample_sizes_BR, sample_times_BR = concatenate_sim(save_path_i_BR)

    time_range_BR = np.linspace(0,2,300)

    probas = probs(sample_sizes_BR,sample_times_BR,time_range_BR,2,T/N)

    proba_function = np.array(time_range_BR, probas)
    #np.save(proba_function, save_path + 'proba_N_' + str(N) + '.npy')

    plt.plot( proba_function[0,:] , proba_function[1,:], label = "N = "+str(N))
    plt.legend()


