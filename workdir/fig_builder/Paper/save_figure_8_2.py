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

save_path = '../../results/figure8_2/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

from compute_probas import probs,state
from concatenator import concatenate_sim
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm, gamma
from scipy.special import gamma as Gamma

Ns = [36,38,40,42,44,46,48,50]
T = 17

D = 10
tmax = 2 * D
compute = 0
time_range_BR = np.linspace(0,tmax,300)
if compute :

    for N in Ns:
        BR_N_folder = 'BR_mono_nc' + str(N) + '_r001_eps05'
        save_path_BR = '../../results/' + BR_N_folder + '/'

        save_path_i_BR = save_path_BR + "alpha_beta_%.3f_%.3f"%(0,1/2)

        sample_sizes_BR, sample_times_BR = concatenate_sim(save_path_i_BR)

        probas = probs(sample_sizes_BR,sample_times_BR,time_range_BR,2,T/N)

        proba_function = np.array([time_range_BR / D, probas])
        np.save( save_path + 'proba_N_' + str(N) + '.npy',proba_function)
else:
    pass

plt.figure(dpi = 200)
for N in Ns:
    proba_function = np.load( save_path + 'proba_N_' + str(N) + '.npy')

    plt.plot( proba_function[0,:] , proba_function[1,:], label = "N = "+str(N))

plt.legend()
plt.savefig(fig_path + "figure8_2.png")
plt.show()


