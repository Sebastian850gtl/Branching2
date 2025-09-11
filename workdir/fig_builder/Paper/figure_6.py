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

BR_folder = 'BR_mono_nc20_r001_eps05'

save_path_BR = '../../results/' + BR_folder + '/'

from compute_probas import probs,state
from concatenator import concatenate_sim
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm, gamma
from scipy.special import gamma as Gamma


save_path_i_BR = save_path_BR + "alpha_beta_%.3f_%.3f"%(0,1/2)
sample_sizes_BR, sample_times_BR = concatenate_sim(save_path_i_BR)
# Warning sample times were computed for a diffusion coefficient equal to 1 we need to rescale the time

D = 1e-2

sample_times_BR = sample_times_BR

n_samples,_ = sample_times_BR.shape
n_times = 300
time_max = 1

times = np.linspace(0,time_max,n_times)
indices_times = state(sample_times_BR,times)

list_of_k = list(range(0,15,2))

plt.figure(dpi = 200)

for k in list_of_k:
    prob_fun = []
    for ind_t in range(n_times):
        proba = len(np.where(indices_times[:,ind_t] == k)[0])/ n_samples
        prob_fun.append(proba)
    plt.plot(times, prob_fun, label = 'k = '+str(k))

plt.legend()
plt.savefig("figure6.png")
plt.show()