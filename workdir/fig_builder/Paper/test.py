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

save_path = '../../results/figure8_1/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


from compute_probas import probs,state
from concatenator import concatenate_sim
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm, gamma
from scipy.special import gamma as Gamma


T = 6
D = 1
tmax = 10 * D
time_range_BR = np.linspace(0,tmax,500)
N = 13


BR_N_folder = 'BR_mono_nc'+str(N)+'_r001_eps05_no_int'
save_path_BR = '../../results/' + BR_N_folder + '/'

save_path_i_BR = save_path_BR + "alpha_beta_%.3f_%.3f"%(0,1/2)

sample_sizes_BR, sample_times_BR = concatenate_sim(save_path_i_BR)

probas = probs(sample_sizes_BR,sample_times_BR,time_range_BR,2,T/N)

proba_function = np.array([time_range_BR / D, probas])


BR_N_folder_2 = 'BR_mono_nc'+str(N)+'_r001_eps05_int'
save_path_BR2 = '../../results/' + BR_N_folder_2 + '/'

save_path_i_BR2 = save_path_BR2 + "alpha_beta_%.3f_%.3f"%(0,1/2)

sample_sizes_BR2, sample_times_BR2 = concatenate_sim(save_path_i_BR2)

probas2 = probs(sample_sizes_BR2,sample_times_BR2,time_range_BR,2,T)

#print(probas2)
#print(np.mean(sample_times_BR2[:,-1]))
print("Nombre d'erreurs",400 * np.sum(np.abs(probas2 - probas)))

proba_function_int = np.array([time_range_BR / D, probas2])

plt.figure(dpi = 200)

plt.plot( time_range_BR, probas, label = "N = "+str(N))
plt.plot(  time_range_BR , probas2, label = "N = "+str(N))
plt.legend()
plt.savefig(fig_path + "figure8_1.png")
plt.show()


