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
from CAML import CAML

parameters_file = sys.argv[1]
runtag = sys.argv[2]  # Simulation tag

np.random.seed(int(runtag))
#Files locations

save_path = '../../results/'+parameters_file+'/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

fig_path = '../../results/fig/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

save_path = save_path
#Parameters

param_module = importlib.import_module(parameters_file)
n_samples = param_module.n_samples
n_clusters = param_module.n_clusters
alpha_range =  param_module.alpha_range
# Initial distribution 
init_clusters = param_module.init_clusters

# Simulation
for i,alpha in enumerate(alpha_range):
    M2 = CAML(n_clusters = n_clusters,alpha = alpha)
    save_path_i_CAML = save_path + 'CAML_' + str(i) + '/tmp'
    if not os.path.exists(save_path_i_CAML):
        os.makedirs(save_path_i_CAML)
    save_name = save_path_i_CAML + "/simtag_" +runtag 
    M2.run(n_samples = n_samples, init = init_clusters, save_name = save_name)