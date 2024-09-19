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
from Brownian import Modelv3 as Model


parameters_file_name = sys.argv[1] # parameters file
runtag = sys.argv[2]  # Simulation tag

np.random.seed(int(runtag))

#Import parameters

param_module = importlib.import_module(parameters_file_name)
n_samples = param_module.n_samples

#Files locations

save_path = '../../results/'+parameters_file_name+'/'
fig_path = '../../results/fig/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Initial distribution

for i,alpha in enumerate(param_module.alpha_range):
    for j,beta in enumerate(param_module.beta_range):

        size_init = param_module.init_clusters(alpha)
        print(len(size_init.shape))
        radiusf = lambda x : param_module.radius_0 * (param_module.n_clusters*x)**(beta)
        sigmaf = lambda x : np.sqrt(2*param_module.D0*(param_module.n_clusters*x)**(-alpha))
        
        M = Model(n_clusters = param_module.n_clusters,sigmafun = sigmaf,radfun = radiusf)
        save_path_n = save_path +"alpha_beta_%.3f_%.3f/tmp"%(alpha,beta)
        if not os.path.exists(save_path_n):
            os.makedirs(save_path_n)
        save_name = save_path_n +"/simtag_" +runtag
        M.run(Ntmax = param_module.Ntmax,tol = param_module.tol,
                n_samples = n_samples,save_name = save_name,stop = 1,size_init = size_init)
        