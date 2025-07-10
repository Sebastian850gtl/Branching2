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

parameters_file_name = sys.argv[1]
runtag = sys.argv[2]  # Simulation tag

np.random.seed(int(runtag))
#Files locations

save_path = '../../results/'+parameters_file_name+'/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

#Parameters
param_module = importlib.import_module(parameters_file_name)
n_samples = param_module.n_samples
n_clusters = param_module.n_clusters
alpha_range =  param_module.alpha_range
# Initial distribution 
init_clusters_fun = param_module.init_clusters

print("yes")
for i,alpha in enumerate(alpha_range):
    init_clusters = init_clusters_fun(alpha)
    print(" Running Stochastic Coalescence")
    kernel = lambda x,y : 1/x**alpha + 1/y**alpha
    M1 = Coalescence(n_clusters = n_clusters,kernel = kernel)
    save_path_i_ML = save_path + save_path + 'alpha_%.3f/tmp'%(alpha)
    if not os.path.exists(save_path_i_ML):
        os.makedirs(save_path_i_ML)
    save_name = save_path_i_ML + "/simtag_" +runtag 
    M1.run(n_samples = n_samples, init = init_clusters, save_name = save_name)