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

runtag = sys.argv[1]  # Simulation tag
samples = int(sys.argv[2]) # Number of samples used only if runvar == 1

np.random.seed(int(runtag))
n_sample = samples
#Files locations

save_path = '../../results/'+file_name+'/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

fig_path = '../../results/fig/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

save_path = save_path
#Parameters
# ML run with parameters that relate to the Brownina run

n_clusters = 40


alpha_range = [0,1/3,2/3,1]
beta_range = [0,1/3,1/2,1]
# Initial distribution

monodisperse = np.ones([n_clusters])/n_clusters
# Simulation

for i,alpha in enumerate(alpha_range):
    print("Running ML")

    kernel = lambda x,y : (1/x**alpha + 1/y**alpha) 
    M = Coalescence(n_clusters = n_clusters,kernel = kernel)
    save_path_i_ML = save_path + 'ML_' + str(i) + '/tmp'
    if not os.path.exists(save_path_i_ML):
        os.makedirs(save_path_i_ML)
    save_name = save_path_i_ML + "/simtag_" +runtag 
    M.run(n_samples = n_sample, init = monodisperse, save_name = save_name)