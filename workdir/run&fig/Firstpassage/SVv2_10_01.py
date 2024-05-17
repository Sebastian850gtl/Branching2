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
from Brownian import Modelv2 as Model


runtag = sys.argv[1]  # Simulation tag
samples = sys.argv[2] # Number of samples used only if runvar == 1

np.random.seed(runtag)
n_sample = samples
#Files locations

save_path = '../../results/'+file_name+'/tmp/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path = save_path +runtag

# Parameters
radius = 0.01 # rayon d'un cluster de taille 1
sigma = 1
Ntmax = np.inf

tol = 1e-4
sigmaf = lambda x : sigma # diffusion d'un clu
radiusf = lambda x : radius

n_clusters_range = [2,5,10,20,50]
# Simulation

for i,n_clusters in enumerate(n_clusters_range):

    print(n_clusters)
    monodisperse = np.ones([n_clusters])/n_clusters

    M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)

    save_name = save_path + '_' + str(i) 
    M.run(Ntmax = Ntmax,tol = tol,
                n_samples = n_sample,save_name = save_name,stop = n_clusters-1,size_init = monodisperse)
