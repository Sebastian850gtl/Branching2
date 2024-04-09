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
print(" Simulation of "+str(file_name)+", simulation tag : "+str(runtag))
print(" Note that the tag serves also as a seed")
np.random.seed(int(runtag))
n_sample = int(samples)
#Files locations

save_path = '../../results/'+file_name+'/tmp/'

if not os.path.exists(save_path):
    os.makedirs(save_path)


#Parameters
radius = 0.001 # rayon d'un cluster de taille 1
sigma1 = 1
n_clusters = 2
Ntmax = np.inf

radiusf = lambda x : radius

#M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)
tol = 1e-4
diffusion_range = np.arange(11)/2
# Simulation

for i,D2 in enumerate(diffusion_range):
    sigma2 = np.sqrt(2*D2)
    sigmaf = lambda x : sigma1*(x<= 1) + sigma2*(x > 1)
    M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)

    save_path_i = save_path +"diffusion_"+ str(i)
    if not os.path.exists(save_path_i):
        os.makedirs(save_path_i)
    save_name = save_path_i +"/simtag_" +runtag 
    M.run(Ntmax = Ntmax,tol = tol,
                n_samples = n_sample,save_name = save_name,stop = 1,size_init = np.array([2,1]))