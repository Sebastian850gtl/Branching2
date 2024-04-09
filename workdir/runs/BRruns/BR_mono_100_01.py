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
#Parameters

n_clusters = 100
Ntmax = np.inf
radius_0 = 0.01 # radius of a cluster that has mass 1/N0
D0 = 1

# Parameters of time step scaling
tol = 1e-4

# range of paramters for the radius and diffusion function
alpha_range = [0,1/3,2/3,1]
beta_range = [0,1/3,1/2,1] 
# Initial distribution

monodisperse = np.ones([n_clusters])/n_clusters
# Simulation
for i,alpha in enumerate(alpha_range):
    for j,beta in enumerate(beta_range):
        print(alpha,beta)
        radiusf = lambda x : radius_0 * (n_clusters*x)**(beta)
        sigmaf = lambda x : np.sqrt(2*D0*n_clusters*x**(-alpha))
        
        M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)

        save_name = save_path + '_' + str(i) + '_' +str(j)
        M.run(Ntmax = Ntmax,tol = 1e-4,
                n_samples = n_sample,save_name = save_name,stop = 1,size_init = monodisperse)