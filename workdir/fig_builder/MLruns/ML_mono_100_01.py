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
samples = sys.argv[2] # Number of samples used only if runvar == 1

np.random.seed(runtag)
n_sample = samples
#Files locations

save_path = '../../results/'+file_name+'/tmp/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path = save_path +runtag
#Parameters
# ML run with parameters that relate to the Brownina run

n_clusters = 100

R = 1 # Complete radius of the sphere
radius_0 = 0.01 # radius of a cluster that has mass 1/N0
D0 = 1 # Diffusion coefficient that has mass 1/N0

alpha_range = [0,1/3,2/3,1]
beta_range = [0,1/3,1/2,1]
# Initial distribution

monodisperse = np.ones([n_clusters])/n_clusters
# Simulation

for i,alpha in enumerate(alpha_range):
    for j,beta in enumerate(beta_range):
        print(alpha,beta)
        
        kernel = lambda x,y : (1/x**alpha + 1/y**alpha) * 1/(-R**2*np.log(radius_0/R*((n_clusters*x)**beta+(n_clusters*y)**beta)) + R**2)
        M = Coalescence(n_clusters = n_clusters,kernel = kernel)

        save_name = save_path + '_' + str(i) + '_' +str(j)
        M.run(n_samples = n_sample, init = monodisperse, save_name = save_name)
