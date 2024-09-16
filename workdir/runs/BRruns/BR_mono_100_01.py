import numpy as np
import os,sys
file_name =  os.path.splitext(os.path.basename(sys.argv[0]))[0] #file_name is also a parameter that will serve to store results
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('../..')
wd = os.getcwd() # Defines the working directory

#Parameters

n_clusters = 100
Ntmax = np.inf
radius_0 = 0.01 # radius of a cluster that has mass 1/N0
D0 = 1
# Parameters of time step scaling
tol = 1/10

# range of paramters for the radius and diffusion function
alpha_range = [0,1/3,2/3,1]
beta_range = [0,1/2] 
# Initial distribution

init_clusters = np.ones([n_clusters])/n_clusters

# Simulation parameters :
n_samples = 8000