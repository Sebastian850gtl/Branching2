import numpy as np

#Parameters

n_clusters = 100
Ntmax = np.inf
radius_0 = 0.01 # radius of a cluster that has mass 1/N0
D0 = 1
# Parameters of time step scaling
tol = 0.05

# range of paramters for the radius and diffusion function
alpha_range = [0,1/3,2/3,1]
beta_range = [0,1/2] 
# Initial distribution
monodisperse = np.ones([n_clusters])/n_clusters
init_clusters = lambda alpha : monodisperse

# Simulation parameters :
n_samples = 2000
n_runs = 10