import numpy as np

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

# Simulation parameters :
n_samples = 4000
n_runs = 4

# Initial distribution
monodisperse = np.ones([n_clusters])/n_clusters
init_clusters = lambda alpha : np.random.dirichlet((1 + alpha)*np.ones([n_clusters]),size = n_samples)