import numpy as np
n_clusters = 100
alpha_range = [0,1/3,2/3,1]
# Initial distribution 
init_clusters = np.ones([n_clusters])/n_clusters

# Simulation
n_samples = 20000
n_runs = 1