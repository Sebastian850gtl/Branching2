import numpy as np
n_clusters = 300
alpha_range = [0,1/3,2/3,1]
# Initial distribution 
monodisperse = np.ones([n_clusters])/n_clusters
init_clusters = lambda alpha : monodisperse

# Simulation
n_samples = 5000
n_runs = 4