import numpy as np
n_clusters = 100
alpha_range = [0,1/3,2/3,1]

# Simulation
n_samples = 4000
n_runs = 5

# Initial distribution dirichlet 1 + alpha
init_clusters = lambda alpha : np.random.dirichlet((1 + alpha)*np.ones([n_clusters]),size = n_samples)