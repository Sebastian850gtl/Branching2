import numpy as np

n_clusters = 100
alpha_range = [0,1/3,2/3,1]
# Initial distribution monodisperse
monodisperse = np.ones([n_clusters])/n_clusters
init_clusters = lambda alpha : monodisperse

# Simulation
n_samples = 2000
n_runs = 5