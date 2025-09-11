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
from scipy.stats import norm
from Brownian import Modelv3 as Model
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
import matplotlib.pyplot as plt

runtag = sys.argv[1]  # Simulation tag
#runtag =  str(np.random.randint(int(1e7)))

print(" Seed =",runtag)

print(" Simulation of "+str(file_name)+", simulation tag : "+str(runtag))
print(" Note that the tag serves also as a seed")
np.random.seed(int(runtag))
n_sample = 1

#%% Parameters
r = 0.01 # rayon d'un cluster de taille 1
sigma1 = 1
sigma2 = 1
n_clusters = 2
Ntmax = np.inf
tol = 5e-2

sigmaf = lambda x : sigma1*(x<= 1) + sigma2*(x>=2)
radiusf = lambda x : r

M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)


# simulation
M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)
trajectories = M.run(Ntmax = Ntmax,tol = tol,n_samples = n_sample,save_name = None,
        stop = 1,position_init = None, mass_init = np.array([2,1]),save_trajectories = 1)

# Plots
# Simus with different Rslow
R = 1
Ttheoric = (-np.log(2*r/R) +np.log(2))* R**2/(sigma1**2/2 + sigma2**2/2)




# Create figure and 3D axis
fig = plt.figure(dpi = 200)
ax = fig.add_subplot(111, projection='3d')
# Hide grid lines
ax.grid(False)

# Hide axes ticks
ax.axis('off')
# Scatter plot of the coordinates
N_points,number_of_trajectories,_ = trajectories.shape

# # Compute global limits for all trajectories
# x_min, x_max = np.min(trajectories[:, :, 0]), np.max(trajectories[:, :, 0])
# y_min, y_max = np.min(trajectories[:, :, 1]), np.max(trajectories[:, :, 1])
# z_min, z_max = np.min(trajectories[:, :, 2]), np.max(trajectories[:, :, 2])

# # Set the same range for all axes (to avoid distortion)
# max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
# mid_x = (x_max + x_min) / 2
# mid_y = (y_max + y_min) / 2
# mid_z = (z_max + z_min) / 2

# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, mid_y + max_range)
# ax.set_zlim(mid_z - max_range, mid_z + max_range) 

print(N_points,number_of_trajectories)
for i in range(number_of_trajectories):
    up_from = 500
    up_to = 1
    points = trajectories[:,i,:]
    ax.plot(points[-up_from:-up_to, 0], points[-up_from:-up_to, 1], points[-up_from:-up_to, 2],linewidth = 0.9)

# Generate hemisphere surface
u = np.linspace(0, 2 * np.pi, 50)   # azimuth
v = np.linspace(0, np.pi/2, 50)     # polar (0 to pi/2 for hemisphere)
u, v = np.meshgrid(u, v)

# Parametric equations for the sphere of radius 1
x = np.cos(u) * np.sin(v) * R
y = np.sin(u) * np.sin(v) * R
z = np.cos(v) * R

ax.set_box_aspect((np.ptp([-1,1]), np.ptp([-1,1]), np.ptp([0,1])))

# Plot surface
ax.plot_surface(x, y, z, color='b', alpha=0.05, rstride=1, cstride=1, linewidth=0)

# Formatting
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#ax.set_title("3D Points with Hemisphere Surface")
ax.legend()

save_path = '../../results/fig/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

plt.savefig(save_path + file_name + '_' + runtag + '.png')
plt.show()