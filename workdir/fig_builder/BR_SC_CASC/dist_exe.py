import os,sys
import importlib
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
from CAML import CAML

#Files locations


fig_path = '../../results/fig/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

parameters_file_name = sys.argv[1]
# parameters_file_name = "mono_nc100_r001_eps05"
# #Parameters

param_module = importlib.import_module(parameters_file_name)

radius = param_module.radius
alpha_range = param_module.alpha_range
k = param_module.k
xmin = param_module.x
Nx = param_module.Nx
Nt = param_module.Nt


BR_folder = param_module.BR_folder
SC_folder = param_module.SC_folder
CASC_folder = param_module.CASC_folder

# radius = 0.001
# alpha_range = [0,1/3,2/3]
# k = 2

# BR_folder = 'BR_mono_nc100_r001_eps05'
# SC_folder = 'SC_mono_nc100'
# CASC_folder = 'CASC_nc100'

save_path_BR = '../../results/' + BR_folder + '/'
save_path_SC = '../../results/' + SC_folder + '/'
save_path_CASC = '../../results/' + CASC_folder + '/'

from compute_probas import probs,prob_fun
from concatenator import concatenate_sim
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
from scipy.special import gamma as Gamma
from time import time

def dist(A,B,d):
    """ 
    A and B are 2-dimensional matrices corresponding to the probability functions first dimension is the mass discretization and second time
    d is the order of the distance on the mass dimension
    """

    Nx,Nt = A.shape # len(times) = Nt 
    res = np.zeros([Nt])
    D1 = np.abs(A-B)**d
    D2 = 1/Nx*np.sum(D1, axis = 0)**(1/d)
    maxi = 0
    for nt in range(Nt):
        maxi = max(maxi, D2[nt])
        res[nt] = maxi
    return res

def dist_inf(A,B):
    """ 
    A and B are 2-dimensional matrices corresponding to the probability functions first dimension is the mass discretization and second time
    d is the order of the distance on the mass dimension
    """

    Nx,Nt = A.shape # len(times) = Nt 
    D1 = np.abs(A-B)
    return np.max(D1)


C1 = lambda alpha : Gamma(1 + alpha)**(1/(1+alpha))/(1 + alpha)
C2 = lambda alpha : np.sqrt(C1(alpha)/(3 + 2*alpha))

p = lambda x,alpha : 1 - gamma.cdf(x , 1 + alpha, scale = (1+alpha)**(-1))
q = lambda x,t,alpha : p(x*(C1(alpha)*t**(-1/(1+alpha))),alpha)
f1 = lambda x,t,alpha : (1 - q(x,t,alpha))**((C1(alpha) + C2(alpha)**2/2*np.log(1 - q(x,t,alpha)))*t**(-1/(1+alpha))-1)

f2 = lambda x,t,alpha : (C1(alpha) + C2(alpha)**2 * np.log(1 - q(x,t,alpha)))*q(x,t,alpha)*t**(-1/(1+alpha)) - q(x,t,alpha) + 1

f3 = lambda x,t,alpha : (k-1 - C1(alpha)*t**(-1/(1+alpha)))/(C2(alpha)*t**(-1/(2+2*alpha))) - C2(alpha)*t**(-1/(2+2*alpha))*np.log(1- q(x,t,alpha))

h = lambda t,alpha : (k-1 - C1(alpha)*t**(-1/(1+alpha)))/(C2(alpha)*t**(-1/(2+2*alpha)))

f4 = lambda x,t,alpha : 1 - norm.cdf(f3(x,t,alpha))

f5 = lambda x,t,alpha : 1/np.sqrt(2*np.pi)*q(x,t,alpha)*C2(alpha) *t**(-1/(2 + 2*alpha))*np.exp(-f3(x,t,alpha)**2/2)

f = lambda x,t,alpha :  1 - norm.cdf(h(t,alpha)) - f1(x,t,alpha)*( f2(x,t,alpha)*f4(x,t,alpha) + f5(x,t,alpha))
analytic = lambda x,t,alpha : f(x,t,alpha)



use_saved_branching_proba = 1
#Nx, Nt = 100, 200 # Used only if use_saved_branching_proba = 0
masses = np.linspace(xmin,1,Nx) # Used only if use_saved_branching_proba = 0
d = 1
k = 2
for i,alpha in enumerate(alpha_range):
    # Browninan
    save_path_i_BR = save_path_BR + "alpha_beta_%.3f_%.3f"%(alpha,0)
    save_file = save_path_i_BR + "_pb.npy"
    save_times = save_path_i_BR + "_times_range.npy"
    if use_saved_branching_proba and os.path.exists(save_file) and os.path.exists(save_times):
        print("Loading saved branching probability")
        t0 = time()
        branching_proba_BR = np.load(save_file)
        time_range = np.load(save_times)
        print("...done in %.3f s"%(time()- t0))
    else:
        print("Loading Brownian samples \n ...parameters: alpha = %.3f, beta = %.3f"%(alpha,0))
        t0 = time()

        sample_sizes_BR, sample_times_BR = concatenate_sim(save_path_i_BR)

        n_samples,n_clusters = sample_times_BR.shape
        print("...done in %.3f s"%(time()- t0))

        time_range_BR = np.linspace(0,np.mean(sample_times_BR[:,-1])*3,Nt)
        
        time_range =  time_range_BR *(n_clusters)**(-alpha) *1/(-np.log(2*radius) + np.log(2)) #To be able to compare BR with SC and CASC we divide be the costant in front of the kernel.
        print("Computing branching probability")
        t0 = time()

        branching_proba_BR = prob_fun(sample_sizes_BR,sample_times_BR,time_range_BR,masses,k)
        print("...done in %.3f s"%(time()- t0))

        np.save(save_file, branching_proba_BR)
        np.save(save_times,time_range)
    # SC
    save_path_i_SC = save_path_SC + 'alpha_%.3f'%(alpha)
    save_file = save_path_i_SC + "_pb.npy"
    if use_saved_branching_proba and os.path.exists(save_file):
        print("Loading saved branching probability")
        t0 = time()
        branching_proba_SC = np.load(save_file)
        print("...done in %.3f s"%(time()- t0))
    else:
        print("Loading SC samples \n ...parameters: alpha = %.3f"%(alpha))
        t0 = time()
        sample_sizes_SC, sample_times_SC = concatenate_sim(save_path_i_SC)
        print("...done in %.3f s"%(time()- t0))

        print("Computing branching probability")
        t0 = time()
        branching_proba_SC = prob_fun(sample_sizes_SC,sample_times_SC,time_range,masses,k)
        print("...done in %.3f s"%(time()- t0))

        np.save(save_file, branching_proba_SC)

    # CASC
    save_path_i_CASC = save_path_CASC + 'alpha_%.3f'%(alpha)
    save_file = save_path_i_CASC + "_pb.npy"
    if use_saved_branching_proba and os.path.exists(save_file):
        print("Loading saved branching probability")
        t0 = time()
        branching_proba_CASC = np.load(save_file)
        print("...done in %.3f s"%(time()- t0))
    else:
        print("Loading CASC samples \n ...parameters: alpha = %.3f"%(alpha))
        t0 = time()
        sample_sizes_CASC, sample_times_CASC = concatenate_sim(save_path_i_CASC)
        print("...done in %.3f s"%(time()- t0))
        
        print("Computing branching probability")
        t0 = time()
        branching_proba_CASC = prob_fun(sample_sizes_CASC,sample_times_CASC,time_range,masses,k)
        print("...done in %.3f s"%(time()- t0))
        np.save(save_file, branching_proba_CASC)
    # #Plot
    # plt.figure(dpi = 300)
    # plt.plot(time_range,branching_proba_BR[28,:])
    # plt.plot(time_range,branching_proba_SC[28,:])
    # plt.plot(time_range,branching_proba_CASC[28,:])
    # plt.savefig(fig_path+parameters_file_name+'_%.3f_plot.png'%(alpha))
    # Distances
    # BR-SC
    time_range = time_range[:Nt]
    print("Computing distances between BR and SC")
    t0 = time()
    dists = dist(branching_proba_BR,branching_proba_SC,d)
    dists_inf = dist_inf(branching_proba_BR,branching_proba_SC)


    print("...done in %.3f s"%(time()- t0))
    print("Distance  1 for alpha = %.3f, radius = %.3f: %.6f"%(alpha,radius,dists[-1]))
    print("Distance sup for alpha = %.3f, radius = %.3f: %.6f"%(alpha,radius,dists_inf))

    #plt.savefig(fig_path+parameters_file_name+"BR_SC"+'_%.3f_dist.png'%(alpha))

    # SC-CASC
    #plt.figure(dpi = 300)
    print("Computing distances between SC and CASC")
    t0 = time()
    dists = dist(branching_proba_CASC,branching_proba_SC,d)
    dists_inf = dist_inf(branching_proba_CASC,branching_proba_SC)
    #plt.plot(time_range, dists[:Nt], label = r'distance $d(p_{SC},p_{CASC})$')

    print("...done in %.3f s"%(time()- t0))

    print("Distance  1 for alpha = %.3f, radius = %.3f: %.6f"%(alpha,radius,dists[-1]))
    print("Distance sup for alpha = %.3f, radius = %.3f: %.6f"%(alpha,radius,dists_inf))


    # BR-CASC
    #plt.figure(dpi = 300)
    print("Computing distances between BR and CASC")
    t0 = time()
    dists = dist(branching_proba_BR,branching_proba_CASC,d)
    dists_inf = dist_inf(branching_proba_BR,branching_proba_CASC)

    #plt.plot(time_range, dists[:Nt], label = r'distance $d(p_{BR},p_{CASC})$')
    print("...done in %.3f s"%(time()- t0))
    #plt.savefig(fig_path+parameters_file_name+"BR_CASC"+'_%.3f_dist.png'%(alpha))
    print("Distance  1 for alpha = %.3f, radius = %.3f: %.6f"%(alpha,radius,dists[-1]))
    print("Distance sup for alpha = %.3f, radius = %.3f: %.6f"%(alpha,radius,dists_inf))

    #BR_Analytic
    time_range_2 = time_range.copy()
    time_range_2[0] = time_range[1]/2
    
    grid_times, grid_mass  = np.meshgrid(time_range_2,masses)

    #print(grid_mass.shape)
    branching_proba_Analytic = analytic(grid_mass,grid_times,alpha)
    print("Computing distances between BR and Analytic")
    t0 = time()
    dists = dist(branching_proba_BR,branching_proba_Analytic,d)
    dists_inf = dist_inf(branching_proba_BR,branching_proba_Analytic)
    
    print("...done in %.3f s"%(time()- t0))
    print("Distance  1 for alpha = %.3f, radius = %.3f: %.6f"%(alpha,radius,dists[-1]))
    print("Distance sup for alpha = %.3f, radius = %.3f: %.6f"%(alpha,radius,dists_inf))

    #plt.legend()
    #plt.savefig(fig_path+parameters_file_name+'_%.3f_dist.png'%(alpha))
    
    # plt.figure(dpi = 200)
    # plt.title("Comparpar mass alpha = "+str(alpha))
    # plt.plot(masses,branching_proba_BR[:,100])
    # plt.plot(masses,branching_proba_SC[:,100])
    # plt.plot(masses,branching_proba_CASC[:,100])
    
    # plt.figure(dpi = 200)
    # plt.title("Comparpar time alpha = "+str(alpha))
    # plt.plot(time_range,branching_proba_BR[37,:Nt])
    # plt.plot(time_range,branching_proba_SC[37,:Nt])
    # plt.plot(time_range,branching_proba_CASC[37,:Nt])


    # Plotting the 3D plot of the branching probability
    Ntplot = 120
    Nxplot = 40

    from matplotlib.colors import LightSource
    
    M, T = np.meshgrid(masses[:Nxplot], time_range[:Ntplot])
    branching_proba_BR
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
    M,
    T,
    branching_proba_BR.T[:Ntplot, :Nxplot],
    cmap='viridis',           # smooth and perceptually uniform
    linewidth=0,              # no grid lines on surface
    antialiased=True,         # smoother surface
    shade=True,               # enables light shading
    alpha=0.0,               # slight transparency looks nice
    )

    ls = LightSource(azdeg=180, altdeg=20)
    rgb = ls.shade(branching_proba_BR.T[:Ntplot, :Nxplot], cmap=plt.cm.viridis_r, vert_exag=0.5, blend_mode='soft')
    surf = ax.plot_surface(M, T, branching_proba_BR.T[:Ntplot, :Nxplot], facecolors=rgb, linewidth=0, antialiased=True)
    # # Optional: add custom lighting
    # surf.set_facecolor((0, 0, 0, 0))  # makes the background transparent
    # ax.plot_surface(
    #     M,
    #     T,
    #     branching_proba_BR.T[:Ntplot, :Nxplot],
    #     cmap='viridis',
    #     rstride=1, cstride=1,
    #     linewidth=0,
    #     antialiased=True,
    #     shade=True,
    # )

    # Labels
    ax.set_xlabel('Mass', labelpad=10)
    ax.set_ylabel('Time', labelpad=10)
    ax.set_zlabel('Branching Probability', labelpad=10)
    # ax.set_title('Branching Probability as a Function of Mass and Time')

    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label='Probability')

    # Optional tweaks for better perspective
    ax.view_init(elev=30, azim=240)  # change viewing angle
    ax.dist = 10                      # camera distance

    plt.tight_layout()
    plt.show()