# Plotting the 3D plot of the branching probability
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

radius = 0.001
alpha_range = [0,1/3, 2/3, 1]
#alpha_range = [0]
k = 2

BR_folder = 'BR_mono_nc100_r001_eps05'
SC_folder = 'SC_mono_nc100'
CASC_folder = 'CASC_nc100'

save_path_BR = '../../results/' + BR_folder + '/'
save_path_SC = '../../results/' + SC_folder + '/'
save_path_CASC = '../../results/' + CASC_folder + '/'

from compute_probas import probs,prob_fun
from concatenator import concatenate_sim
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
from scipy.special import gamma as Gamma
from time import time


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
Nx, Nt = 200, 300 # Used only if use_saved_branching_proba = 0
small_time_Nt = 50

xmin = 0.01
masses = np.linspace(xmin,0.5,Nx) # Used only if use_saved_branching_proba = 0
d = 1
k = 2
for i,alpha in enumerate(alpha_range):
    # Browninan
    save_path_i_BR = save_path_BR + "alpha_beta_%.3f_%.3f"%(alpha,0)
    save_file = save_path_i_BR + "_pb_3d.npy"
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

        max_time = np.mean(sample_times_BR[:,-1])*2
        time_range_BR_small_time = np.linspace(0,max_time/10, small_time_Nt) 
        time_range_BR_large = np.linspace(max_time/10,max_time,Nt)

        time_range_BR = np.concatenate((time_range_BR_small_time,time_range_BR_large))
        print(time_range_BR.shape)
        
        time_range =  time_range_BR *(n_clusters)**(-alpha) *1/(-np.log(2*radius) + np.log(2)) #To be able to compare BR with SC and CASC we divide be the costant in front of the kernel.
        print("Computing branching probability")
        t0 = time()

        branching_proba_BR = prob_fun(sample_sizes_BR,sample_times_BR,time_range_BR,masses,k)
        print("...done in %.3f s"%(time()- t0))

        np.save(save_file, branching_proba_BR)
        np.save(save_times,time_range)
    # SC
    save_path_i_SC = save_path_SC + 'alpha_%.3f'%(alpha)
    save_file = save_path_i_SC + "pb_3d.npy"
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
    save_file = save_path_i_CASC + "_pb_3d.npy"
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


    from matplotlib.colors import LightSource
    from matplotlib.cm import ScalarMappable
    #from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator


    print(f"branching_proba_BR shape: {branching_proba_BR.shape}")
    print(f"time_range: {len(time_range)}, masses: {len(masses)}")


    # Single proba branching 

    M, T = np.meshgrid(masses, time_range)

    
    fun_to_plot = branching_proba_BR

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    # Define the colormap and normalization
    cmap = plt.cm.inferno
    #norm = Normalize(vmin=np.nanmin(fun_to_plot), vmax=np.nanmax(fun_to_plot))


    ls = LightSource(azdeg= -90, altdeg = 10)  # lower altdeg = darker shading
    rgb = ls.shade(fun_to_plot.T, cmap=cmap, vert_exag = 1, blend_mode='soft')

    # Plot the shaded surface
    surf = ax.plot_surface(M, T, fun_to_plot.T, facecolors=rgb,
                        linewidth=0, antialiased=True, alpha = 1)
    
    # surf = ax.plot_surface(M, T, fun_to_plot.T, cmap= cmap,
    #                     linewidth=0, antialiased=True, alpha = 0.9)
    

    # Labels
    ax.set_xlabel(r'Mass $x$', labelpad = 5)
    ax.set_ylabel(r'Time $t$', labelpad = 5)
    ax.set_zlabel('Branching Probability', labelpad = 5)

    # Suppose ax is your 3D axis
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))  # at most 4 ticks on x-axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))  # at most 4 ticks on y-axis
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))  # at most 4 ticks on z-axis

    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(30)  # adjust angle as needed

    # for tick in ax.get_yticklabels():
    #     tick.set_rotation(-45)

    # ax.set_title('Branching Probability as a Function of Mass and Time')

#    # Colorbar consistent with data (not RGB facecolors)
#     mappable = ScalarMappable(norm=norm, cmap=cmap)
#     mappable.set_array([])
#     fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, label='Probability')

    # Optional tweaks for better perspective
    ax.view_init(elev = 32, azim = -8)  # change viewing angle
    ax.dist = 10                      # camera distance

    plt.tight_layout()
    fig.savefig(fig_path + file_name + '%.3f.png'%(alpha), dpi=200)#, bbox_inches='tight')
    # Figure comparing both
    time_range = time_range[1:]

    M, T = np.meshgrid(masses, time_range)

    fun_to_plot2 = analytic(M,T,alpha).T
    
    fun_to_plot = branching_proba_BR[:,1:]


    time_range = time_range

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    # Define the colormap and normalization
    cmap = plt.cm.inferno
    #norm = Normalize(vmin=np.nanmin(fun_to_plot), vmax=np.nanmax(fun_to_plot))


    #ls = LightSource(azdeg= 300, altdeg = 80)  # lower altdeg = darker shading
    #rgb = ls.shade(fun_to_plot.T, cmap=cmap, vert_exag = 0.9, blend_mode='soft')

    # Plot the shaded surface
    # surf = ax.plot_surface(M, T, fun_to_plot.T, facecolors=rgb,
    #                     linewidth=0, antialiased=True)
    surf = ax.plot_surface(M, T, fun_to_plot.T,
                        linewidth=0, antialiased=True)
    

    # Colormap and lighting for second surface
    cmap2 = plt.cm.viridis
    ls2 = LightSource(azdeg= 300, altdeg = 80)  # lower altdeg = darker shading
    rgb2 = ls2.shade(fun_to_plot2.T, cmap=cmap2, vert_exag=0.8, blend_mode='soft')
    surf2 = ax.plot_surface(M, T, fun_to_plot2.T, facecolors=rgb2, linewidth=0, antialiased=True, alpha=0.7)

    # Labels
    ax.set_xlabel('Mass', labelpad = 5)
    ax.set_ylabel('Time', labelpad = 5)
    ax.set_zlabel('Branching Probability', labelpad = 5)

    # Suppose ax is your 3D axis
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))  # at most 4 ticks on x-axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))  # at most 4 ticks on y-axis
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))  # at most 4 ticks on z-axis

    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(30)  # adjust angle as needed

    # for tick in ax.get_yticklabels():
    #     tick.set_rotation(-45)

    # ax.set_title('Branching Probability as a Function of Mass and Time')

#    # Colorbar consistent with data (not RGB facecolors)
#     mappable = ScalarMappable(norm=norm, cmap=cmap)
#     mappable.set_array([])
#     fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, label='Probability')

    # Optional tweaks for better perspective
    ax.view_init(elev = 20, azim = 40)  # change viewing angle
    ax.dist = 10                      # camera distance

    plt.tight_layout()
    fig.savefig(fig_path + file_name + '%.3fcompare.png'%(alpha), dpi=200)#, bbox_inches='tight')
#plt.show()