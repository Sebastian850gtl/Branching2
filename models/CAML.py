import numpy as np
from scipy.special import gamma
from time import time
#%% core functions


def lambda_alpha(n_clusters,k,alpha,assymp = 20):
    p1 = k
    var2 = k*(1 + alpha)
    if var2 < assymp:
        p2 = gamma(var2)/gamma(var2 - 1 - alpha)
    else:
        p2 = (var2 -1)**(1 + alpha) 
    var3 = n_clusters + k*alpha
    if var3 < assymp:
        p3 = n_clusters**alpha * gamma(var3 - alpha)/gamma(var3)
    else:
        p3 = (n_clusters/(var3))**alpha
    return 1/2 * p1 * p2 * p3

def compute_lambdas(n_clusters,alpha):
    res = np.zeros([n_clusters-1])
    for k in range(n_clusters,1,-1):
            res[n_clusters - k] = lambda_alpha(n_clusters,k,alpha,assymp = 20)
    return res

class CAML:
    def __init__(self,n_clusters,alpha):
        """ alpha is the parameter in the kernel $K_\alpha$"""
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        self.active_nodes = None
        self.current_sizes = None
        self.current_times = None
        
        self.sample_sizes = None
        self.sample_times = None
        self.lambdas = compute_lambdas(n_clusters, alpha)

    def run(self,n_samples,init = None,save_name = None):
        """  Runs the model
        Arguments:
            n_samples : Number  of sammples
            save__name : File name where to save results 
        Results saved as npy file:
            _times : numpy ndarray shape = (n_samples,n_clusters) ; For each sample gives all the collision times, the first column is zeros since there is n_clusters-1 collisions.
            _sizes : numpy ndarray shape = (n_samples,n_clusters,n_clusters); For T =  _times[i,j] the time at sample  i and collision j, _sizes[i,j,:]  is the cluster distribution at sample  i and collision number j.
        """
        self.sample_sizes = np.zeros([n_samples,self.n_clusters,self.n_clusters])
        self.sample_times = np.zeros([n_samples,self.n_clusters])
        
        t0 = time()

        sample_single_times = np.random.exponential(np.tile(1/self.lambdas,(n_samples,1))).reshape(n_samples,self.n_clusters -1)
        sample_times = np.cumsum(sample_single_times, axis = 1)
        sample_times = np.concatenate((np.zeros([n_samples,1]),sample_times),axis = 1)
        #print(sample_times.shape)

        for k in range(self.n_clusters,0,-1):
            array_alpha = (1 + self.alpha) * np.ones([k])
            sample_sizes_k = np.random.dirichlet(array_alpha,size = (n_samples))
            # if k < self.n_clusters:
            #     print(sample_sizes_k.shape)
            #     print(np.zeros([n_samples,self.n_clusters - k]).shape)
            sample_sizes_k = np.concatenate((sample_sizes_k,np.zeros([n_samples,self.n_clusters - k])),axis = 1)
            self.sample_sizes[:,self.n_clusters - k,:] = sample_sizes_k


        #self.sample_sizes = sample_sizes
        self.sample_times = sample_times

        print('done in %.2fs.' % (time() - t0))
        #print("End")
        print("Saving samples")
        if save_name:
            try:
                previous_save_sizes = np.load(save_name+'_sizes.npy')
                previous_save_times = np.load(save_name+'_times.npy')
                np.save(save_name+'_sizes.npy',np.concatenate((self.sample_sizes,previous_save_sizes),axis = 0))
                np.save(save_name+'_times.npy',np.concatenate((self.sample_times,previous_save_times),axis = 0))
            except:
                np.save(save_name+'_sizes.npy',self.sample_sizes)
                np.save(save_name+'_times.npy',self.sample_times)
        else:
            pass