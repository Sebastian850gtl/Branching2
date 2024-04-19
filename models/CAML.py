import numpy as np



class CAML:
    def __init__(self,n_clusters,kernel):
        self.n_clusters = n_clusters
        self.kernel = np.vectorize(kernel)
        
        self.active_nodes = None
        self.current_sizes = None
        self.current_times = None
        
        self.sample_sizes = None
        self.sample_times = None
        
    def update(self):
        array = self.current_sizes[self.active_nodes]
        kernels,triu_indices = apply_to_couples(array,self.kernel)
        sum_kernels = np.sum(kernels)
        probabilities = kernels/sum_kernels
        
        self.current_times = self.current_times + np.random.exponential(1/sum_kernels)
        
        len_array = triu_indices[0].size
        ind_couple = np.random.choice(len_array,p = probabilities)
        i, j = triu_indices[0][ind_couple], triu_indices[1][ind_couple]
        
        array[i] = array[i] + array[j]
        array[j] = 0
        self.current_sizes[self.active_nodes] = array
        self.active_nodes.pop(j)


    def run(self,n_samples,init = None,save_name = None):
        
        self.sample_sizes = np.zeros([n_samples,self.n_clusters,self.n_clusters])
        self.sample_times = np.zeros([n_samples,self.n_clusters])
        
        t0 = time()
        for idi in range(n_samples):
            try: 
                init.shape
                self.current_sizes = init.copy()
            except:
                 self.current_sizes = np.ones([self.n_clusters])
            self.active_nodes = list(range(self.n_clusters))
            self.current_times = 0
            
            self.sample_sizes[idi,0,:] = self.current_sizes
            self.sample_times[idi,0] = self.current_times
            for k in range(self.n_clusters-1):
                self.update()
                self.sample_sizes[idi,k+1,:] = self.current_sizes
                self.sample_times[idi,k+1] = self.current_times

            print('\r',   'Advancement : %.1f'%(((idi+1)/n_samples)*100)+' %', 'done in %.2fs.' % (time() - t0),end='')
        print("End")
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