# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np


def apply_to_couples_diffv2(arr,triu_indices):
    """ For arr =[a1,a2,a3] returns the array [a1 - a2, a1 - a3, a2 - a3]"""
    #arr1 = np.apply_along_axis(auxu,axis = 0,arr = arr)[triu_indices]
    #arr2 = np.apply_along_axis(auxl,axis = 0,arr = arr)[triu_indices]
    arr1 = np.tril(arr,k = -1).T[triu_indices]
    arr2 = np.triu(arr,k = 1)[triu_indices]
    return arr1 - arr2

def test_a_in_B(a,B):
    if a > B[-1]:
        return False
    else:
        return a in B
    

    

def colliding_sets(I,J):
    def aux(level_set,I_var,J_var):
        print("level_set",level_set)
        if len(level_set) == 0:
            return level_set
        else:
            new_level_set = []
            ind_in_level_set = 0
            n_var = len(I_var)
            while n_var > 0 and ind_in_level_set < len(level_set):
                i = level_set[ind_in_level_set]
                
                current_ind = 0
                to_remove = []
                while current_ind < n_var and I_var[current_ind] <= i:
                    j_current, i_current = J_var[current_ind], I_var[current_ind]
                    
                    if i_current == i and j_current not in new_level_set:
                        to_remove.append(current_ind)
                        if j_current not in level_set:
                            new_level_set.append(j_current)
                        else:
                            pass
                    elif j_current == i:
                        to_remove.append(current_ind)
                        if i_current not in new_level_set:
                            new_level_set.append(i_current)
                        else:
                            pass
                    else:
                        pass
                    current_ind += 1
                    print("new",new_level_set)
                    print("to_remove",to_remove)
                for removed_count, ind_to_remove in enumerate(to_remove):
                    I_var.pop(ind_to_remove - removed_count)
                    J_var.pop(ind_to_remove - removed_count)
                    n_var += -1
                print("I_var",I_var)
                print("J_var",J_var)
                ind_in_level_set += 1
            return level_set + aux(new_level_set,I_var,J_var)
                        
    print("I",I)
    print("J",J)      
    I_var, J_var = list(I), list(J)
    n_var = len(I_var)
    sets = []
    while n_var > 0:
        total_set = aux([I_var[0]],I_var,J_var)
        sets.append(total_set)
        n_var = len(I_var)
    return sets
            
            
            

if __name__ =="__main__":

    n_clusters = 50
    
    arr = np.zeros([n_clusters])
    
    arr[2:] = np.arange(n_clusters -2) + 3
    
    triu_indices = np.triu_indices(n_clusters,k=1)
    
    diff = apply_to_couples_diffv2(arr,triu_indices)
    contact_indices_glob = np.where(np.abs(diff) < 1.5)
    
    I,J = triu_indices[0][contact_indices_glob], triu_indices[1][contact_indices_glob]
    
    sets = colliding_sets(I, J)
    print(sets)