import numpy as np
from numpy import cos,sin

# Code of the exact Brownian increments on the sphere from  "Mijatović, A., Mramor, V., & Bravo, G. U. (2020). A note on the exact simulation of spherical Brownian motion. Statistics & Probability Letters, 165, 108836." Algorithm 1

# Intermediate functions 

def project(array):
    """ Orthogonal projection of all the points on the semi-sphere of radius 1 """
    return np.einsum("ij,i -> ij",array,1/np.sqrt(np.sum(array**2,axis = 1)))

def first_decay(t,m):
    k = m
    while ((2*k + 3)/(2*k + 1))*((m + k + 1)/(k-m + 1)) > np.exp((k+1)*t):
        k = k + 1
    return k

def coeffs_A(m,k,previous_A):
    try:
        previous_A[m,k]
        return  previous_A
    except:
        M,K = previous_A.shape
        new_array = np.zeros([max(m+1,M),max(k+1,K)])
        new_array[:M,:K] = previous_A
        for n in range(M,m+1):
            new_array[n,n] = new_array[n-1,n-1]* 2 * (2*n + 1)/(n+1)
            for k in range(n+1,max(k+1,K)):
                new_array[n,k] = (2*k +1)/(2*k - 1) * (n + k)/(k - n) * new_array[n,k-1]
        for n in range(M):
            for k in range(K,k+1):
                new_array[n,k] = (2*k +1)/(2*k - 1) * (n + k)/(k - n) * new_array[n,k-1]
        return new_array

def sum_on_i(t,k,M,A):
    s = 0
    A = coeffs_A(M,M+k,A)
    for i in range(k+1):
        s += (-1)**i * A[M,M+i]* np.exp(-(M+i)*(M+i+1)*t/2)
    return s, A

def sum_on_m1(t,karray,A):
    s = 0
    M = len(karray) - 1
    A = coeffs_A(M ,M + 2*np.max(karray) + 2,A)
    for m, km in enumerate(karray):
        i = 2*km + 2
        s = s + A[m,m + i ] *np.exp(-(m + i)*(m + i + 1)*t/2)
    return s, A

def sum_on_m2(t,karray,A):
    s = 0
    M = len(karray) - 1
    A = coeffs_A(M ,M + 2*np.max(karray) + 3,A)
    for m, km in enumerate(karray):
        i = 2*km + 3
        s = s + A[m,m + i ] *np.exp(-(m + i)*(m + i + 1)*t/2)
    return s, A

def sampleA_infinte_exact(t):
    m  = 0
    k0 = int(first_decay(t,m)/2) + 1
    karray = np.array([k0])
    U = np.random.rand()

    A = np.ones([1,1])

    Skplus_m, A = sum_on_i(t,2*k0,m,A)
    Skminus_m, A = sum_on_i(t,2*k0 + 1,m,A)
    while True:
        #print( "Step m = "+str(m))
        #print(m, Skminus_m,Skplus_m, U)
        while Skminus_m < U  and Skplus_m > U:
            #print("Thining for m = "+str(m))
            #print(Skminus_m, Skplus_m, U)
            sum, A = sum_on_m1(t,karray,A)
            Skplus_m = Skminus_m + sum

            sum, A = sum_on_m2(t,karray,A)
            Skminus_m = Skplus_m - sum

            karray += 1
        if Skminus_m > U:
            return m
        else:
            m = m + 1
            km = int(first_decay(t,m)/2) + 1
            karray = np.concatenate((karray,np.array([km])))

            sum, A = sum_on_i(t,2*km,m,A)
            Skplus_m  = Skplus_m + sum

            sum, A = sum_on_i(t,2*km + 1,m,A)
            Skminus_m = Skminus_m + sum

def sampleA_infinite_normal(t):
    """ For small time steps"""
    beta = t/2
    eta = beta/(np.exp(beta) - 1)
    mu = 2*eta/t
    sigma_sample_inf = np.sqrt(mu* (1 + eta/(eta + beta) - 2*eta))*(eta + beta)/beta
    return int(mu + sigma_sample_inf*np.random.randn()) 

def sampleA_infinte(t):
    """ Treshold given in the article"""
    if t >= 0.1:
        return sampleA_infinte_exact(t)
    else:
        return sampleA_infinite_normal(t)


def sample_WF(t,a,b):
    #print(t)
    #M = sample_infinite_descent(t,tol = tol)
    M = np.vectorize(sampleA_infinte)(t)
    Y = np.random.beta(a,b+M)
    return Y.reshape(-1)

# Exact Brownian sampling

def sample_exactBr(start,dt):
    size,_ = start.shape
    X = sample_WF(dt,1,1)

    phi = np.random.rand(size)*2*np.pi
    u = -start.copy()
    u[:,2] = 1 + u[:,2] 
    u = project(u)
    #u[np.where( u[:,2] > 0)]
    # if u[:,2] > 0:
    #     u = project(u)
    # else:
    #     pass
    O = np.einsum('ij,ik-> ijk',u,u)

    var = 2 * np.sqrt(X*(1-X))
    first_coordinate,second_coordinate,third_coordinate  = var *cos(phi) , var * sin(phi), 1 - 2*X
    vec = np.stack((first_coordinate,second_coordinate,third_coordinate),axis = 1)

    res = vec - 2*np.einsum('ijk,ik-> ij',O,vec)
    #print(np.sum(res**2)/size)
    return res