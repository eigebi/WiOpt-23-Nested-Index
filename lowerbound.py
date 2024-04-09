# this .py file calculate the optimal solution and corresponding dual variable for each sub problems


import numpy as np
from matplotlib import pyplot as plt

# N: the number of users
# K: the number of edge servers
# prob_: The transition probability for each type of user
# t_: the minimum execution time \tau_min for each type of user
# n_: the number of each type of user
N = 50
K = 2
prob=[0.8,0.7,0.6,0.5,0.3,0.1]
t_=[2,4,8,16,32,64]
n_=[5,10,5,5,10,15]

# generate the complete lists of transition probability and \tau_min
prob_n=[]
t=[]
for i,pp,tt in zip(n_,prob,t_):
    prob_n += [pp for _ in range(i)]
    t += [tt for _ in range(i)]
p = np.array(prob_n)


steps = 10000
from bandit import index_func,threshold_tab, thres_L, expect_u

# Derive the optimal dual virable \nu of the subproblem using gradient ascent
compare = []
nu = 10000
nu_list= []
nu_list.append(nu)
beta = 500
tab = threshold_tab(t, p)
for i in range(10000):
    # for each subproblem, the problem is to find the optimal threshold of current age to generate and offload a new task. Therefore, we utilize the thres_L() func to find...
    # such a threshold, (since zero-wait is optimal, the first input is always \tau_min)
    # the expect_u is utilize the threshold just calculated to derive the equivelant dual variable \nu.
    H = thres_L(tab, nu)
    temp = expect_u(H,t,p)
    nu = nu+beta*(temp-K)
    nu_list.append(nu)
    

# ave: average aoi    
ave = 0    
# the calculation of average aoi is quite intuitive, just as many surveys showed: the expected sum age divided by the expected computing time
for L,_t,_p in zip(H,t,p):
    EA1 = 0
    EA2 = 0
    ET = 0
    for i in range(L-_t-1): 
        temp = _p*np.power(1-_p,i)
        EA1 += temp
        EA2 += temp * (L+_t+i)*(L-_t-i-1)/2
        ET += temp * (L-_t-i-1)
    EA1 = (EA1 + _p*np.power(1-_p,L-_t-1)) * L
    
    temp = (EA1-0.5)*(_t+(1-_p)/_p) + EA2 + (_t+1)*(_t+1)+(1-_p)/_p*(2*_t+3)+2*(1-_p)*(1-_p)/(_p*_p)
    temp = temp/(ET + _t + (1-_p)/_p)
    ave += temp
    
ave = ave/N
print(ave)

#97.25
#160.97