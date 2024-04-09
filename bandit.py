import numpy as np

# something to do to accelerate the calculation (like co_mu was not used)

def index_func(l,t,p):
    
    
    #l is the generated time; t is the current age; p is the transition probability  
    #this calculation for the index is utilized in the non-preemptive condition.
    
    co_mu = 0
    delta_t = 0
    for i in range(1, l-t-1):
        pow_temp = np.power((1-p),i)
        co_mu += p*pow_temp*i
        delta_t -= p*pow_temp*np.floor((2*t+1+i)*i/2)
    pow_temp = np.power(1-p,l-t-1)
    co_mu += pow_temp*(l-t-1)
    delta_t = delta_t - pow_temp*(l-t-1) + pow_temp*(1-p)/p*((t+1+1/p))
    co_mu = l+1/p-1-co_mu
    delta_t += (l+t)*(l-t-1)/2+(2*l+t)*(t+1)/2+(1-p)/p*(l+t+1)+(1-p)*(1-p)/p
    I = ((l+t+1+1/p)*(l+1/p-1)-delta_t)/(t+1/p+1)
    if I <= 0:
        return 0
    return I

    # The index I returns the minimum value of dual value \mu that makes offloading and not offloading the same optimal by definition. 
    # As it has a closed form under the non-preemptive condition, we calculate as in the last equation.

def threshold_tab(tau, p_prob):

    # Calculate a threshold tab for each sub-problem in advance
    threshold = []
    for t,p in zip(tau, p_prob):
        tab = np.zeros((1000))
        for i in range(1000):
            tab[i] = index_func(i, t, p)
        threshold.append(tab)
    return threshold
        
def thres_L(thres_tab, cost):

    # Look up the age when a new task has to be offloaded with given cost.
    temp = []
    for tab in thres_tab:
        L = np.where(tab<cost)[0][-1]+1
        temp.append(L)
    return temp

def expect_u(thres_L, tau, p_prob):

    # this func calculate the equivalent \mu for the practical policy
    temp = 0
    for L,t,p in zip(thres_L, tau, p_prob):
        numerator = 0
        for i in range(L):
            numerator += p*np.power(1-p,i)*(L-i)
        denomin = t + 1/p
        fraction  = denomin/(denomin + numerator)
        temp += fraction
    return temp
if __name__ == '__main__':
    print('test')
    tab = threshold_tab([3,5,6],[0.8,0.7,0.6])
    threshold = thres_L(tab, 5)
    u = expect_u(threshold,[3,5,6],[0.8,0.7,0.6])