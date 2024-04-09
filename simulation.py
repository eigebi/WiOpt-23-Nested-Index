from env import MEC_Model
import numpy as np
from matplotlib import pyplot as plt


# r_max: the maximum scale of the system for simulation
N = 50
K = 2
r_max = 21
prob=[0.8,0.7,0.6,0.5,0.3,0.1]
t_=[2,4,8,16,32,64]
n_=[5,10,5,5,10,15]


prob_n=[]
t=[]
for i,pp,tt in zip(n_,prob,t_):
    prob_n += [pp for _ in range(i)]
    t += [tt for _ in range(i)]
p = np.array(prob_n)

steps = 10000

from bandit import index_func,threshold_tab, thres_L, expect_u

compare = []



if __name__=='__main__': 
    
    final_lambda = []
    #for r in range(2,3):
    for r in range(2, r_max):
        fin = []
        tau = [t for _ in range(r)]
        tau = np.array(tau).reshape(-1)
        prob_n = [prob_n for _ in range(r)]
        prob_n = np.array(prob_n).reshape(-1)
        env = MEC_Model(N*r, tau, prob_n, preemptive=False, max_t=K*r)
        status,_ = env.reset()
        result = []
        temp = np.zeros([N*r], dtype=np.float32)
        for i in range(steps):
            # heuristic manners, greedily choose server by age
            part = status[:,0]+ (status[:,0]-status[:,1])/prob_n
            #id_t = np.argpartition(status[:,0],-K*r)[-K*r:]
            id_t = np.argpartition(part,-K*r)[-K*r+1:]
            env.step(id_t)
            temp += status[:,0]
            result.append(np.sum(temp)/(i))
        result = np.array(result)/(N*r)
        fin.append(result[-1])
        


        # index policy
        status,_ = env.reset()
        resultp = []
        
        alpha = 0.01
        converge_lambda = []
        temp = np.zeros([N*r], dtype=np.float32) 
        lambda_result = -1.
        
        for i in range(steps):
            
            L = status[:,0]
            I = []
            for arm in range(N*r):
                I.append(index_func(L[arm],tau[arm],prob_n[arm]))
            I = np.array(I)
            lambda_temp = np.min(I)
            while(np.where(I > lambda_temp)[0].shape[0] > K*r):
                lambda_temp += 0.1
            lambda_result = (1-alpha)*lambda_result + alpha * (lambda_temp)
            converge_lambda.append(lambda_result)
            id_t = np.argpartition(I, -K*r)[-K*r:]
            env.step(id_t)
            temp += status[:,0]
            resultp.append(np.sum(temp)/(i+1.))
        final_lambda.append(converge_lambda)
        resultp = np.array(resultp)/(N*r)
        fin.append(resultp[-1])
        print(fin)
        compare.append(resultp[-1])
        

    p = np.array(prob_n)
    t = np.array(tau[:N])

    

    # calculation of the equivalent dual variable for the index policy
    nu = 10
    nu_list= []
    nu_list.append(nu)
    beta = 0.5
    tab = threshold_tab(t, p)
    for i in range(10000):
        H = thres_L(tab, nu)
        temp = expect_u(H,t,p)
        nu = nu+beta*(temp-K)
        nu_list.append(nu)
    plt.plot(nu_list, label = "the convergence of relaxed problem")
    plt.plot(final_lambda[0], label='r=3')
    #plt.plot(final_lambda[1], label='r=7') 
    #plt.plot(final_lambda[2], label='r=10')     
    #plt.title('The evolution of $\\nu$.')
    plt.ylabel('cost $\\nu$')
    plt.xlabel('iteration step')
    plt.legend()
    plt.show()

    #np.save("final_lambda10",arr = np.array(final_lambda))
    #np.save("optimal_nu10",arr = np.array(nu_list))
    ''''''
    # RRP policy
    tau = [t for _ in range(8)]
    tau = np.array(tau).reshape(-1)
    prob_n = [0.8,0.7,0.5,0.1,0.88,0.35]
    prob_n = [prob_n for _ in range(8)]
    prob_n = np.array(prob_n).reshape(-1)
    env = MEC_Model(N*r, tau, prob_n, preemptive=False, max_t=K*8)
    status,_ = env.reset()
    resultmm = []
    temp = np.zeros([N*r], dtype=np.float32)
    H = [H for _ in range(8)]
    H = np.array(H).reshape(-1)
    for i in range(steps):
        part = status[:,0] - H
        #id_t = np.argpartition(status[:,0],-K*r)[-K*r:]
        id_t = np.argpartition(part,-K*r)[-K*r:]
        env.step(id_t)
        temp += status[:,0]
        resultmm.append(np.sum(temp)/(i+1.))
    resultmm = np.array(resultmm)/(N*r)



    # optimal solution for the subproblems, same as 'lowerbound.py' does
    ave = 0    
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
        
    result_p = np.ones_like(result)*ave
    
    
    
    

        
        
    plt.figure(1)
    plt.grid(axis="y", linestyle="--")
    plt.xlabel("Time horizon /step", fontsize=10)
    plt.ylabel("Age", fontsize=10)
    #plt.ylim(100,160)
    plt.plot(result, label="Greedy with p")
    #plt.title("Sum of average age over all tasks (stochastic).")
    #plt.plot(result1, label="Whittle's Index")
    #plt.plot(result2, label='whittle prob_n')
    #plt.plot(result2, label='optimal for relaxed problem')
    plt.plot(resultp, label="index for stochastic situation")
    plt.plot(result_p, label='relaxed optimal')
    plt.plot(resultmm, label='RRP')
    plt.legend()
    plt.show()
    
    np.save("reducing_d", arr=np.array(result))
    np.save("RRP_d", arr=np.array(resultmm))
    #np.save("greedy_s",arr = np.array(result))
    #np.save("indexpolicy_s",arr = np.array(resultp))
    #np.save("optimal_s",arr = np.array(result_p))
    plt.figure(2)
    plt.plot(compare)
    #np.save("compare_s",arr = np.array(compare))
    
    #plt.title('Asymptotic optimality with the increase of scaler $r$ (stochastic).')
    plt.ylabel('Average AoI')
    plt.xlabel('system scalar $r$')
    plt.show()