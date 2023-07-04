from env import MEC_Model
import numpy as np
from matplotlib import pyplot as plt
N = 50
K = 2
r_max = 21
prob_n = [0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.7,0.7,0.7,0.7,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.1,0.1,0.1,0.1,0.1,0.1,0.88,0.88,0.88,0.88,0.35,0.35]
#prob_n = [0.8,0.7,0.5,0.1,0.88,0.35,0.7,0.6,0.5,0.8,0.2,0.5]
#t = [3,6,15,2,7,22,8,15,30,14,17,21]

#prob_n = [1.,1.,1.,1.,1.,1.]
#prob_n = [0.8,0.6,0.2,0.9]
#tau = [3,7,15,2,19,22,50,36,28,42]
t = [3,6,15,2,7,22]
t=[3,3,3,3,3,3,3,5,5,5,5,8,8,8,8,8,8,8,22,22,22,22,22,22,2,2,2,2,15,15]

prob=[0.8,0.7,0.6,0.5,0.3,0.1]
t_=[2,4,8,16,32,64]

n_=[5,10,5,5,10,15]
prob_n=[]
t=[]
for i,pp,tt in zip(n_,prob,t_):
    prob_n += [pp for _ in range(i)]
    t += [tt for _ in range(i)]

steps = 10000
from bandit import index_func,threshold_tab, thres_L, expect_u

compare = []



if __name__=='__main__':
    
    final_lambda = []
    for r in range(20,21):
    #for r in [3,7,10]:
        fin = []
        tau = [t for _ in range(r)]
        tau = np.array(tau).reshape(-1)
        prob_n = [prob_n for _ in range(r)]
        prob_n = np.array(prob_n).reshape(-1)
        env = MEC_Model(N*r, tau, prob_n, preemptive=True, max_t=K*r)
        status,_ = env.reset()
        result = []
        temp = np.zeros([N*r], dtype=np.float32)
        for i in range(steps):
            part = status[:,0]+ (status[:,0]-status[:,1])/prob_n
            #id_t = np.argpartition(status[:,0],-K*r)[-K*r:]
            id_t = np.argpartition(part,-K*r)[-K*r+2:]
            env.step(id_t)
            temp += status[:,0]
            result.append(np.sum(temp)/(i))
        result = np.array(result)/(N*r)
        fin.append(result[-1])
        #以上是greedy 算法，可以根据part自由调整
        '''
        status,_ = env.reset()
        result1 = []
        temp = np.zeros([N*r], dtype=np.float32)
        for i in range(steps):
            L = status[:,0]
            I = (L+2)*(L+1)/np.array(tau)
            id_t = np.argpartition(I, -K*r)[-K*r:]
            env.step(id_t)
            temp += status[:,0]
            result1.append(np.sum(temp)/(i+1.))
        result1 = np.array(result1)/(N*r)
        fin.append(result1[-1])
        compare.append(result1[-1])
110.25 5.75
    
        '''
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

    '''
    nu = 50
    beta = 0.05
    for i in range(20000):
        H = np.floor(np.sqrt(2*nu*t)-t)
        for j in range(np.size(H)):
            if H[j] < 0:
                H[j]=1e-6
        temp = np.sum(t/(H+t))
        nu = nu + beta * (temp - K)
    result2 = np.sum((3*t+H+1)/2)/N
    result2 = np.ones_like(result1)*result2
    '''
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