import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#times = FontProperties(fname=r'\usr\share\fonts\truetype\msttcorefonts\times.ttf'[1:])
plt.rc('font', family='Times New Roman')
plt.rc('mathtext', fontset='stix')

#greedy_d = np.load('greedy_d.npy')
greedy_s = np.load('greedy_s.npy')
#index_d = np.load('indexpolicy_d.npy')
index_s = np.load('index_r_20_n_50.npy')
#optimal_d = np.load('optimal_d.npy')
#optimal_s = np.load('optimal_s.npy')
final_nu=np.zeros((10,10000),dtype=np.float32)
optimal_nu = np.load('optiaml_lambda.npy')
for i in range(10):
    temp=np.load("lambda_r_{n}_n_50.npy".format(n=2*i+2))[-1,:]
    final_nu[i,:]=temp

    #final_nu = np.concatenate((final_nu,temp),1)
#final_nu = np.load('lambda_r_20_n_50.npy')
reducing_s = np.load('reducing_s.npy')
rrp_s = np.load('rrp_s.npy')



greedy_s_=[]
index_s_=[]
reducing_s_=[]
rrp_s_=[]
x=500
for i in range(10000-x):
    greedy_s_.append( np.mean(greedy_s[i:i+x]))
    index_s_.append(np.mean(index_s[i:i+x]))
    reducing_s_.append(np.mean(reducing_s[i:i+x]))
    rrp_s_.append(np.mean(rrp_s[i:i+x]))

#compare_d = np.load('compare_d.npy')
#compare_s = np.load('compare_s.npy')


# the optimal dual variable for the subproblem
plt.figure(1)
optimal_nu=7861.6*np.ones(10000)
plt.plot(optimal_nu, label = "Optimal solution")



# convergence of the index policy, utilizing the equivalent dual variable in index policy
area_u = []
area_l = []
mean_nu = []
for i in range(10000-300):
    mean = np.mean(final_nu[0][i:i+300])
    std = np.std(final_nu[0][i:i+300])
    area_u.append(mean + 2*std)
    area_l.append(mean - 2*std)
    mean_nu.append(mean)
area_u = np.array(area_u)
area_l = np.array(area_l)
mean_nu = np.array(mean_nu)
plt.plot(mean_nu, label='Index policy with $r=2$',color='y')

plt.fill_between([i for i in range(10000-300)], area_l, area_u, color='y',alpha=0.2)

# convergence of the index policy, different server
area_u = []
area_l = []
mean_nu = []
for i in range(10000-300):
    mean = np.mean(final_nu[9][i:i+300])
    std = np.std(final_nu[9][i:i+300])
    area_u.append(mean + 2*std)
    area_l.append(mean - 2*std)
    mean_nu.append(mean)
area_u = np.array(area_u)
area_l = np.array(area_l)
mean_nu = np.array(mean_nu)
#plt.plot(final_nu[2], label='index policy with $r=3$')
plt.fill_between([i for i in range(10000-300)], area_l, area_u, color='r',alpha=0.2)
plt.plot(mean_nu, label='Index policy with $r=20$',color='r')     


plt.ylabel('Cost $\\nu$', fontsize =20)
plt.xlabel('Iteration step', fontsize =20)
plt.legend(fontsize =15)
plt.xticks(fontsize =10)
plt.yticks(fontsize =10)
plt.xlim([0,5000])
plt.show()
'''
'''
optimal_s=350.8*np.ones(10000)
plt.figure(2)
plt.grid(axis="y", linestyle="--")
plt.xlabel("Step", fontsize=20)
plt.ylabel("Average EE per 100 steps", fontsize=20)

plt.legend()
plt.xticks(fontsize =13)
plt.yticks(fontsize =13)
plt.legend(fontsize =15)
plt.xlim([0,10000])
#plt.ylim([0,320])
plt.show()

plt.figure(3)
compare=[]
axis=np.array([2,4,6,8,10,12,14,16,18,20])*50
optimal=350.81*np.ones(10)
for i in range(10):
    temp=np.load('index_r_{}_n_50.npy'.format(i*2+2))
    compare.append(np.max(temp))
    #com=np.sort(compare)[::-1][0:10]

plt.plot(axis,compare,label='Nested index',marker='o')
plt.plot(axis,optimal,label='Lower bound',linestyle="--")
plt.ylabel('Normalized Average AoI', fontsize=20)
#plt.xlabel('System Scalar $r$', fontsize=20)
plt.xlabel('Total number of users', fontsize=20)
plt.xticks(fontsize =13)
plt.yticks(fontsize =13)
plt.legend(fontsize =15)
plt.grid()
#plt.ylim([159,163])
plt.show()
