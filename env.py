import numpy as np

seed = 10088
np.random.seed(seed)

#step(id_t) returns the next state (and available num_a) for all arms
#id_t is a list, when preemptive=True, id_t is 2-dimensional: represent the action and type

class MEC_Model:
    def __init__(self, num_t, min_time, prob_n, preemptive=False, max_t=1):
        self.num_t = num_t
        self.min_time = np.array(min_time, dtype=np.int8)
        self.prob_n = np.array(prob_n, dtype=np.float32)
        self.max_t = max_t
        self.status = np.zeros([self.num_t, 2],dtype=np.int32)
        self.preemptive = preemptive

        #plus which to evolve all the arm to the next time slot
        self.evolve = np.ones_like(self.status)
        
    def transitable(self, t):
        #0 is the current age, 1 is the age at the generation time
        if self.status[t,0] - self.status[t, 1] < self.min_time[t]:
            return False
        else:
            return True
           
    def reset(self):
        self.status = np.zeros([self.num_t, 2], dtype=np.int32)
        return self.status, self.max_t
    
    #num_a is the number of available action
    def step(self, id_t):
        self.status += self.evolve
        if not self.preemptive:
            id_comp = np.where(self.status[:,0]!=self.status[:,1])[0]
            num_comp = np.size(id_comp)
            for t in id_comp:
                #assert t not in id_t
                if self.transitable(t):
                    is_finished = bool(np.random.choice(2, p=[1-self.prob_n[t], self.prob_n[t]]))
                    if not is_finished:
                        self.status[t,1] -= 1
                    else:
                        temp = self.status[t,0] - self.status[t,1]
                        self.status[t,0] = temp + 1
                        self.status[t,1] = temp + 1
                else:
                    self.status[t,1] -= 1
            for t in id_t:
                if t not in id_comp:
                    self.status[t,1] -= 1
                    num_comp += 1
                    if num_comp == self.max_t:
                        break

        else:
            id_comp = np.where(self.status[:,0]!=self.status[:,1])[0]
            for t in id_comp:
                if t not in id_t:
                    self.status[t,1] = self.status[t,0]
            for t in id_t:
                if self.transitable(t):
                    is_finished = bool(np.random.choice(2, p=[1-self.prob_n[t], self.prob_n[t]]))
                    if not is_finished:
                        self.status[t,1] -= 1
                    else:
                        temp = self.status[t,0] - self.status[t,1]
                        self.status[t,0] = temp + 1
                        self.status[t,1] = temp + 1
                else:
                    self.status[t,1] -= 1