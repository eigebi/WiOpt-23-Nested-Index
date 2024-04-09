import numpy as np

seed = 10088
np.random.seed(seed)

#step(id_t) returns the next state (and available num_a) for all arms
#id_t is a list, when preemptive=True, id_t is 2-dimensional: represent the action and type

class MEC_Model:
    def __init__(self, num_t, min_time, prob_n, preemptive=False, max_t=1):

        # num_t: the number of arms, integer
        # min_time: tau_min, minimum computation time, list
        # prob_n: the transition probability ,list
        # max_t: 
        # status: record the generated age and the current age of each arm, mat
        # preemptive: flag whether it is preemptive or non-preemptive

        self.num_t = num_t
        self.min_time = np.array(min_time, dtype=np.int8)
        self.prob_n = np.array(prob_n, dtype=np.float32)
        self.max_t = max_t
        self.status = np.zeros([self.num_t, 2],dtype=np.int32)

        self.preemptive = preemptive

        #evolve: plus which to evolve all the arm to the next time slot
        self.evolve = np.ones_like(self.status)
        
    def transitable(self, t):
        # judge whether the computation status is transitable, compared with tau_min
        # index 0 is the current age, index 1 is the age at the generation time
        # to optimize, can return the status as a whole in mat

        if self.status[t,0] - self.status[t, 1] < self.min_time[t]:
            return False
        else:
            return True
           
    def reset(self):
        self.status = np.zeros([self.num_t, 2], dtype=np.int32)
        return self.status, self.max_t
    
    # there maybe more efficient ways for step update
    def step(self, id_t):
        # every age plus one in advance
        self.status += self.evolve
        if not self.preemptive:
            # non-preemptive condition
            id_comp = np.where(self.status[:,0]!=self.status[:,1])[0]
            num_comp = np.size(id_comp)
            for t in id_comp:
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

            # operation according to the action
            for t in id_t:
                if t not in id_comp:
                    # if not in computation, it means to generate a new task and offload 
                    self.status[t,1] -= 1
                    num_comp += 1
                    if num_comp == self.max_t:
                        # some issues to address here in the future
                        break

                    '''
                    assert len(id_t)==self.num_t:
                        这里要让id_comp里不是id_t的那些arm set成idle的
                    '''

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

