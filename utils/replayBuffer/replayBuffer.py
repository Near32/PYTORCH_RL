import numpy as np
from collections import namedtuple
import random

'''
class EXP :
	def __init__(self, s, a, s1, r, done) :
		self.s = s
		self.a = a
		self.s1 = s1
		self.r = r
		self.done = done
'''
EXP = namedtuple('EXP', ('state','action','next_state', 'reward','done') )
Transition = namedtuple('Transition', ('state','action','next_state', 'reward','done') )
TransitionPR = namedtuple('TransitionPR', ('idx','priority','state','action','next_state', 'reward','done') )

		
class EXP_RNN(EXP) :
	def __init__(self, s, a, s1, r, done, rnn_states) :
		EXP.__init__(self,s=s,a=a,s1=s1,r=r,done=done)
		self.rnn_states = rnn_states


class PrioritizedReplayBuffer :
	def __init__(self,capacity, alpha=0.2) :
		self.length = 0
		self.counter = 0
		self.alpha = alpha
		self.epsilon = 1e-6
		self.capacity = int(capacity)
		self.tree = np.zeros(2*self.capacity-1)
		self.data = np.zeros(self.capacity,dtype=object)
	
	def reset(self) :
		self.__init__(capacity=self.capacity,alpha=self.alpha)

	def add(self, exp, priority) :
		idx = self.counter + self.capacity -1
		
		self.data[self.counter] = exp
		
		self.counter += 1
		self.length = min(self.length+1, self.capacity)
		if self.counter >= self.capacity :
			self.counter = 0
		
		self.update(idx,priority)
	
	def priority(self, error) :
		return (error+self.epsilon)**self.alpha
			
	def update(self, idx, priority) :
		change = priority - self.tree[idx]
		
		self.tree[idx] = priority
		
		self._propagate(idx,change)
		
	def _propagate(self, idx, change) :
		parentidx = (idx - 1) // 2
		
		self.tree[parentidx] += change
		
		if parentidx != 0 :
			self._propagate(parentidx, change)
			
	def __call__(self, s) :
		idx = self._retrieve(0,s)
		dataidx = idx-self.capacity+1
		data = self.data[dataidx]
		priority = self.tree[idx]
		
		return (idx, priority, data)
	
	def get(self, s) :
		idx = self._retrieve(0,s)
		dataidx = idx-self.capacity+1
		
		data = self.data[dataidx]
		if not isinstance(data,EXP) :
			raise TypeError
				
		priority = self.tree[idx]
		
		return (idx, priority, *data)
	
	def get_buffer(self) :
		return [ self.data[i] for i in range(self.capacity) if isinstance(self.data[i],EXP) ]

			
	def _retrieve(self,idx,s) :
		 leftidx = 2*idx+1
		 rightidx = leftidx+1
		 
		 if leftidx >= len(self.tree) :
		 	return idx
		 
		 if s <= self.tree[leftidx] :
		 	return self._retrieve(leftidx, s)
		 else :
		 	return self._retrieve(rightidx, s-self.tree[leftidx])
		 	
	def total(self) :
		return self.tree[0]

	def __len__(self) :
		return self.length


class ReplayMemory(object) :
	def __init__(self,capacity) :
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def reset(self) :
		self.__init__(capacity=self.capacity)
	
	def add(self, exp) :
		if len(self.memory) < self.capacity :
			self.memory.append(None)
		self.memory[self.position] = (self.position, 1.0, *exp)
		self.position = (self.position+1) % self.capacity
		self.position = int(self.position)

	def sample(self,batch_size) :
		return random.sample(self.memory, batch_size)

	def __len__(self) :
		return len(self.memory)

