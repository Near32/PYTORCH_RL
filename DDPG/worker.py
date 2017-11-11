import gym
import os
import numpy as np 
import time 
import random

import threading
from utils.replayBuffer import EXP,PrioritizedReplayBuffer
from utils.statsLogger import statsLogger


import logging
bashlogger = logging.getLogger("bash logger")
bashlogger.setLevel(logging.DEBUG)
FORMAT = '[%(asctime)-15s][%(threadName)s][%(levelname)s][%(funcName)s] %(message)s'
logging.basicConfig(format=FORMAT)

from model import Model


class Worker :
	def __init__(self,index,model,env,memory,preprocess=T.ToTensor(),path=None,frompath=None,num_episodes=1000,HER={'use_her':True,'k':4,'strategy':'future','singlegoal':False},use_cuda=True,rendering=False) :
		self.index = index
		self.model = model
		self.envstr = env
		self.env = gym.make(self.envstr)
		self.env.reset()

		self.memory = memory
		self.lr = lr

		self.preprocess = preprocess
		self.path = path
		self.frompath = frompath
		
		self.num_episodes = num_episodes
		
		#HER params :
		self.HER = HER

		self.use_cuda = use_cuda
		self.rendering = rendering
		
		self.sl = statsLogger(path=self.path,filename='logs{}.csv'.format(self.index) )
		self.workerfn = lambda: train( index=self.index,
										model=self.model,
										env=self.env,
										memory=self.memory,
										logger=self.sl,
										preprocess=self.preprocess,
										path=self.path,
										frompath=self.frompath,
										num_episodes=self.num_episodes,
										HER=self.HER,
										use_cuda=self.use_cuda,
										rendering=self.rendering)

		self.thread = threading.Thread(target=self.workerfn)

	def start(self) :
		self.thread.start()

	def join(self) :
		self.thread.join()


def sample_goal(buffer_exp,strategy='final') :
	'''
	params :
		- buffer_exp : buffer of EXP object from which we will sample according to the strategy
		- strategy : 'final' / IDX(int) :
			- 'final' : returns last state of the buffer...
			- IDX(int) : isinstance(int,strategy) returns any state in buffer_exp[IDX:]
	'''
	if isinstance(strategy,int) :
		buff = buffer_exp[strategy:]
	else :
		buff = buffer_exp[-2:]

	return random.choice(buff).state 

def sample_init_goal(memory) :
	buffer_exp = memory.get_buffer()
	shape = buffer_exp[0].state.size()
	goalb = int(shape[1]/2)
	retstate = random.choice(buffer_exp).state
	ret = retstate[:,goalb:].view((1,goalb,shape[2],shape[3]))
	del buffer_exp
	return ret

def reward_function(state,goal,eps=1e-1) :
	#diff = torch.abs(state-goal).mean()
	diff = torch.abs(state-goal).max()
	val = -(diff > eps)
	
	return torch.ones(1)*float(val) 



def get_state(env,action,preprocess) :
	state, reward, done, info = env.step(action)
	#state = state.transpose( (2,0,1) )
	#state = np.ascontiguousarray( state, dtype=np.float32) / 255.0
	state = torch.from_numpy(state)
	state = preprocess(state)
	state = state.unsqueeze(0)
	
	return state, reward, done, info

def get_state_reset(env,preprocess) :
	state = env.reset()
	#state = state.transpose( (2,0,1) )
	#state = np.ascontiguousarray( state, dtype=np.float32) / 255.0
	state = torch.from_numpy(state)
	state = preprocess(state)
	state = state.unsqueeze(0)
	return state


def train(index,model,env,memory,logger=None,preprocess=T.ToTensor(),path=None,frompath=None,num_episodes=1000,HER={'use_her':True,'k':4,'strategy':'future','singlegoal':False},use_cuda=True,rendering=False): 
	try :
		episode_durations = []
		episode_reward = []
		episode_loss = []
		
		accumulateMemory(memory,env,models,preprocess,epsstart=0.5,epsend=0.3,epsdecay=200,k=k,strategy=strategy)

		for i in range(num_episodes) :
			bashlogger.info('Episode : {} : memory : {}/{}'.format(i,len(memory),memory.capacity) )
			
			cumul_reward = 0.0
			last_state = get_state_reset(env,preprocess=preprocess)
			state, reward, done, info = get_state(env,env.action_space.sample(),preprocess=preprocess )
			
			episode_buffer = []
			meanfreq = 0
			episode_loss_buffer = []

			#HER : sample initial goal :
			if HER['use_her'] :
				if not HER['singlegoal'] :
					init_goal = sample_init_goal(memory)
				else :
					init_goal = torch.zeros(current_state.size())

			showcount = 0

			for t in count() :
				#HER :
				if HER['use_her'] :
					evalstate = torch.cat( [state,init_goal], dim=1)
				else :
					evalstate = state

				action = model.act(evalstate, exploitation=False)

				cpu_action = action.squeeze(1).cpu().numpy()

				last_state = state
				state, reward, done, info = get_state(env,cpu_action,preprocess=preprocess)
				cumul_reward += reward
				
				if rendering :
					if showcount >= 10 :
						showcount = 0
						#render(current_state)
						env.render()
					else :
						showcount +=1


				episode_buffer.append( EXP(last_state,action,state,reward,done) )

				if done :
					nbrTrain = 200
					for it in range(nbrTrain) :
						since = time.time()
						critic_loss,actor_loss = model.optimize()
						if critic_loss is not None :
							episode_loss_buffer.append(  np.mean(critic_loss) )
						else :
							episode_loss_buffer.append(0)
						
						elt = time.time() - since
						f = 1.0/elt
						meanfreq = (meanfreq*(it+1) + f)/(it+2)
						#print('{} Hz ; {} seconds.'.format(f,elt) )
						
					episode_durations.append(t+1)
					episode_reward.append(cumul_reward)
					meanloss = np.mean(episode_loss_buffer)
					episode_loss.append(meanloss)

					log = 'Episode duration : {}'.format(t+1) +'---' +' Reward : {} // Mean Loss : {}'.format(cumul_reward,meanloss) +'---'+' {}Hz'.format(meanfreq)
					bashlogger.info(log)
					if logger is not None :
						new = {'episodes':[i],'duration':[t+1],'reward':[cumul_reward],'mean frequency':[meanfreq],'loss':[meanloss]}
						logger.append(new)

					if path is not None :
						model.save(model,path+'.save')
						bashlogger.info('Model saved : {}'.format(path) )
					break


			#Let us add this episode_buffer to the replayBuffer :
			if HER['use_her'] :
				for itexp in range(len(episode_buffer)) :
					el = episode_buffer[itexp]
					#HER : reward with init_goal
					HERreward = reward_function(el.state,init_goal)
					reward = HERreward+el.reward
					
					#store this transition with init_goal :
					init_el = EXP( state=torch.cat( [el.state, init_goal], dim=1),
									action=el.action,
									next_state=torch.cat( [el.next_state, init_goal], dim=1),
									reward=reward,
									done=el.done
								)
					
					init_priority = memory.priority( torch.abs(init_el.reward).numpy() )
					
					memory.add(init_el,init_priority)

					#store for multiple goals :
					#1: sample new goal :
					goals = []
					for j in range(k) :
						goal = None
						if strategy == 'final' :
							goal = sample_goal(episode_buffer, strategy=strategy)
						elif strategy == 'future' :
							# watch out for the empty set...
							index = min(len(episode_buffer)-3,itexp)
							goal = sample_goal(episode_buffer, strategy=index)	
						goals.append(goal)
						

					#For each goal ...
					for goal in goals :
						#2: .. compute reward :
						goalreward = reward_function(el.state,goal)+el.reward
						#3: ... store this transistion with goal :
						goalel = EXP( state=torch.cat( [el.state, goal], dim=1),
										action=el.action,
										next_state = torch.cat( [el.next_state, goal], dim=1),
										reward = goalreward,
										done=el.done
									)
						
						init_priority = memory.priority( torch.abs(goalel.reward).numpy() )
						memory.add(goalel,init_priority)
						
					del el
					del goals
			else :
				for el in range(episode_buffer) :
					#store this transition 
					init_priority = memory.priority( torch.abs(el.reward).numpy() )
					memory.add(el,init_priority)

			del episode_buffer

		bashlogger.info('Learning complete.')
		if path is not None :
			savemodel(model,path+'.save')
			bashlogger.info('Model saved : {}'.format(path) )
		
		env.close()
	
	except Exception as e :
		bashlogger.exception(e)
		