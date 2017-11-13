import gym
import os
import numpy as np 
import time 
import random
import gc
from itertools import count

import threading
from utils.replayBuffer import EXP,PrioritizedReplayBuffer
from utils.statsLogger import statsLogger
#from utils.histogram import HistogramDebug


import logging
bashlogger = logging.getLogger("bash logger")
bashlogger.setLevel(logging.DEBUG)
FORMAT = '[%(asctime)-15s][%(threadName)s][%(levelname)s][%(funcName)s] %(message)s'
logging.basicConfig(format=FORMAT)

import torch
import torchvision.transforms as T



class Worker :
	def __init__(self,index,model,env,memory,preprocess=T.ToTensor(),path=None,frompath=None,num_episodes=1000,HER={'use_her':True,'k':4,'strategy':'future','singlegoal':False},use_cuda=True,rendering=False) :
		self.index = index
		self.model = model
		self.envstr = env
		self.env = gym.make(self.envstr)
		self.env.reset()

		self.memory = memory
		self.optimizers = self.model.generate_optimizers()
		
		self.preprocess = preprocess
		self.path = path
		self.frompath = frompath
		
		self.num_episodes = num_episodes
		
		#HER params :
		self.HER = HER

		self.use_cuda = use_cuda
		self.rendering = rendering
		
		self.sl = statsLogger(path=self.path,filename='logs{}.csv'.format(self.index) )
		
		self.workerfn = lambda: self.trainIN( index=self.index,
										model=self.model,
										env=self.env,
										memory=self.memory,
										optimizers=self.optimizers,
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


	def trainIN(self,index,model,env,memory,optimizers,logger=None,preprocess=T.ToTensor(),path=None,frompath=None,num_episodes=1000,HER={'use_her':True,'k':4,'strategy':'future','singlegoal':False},use_cuda=True,rendering=False): 
		try :
			episode_durations = []
			episode_reward = []
			episode_qsa = []
			episode_grad_actor = []
			episode_loss = []
			
			#accumulateMemory(memory,env,models,preprocess,epsstart=0.5,epsend=0.3,epsdecay=200,k=k,strategy=strategy)

			#hd = HistogramDebug()
			#hd.setXlimit(-2.5,2.5)

			for i in range(num_episodes) :
				bashlogger.info('Episode : {} : memory : {}/{}'.format(i,len(memory),memory.capacity) )
				
				cumul_reward = 0.0
				last_state = get_state_reset(env,preprocess=preprocess)
				state, reward, done, info = get_state(env,env.action_space.sample(),preprocess=preprocess )
				
				episode_buffer = []
				
				meanfreq = 0
				
				episode_qsa_buffer = []
				episode_loss_buffer = []
				episode_grad_actor_buffer = []
				action_buffer = []

				#HER : sample initial goal :
				if HER['use_her'] :
					#if not HER['singlegoal'] :
					#	init_goal = sample_init_goal(memory)
					#else :
					#	init_goal = torch.zeros(state.size())
					init_goal = torch.zeros(state.size())

				showcount = 0

				for t in count() :
					#HER :
					if HER['use_her'] :
						evalstate = torch.cat( [state,init_goal], dim=1)
					else :
						evalstate = state

					action = model.act(evalstate, exploitation=False)
					action_buffer.append(float(action) )

					taction = torch.from_numpy(action.astype(np.float32))

					last_state = state
					state, reward, done, info = get_state(env,action,preprocess=preprocess)
					cumul_reward += float(reward)
					treward = torch.from_numpy(reward.astype(np.float32))

					if rendering :
						if showcount >= 10 :
							showcount = 0
							#render(current_state)
							env.render()
						else :
							showcount +=1


					episode_buffer.append( EXP(last_state,taction,state,treward,done) )

					episode_qsa_buffer.append( model.evaluate(evalstate, taction) )

					#Optimize model :
					since = time.time()
					retloss = model.optimize(optimizer_critic=optimizers['critic'],optimizer_actor=optimizers['actor'])
					if retloss is not None :
						critic_loss,actor_loss, actor_grad = retloss
						episode_loss_buffer.append(  np.mean(critic_loss) )
						episode_grad_actor_buffer.append(  actor_grad )
					else :
						episode_loss_buffer.append(0)
						episode_grad_actor_buffer.append(0)

					elt = time.time() - since
					f = 1.0/elt
					meanfreq = (meanfreq*(t+1) + f)/(t+2)
					#print('{} Hz ; {} seconds.'.format(f,elt) )
					
					
					if done :
						episode_durations.append(t+1)
						episode_reward.append(cumul_reward)
						meanloss = np.mean(episode_loss_buffer)
						episode_loss.append(meanloss)
						meanqsa = np.mean(episode_qsa_buffer) 
						maxqsa = np.max(episode_qsa_buffer) 
						episode_qsa.append( meanqsa)
						meanactorgrad = np.max(episode_grad_actor_buffer) 
						episode_grad_actor.append( meanactorgrad)
						meanaction = np.mean(action_buffer)
						sigmaaction = np.std(action_buffer)
						#hd.append(np.array(action_buffer) )

						log = 'Episode duration : {}'.format(t+1) +'---' +' Action : mu:{:.4f} sig:{:.4f} // Reward : {} // Mean Loss : {:.4f} // Mean/MaxQsa : {:.4f}/{:.4f} // Mean Actor Grad : {:.8f}'.format(meanaction,sigmaaction,cumul_reward,meanloss,meanqsa,maxqsa,meanactorgrad) +'---'+' {}Hz'.format(meanfreq)
						bashlogger.info(log)
						if logger is not None :
							new = {'episodes':[i],'duration':[t+1],'reward':[cumul_reward],'mean frequency':[meanfreq],'loss':[meanloss],'max qsa':[maxqsa],'mean qsa':[meanqsa],'mean action':[meanaction]}
							logger.append(new)

						if path is not None :
							model.save(path+'.save')
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
						for j in range(HER['k']) :
							goal = None
							if HER['strategy'] == 'final' :
								goal = sample_goal(episode_buffer, strategy=HER['strategy'])
							elif HER['strategy'] == 'future' :
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
					for el in episode_buffer :
						#store this transition 
						init_priority = memory.priority( torch.abs(el.reward).numpy() )
						memory.add(el,init_priority)

				del episode_buffer
				# check memory consumption and clear memory
				gc.collect()

			bashlogger.info('Learning complete.')
			if path is not None :
				savemodel(model,path+'.save')
				bashlogger.info('Model saved : {}'.format(path) )
			
			env.close()
		
		except Exception as e :
			bashlogger.exception(e)
			


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
	state = np.reshape(state,(1,-1))
	reward = np.reshape(reward, (1))
	#state = state.transpose( (2,0,1) )
	#state = np.ascontiguousarray( state, dtype=np.float32) / 255.0
	state = np.ascontiguousarray( state, dtype=np.float32)
	state = torch.from_numpy(state)
	#state = preprocess(state)
	#state = state.unsqueeze(0)
	
	return state, reward, done, info

def get_state_reset(env,preprocess) :
	state = env.reset()
	state = np.reshape(state,(1,-1))
	#state = state.transpose( (2,0,1) )
	#state = np.ascontiguousarray( state, dtype=np.float32) / 255.0
	state = np.ascontiguousarray( state, dtype=np.float32)
	state = torch.from_numpy(state)
	#state = preprocess(state)
	#state = state.unsqueeze(0)
	return state



		


