from __future__ import division

import math
import random
import os

import numpy as np

import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import threading
from threading import Lock
import copy

from utils.replayBuffer import EXP,PrioritizedReplayBuffer
from utils.statsLogger import statsLogger
from utils.utils import hard_update, soft_update

import torchvision.transforms as T
import logging

import gym


bashlogger = logging.getLogger("bash logger")
bashlogger.setLevel(logging.DEBUG)
FORMAT = '[%(asctime)-15s][%(threadName)s][%(levelname)s][%(funcName)s] %(message)s'
logging.basicConfig(format=FORMAT)


use_cuda = True#torch.cuda.is_available()
rendering = False
MAX_STEPS = 1000
REWARD_SCALER = 1.0

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

Transition = namedtuple('Transition', ('state','action','next_state', 'reward','done') )
TransitionPR = namedtuple('TransitionPR', ('idx','priority','state','action','next_state', 'reward','done') )

class ReplayMemory(object) :
	def __init__(self,capacity) :
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args) :
		if len(self.memory) < self.capacity :
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position+1) % self.capacity
		self.position = int(self.position)

	def sample(self,batch_size) :
		return random.sample(self.memory, batch_size)

	def __len__(self) :
		return len(self.memory)


mutex = Lock()

class DQN(nn.Module) :
	def __init__(self,nbr_actions=2,actfn= lambda x : F.leaky_relu(x,0.1) ) :
		super(DQN,self).__init__()
		
		self.nbr_actions = nbr_actions

		self.actfn = actfn

		self.conv1 = nn.Conv2d(3,16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)
		#self.head = nn.Linear(448,self.nbr_actions)
		self.head = nn.Linear(192,self.nbr_actions)

	def forward(self, x) :
		x = self.actfn( self.bn1(self.conv1(x) ) )
		x = self.actfn( self.bn2(self.conv2(x) ) )
		x = self.actfn( self.bn3(self.conv3(x) ) )
		x = x.view( x.size(0), -1)
		x = self.head( x )
		return x


	def lock(self) :
		global mutex
		mutex.acquire()

	def unlock(self) :
		global mutex
		mutex.release()


class DuelingDQN(nn.Module) :
	def __init__(self,nbr_actions=2,actfn= lambda x : F.leaky_relu(x,0.1), useCNN={'use':True,'dim':3} ) :
		super(DuelingDQN,self).__init__()
		
		self.nbr_actions = nbr_actions

		self.actfn = actfn
		self.useCNN = useCNN

		if self.useCNN['use'] :
			self.conv1 = nn.Conv2d(3,16, kernel_size=5, stride=2)
			self.bn1 = nn.BatchNorm2d(16)
			self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
			self.bn2 = nn.BatchNorm2d(32)
			self.conv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
			self.bn3 = nn.BatchNorm2d(32)
			#self.f = nn.Linear(192,128)
			self.f = nn.Linear(1120,32)
		
		else :
			self.f1 = nn.Linear(self.useCNN['dim'], 1024)	
			self.f2 = nn.Linear(1024, 256)
			self.f = nn.Linear(256,64)	
			
		self.value = nn.Linear(64,1)
		self.advantage = nn.Linear(64,self.nbr_actions)
		

	def forward(self, x) :
		if self.useCNN['use'] :
			x = self.actfn( self.bn1(self.conv1(x) ) )
			x = self.actfn( self.bn2(self.conv2(x) ) )
			x = self.actfn( self.bn3(self.conv3(x) ) )
			x = x.view( x.size(0), -1)
		
			#print(x.size())
		else :
			x = self.actfn( self.f1(x) )
			x = self.actfn( self.f2(x) )

		fx = self.actfn( self.f(x) )

		v = self.value(fx)
		va = torch.cat( [ v for i in range(self.nbr_actions) ], dim=1)
		a = self.advantage(fx)

		suma = torch.mean(a,dim=1,keepdim=True)
		suma = torch.cat( [ suma for i in range(self.nbr_actions) ], dim=1)
		
		x = va+a-suma
		
			
		return x

	
	def lock(self) :
		global mutex
		mutex.acquire()

	def unlock(self) :
		global mutex
		mutex.release()




def get_screen(task,action,preprocess) :
	global REWARD_SCALER
	screen, reward, done, info = task.step(action)
	reward = reward/REWARD_SCALER
	#screen = screen.transpose( (2,0,1) )
	#screen = np.ascontiguousarray( screen, dtype=np.float32) / 255.0
	screen = np.ascontiguousarray( screen, dtype=np.float32)
	screen = torch.from_numpy(screen)
	#screen = preprocess(screen)
	screen = screen.unsqueeze(0)
	#screen = screen.type(Tensor)
	return screen, reward, done, info

def get_screen_reset(task,preprocess) :
	screen = task.reset()
	#screen = screen.transpose( (2,0,1) )
	#screen = np.ascontiguousarray( screen, dtype=np.float32) / 255.0
	screen = np.ascontiguousarray( screen, dtype=np.float32)
	screen = torch.from_numpy(screen)
	#screen = preprocess(screen)
	screen = screen.unsqueeze(0)
	return screen


def select_action(model,state,steps_done=[],epsend=0.05,epsstart=0.9,epsdecay=200) :
	global nbr_actions
	sample = random.random()
	if steps_done is [] :
		steps_done.append(0)

	eps_threshold = epsend + (epsstart-epsend) * math.exp(-1.0 * steps_done[0] / epsdecay )
	steps_done[0] +=1

	#print('SAMPLE : {} // EPS THRESH : {}'.format(sample, eps_threshold) )
	if sample > eps_threshold :
		output = model( Variable(state, volatile=True).type(FloatTensor) ).data
		action = output.max(1)[1].view(1,1)
		qsa = output.max(1)[0].view(1,1)[0,0]
		return action, qsa
	else :
		return LongTensor( [[random.randrange(nbr_actions) ] ] ), 0.0

def exploitation(model,state) :
	global nbr_actions
	output = model( Variable(state, volatile=True).type(FloatTensor) ).data.max(1)
	action = output[1].view(1,1)
	qsa = output[0].view(1,1)[0,0]
	return action,qsa

	
def plot_durations() :
	plt.figure(2)
	plt.clf()
	durations_t = torch.FloatTensor(episode_durations)
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Duration')
	plt.plot(durations_t.numpy())
	# Take 100 episode averages and plot them too
	if len(durations_t) >= 100:
	    means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
	    means = torch.cat((torch.zeros(99), means))
	    plt.plot(means.numpy())

	plt.pause(0.001)  # pause a bit so that plots are updated
	if is_ipython:
	    display.clear_output(wait=True)
	    display.display(plt.gcf())




def render(frame) :
	if use_cuda :
		plt.imshow( frame.cpu().squeeze(0).permute( 1, 2 ,0 ).numpy(), interpolation='none')
	else :
		plt.imshow( frame.squeeze(0).permute( 1, 2 ,0 ).numpy(), interpolation='none')
	plt.title('Current Frame')
	plt.pause(1e-3)



class Worker :
	def __init__(self,index,model,env,memory,lr=1e-3,preprocess=T.ToTensor(),path=None,frompath=None,num_episodes=1000,epsend=0.05,epsstart=0.9,epsdecay=10,TAU=1e-3) :
		self.index = index
		self.model = model

		self.wmodel = copy.deepcopy(model)
		hard_update(self.wmodel,self.model)
		global use_cuda
		if use_cuda :
				self.wmodel = self.wmodel.cuda()
			
		self.envstr = env
		self.env = gym.make(self.envstr)
		self.env.reset()

		self.memory = memory
		self.lr = lr
		self.TAU = TAU

		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr )
		bashlogger.info('Optimizer {}: ok.'.format(self.index) )

		self.preprocess = preprocess
		self.path = path
		self.frompath = frompath
		self.num_episodes = num_episodes
		self.epsend = epsend
		self.epsstart = epsstart
		self.epsdecay = epsdecay

		self.sl = statsLogger(path=self.path,filename='logs{}.csv'.format(self.index) )
		self.workerfn = lambda: self.train(model=self.wmodel,
										env=self.env,
										memory=self.memory,
										optimizer=self.optimizer,
										logger=self.sl,
										preprocess=self.preprocess,
										path=self.path,
										frompath=self.frompath,
										num_episodes=self.num_episodes,
										epsend=self.epsend,
										epsstart=self.epsstart,
										epsdecay=self.epsdecay)

		self.thread = threading.Thread(target=self.workerfn)

	def start(self) :
		self.thread.start()

	def join(self) :
		self.thread.join()
	
	def from_worker2model(self) :
		self.model.lock()

		self.optimizer.zero_grad()

		decay_lambda = 1e-0
		decay_loss = decay_lambda * 0.5*sum( [torch.mean(param*param) for param in self.model.parameters()])
		decay_loss.backward()
		
		for wparam, mparam in zip(self.wmodel.parameters(), self.model.parameters() ) :
			if mparam.grad is not None:
				if wparam.grad is not None :
					mparam.grad =  mparam.grad + wparam.grad
				
		self.optimizer.step()

		#update wmodel :
		hard_update(self.wmodel,self.model)

		#zero the working model gradients :
		self.wmodel.zero_grad()
		
		self.model.unlock()


	def optimize_model(self,model,model_,memory,optimizer) :
		try :
			global last_sync
			global use_cuda
			global MIN_MEMORY
			global nbr_actions
			
			if len(memory) < MIN_MEMORY :
				return
			
			#Create Batch with PR :
			prioritysum = memory.total()
			
			# Random Experience Sampling with priority
			#randexp = np.random.random(size=BATCH_SIZE)*prioritysum
			
			# Sampling within each sub-interval :
			#step = prioritysum / BATCH_SIZE
			#randexp = np.arange(0.0,prioritysum,step)
			fraction = 0.0#0.8
			low = 0.0#fraction*prioritysum 
			step = (prioritysum-low) / BATCH_SIZE
			try:
				randexp = np.arange(low,prioritysum,step)+np.random.uniform(low=0.0,high=step,size=(BATCH_SIZE))
			except Exception as e :
				print( prioritysum, step)
				raise e 
			# Sampling within each sub-interval with (un)trunc normal priority over the top :
			#randexp = np.random.normal(loc=0.75,scale=1.0,size=self.batch_size) * prioritysum

			
			batch = list()
			priorities = np.zeros(BATCH_SIZE)
			for i in range(BATCH_SIZE):
				try :
					el = memory.get(randexp[i])
					priorities[i] = el[1]
					batch.append(el)
				except TypeError as e :
					continue
					#print('REPLAY BUFFER EXCEPTION...')
			
			batch = TransitionPR( *zip(*batch) )
			
			# Create Batch with replayMemory :
			#transitions = memory.sample(BATCH_SIZE)
			#batch = Transition(*zip(*transitions) )

			beta = 1.0
			priorities = Variable( torch.from_numpy(priorities ), requires_grad=False).float()
			importanceSamplingWeights = torch.pow( len(memory) * priorities , -beta)
			
			next_state_batch = Variable(torch.cat( batch.next_state), requires_grad=False)
			state_batch = Variable( torch.cat( batch.state) , requires_grad=False)
			action_batch = Variable( torch.cat( batch.action) , requires_grad=False)
			reward_batch = Variable( torch.cat( batch.reward ), requires_grad=False ).view((-1,1))
			done_batch = [ 0.0 if batch.done[i] else 1.0 for i in range(len(batch.done)) ]
			done_batch = Variable( torch.FloatTensor(done_batch), requires_grad=False ).view((-1,1))
			
			if use_cuda :
				importanceSamplingWeights = importanceSamplingWeights.cuda()
				next_state_batch = next_state_batch.cuda()
				state_batch = state_batch.cuda()
				action_batch = action_batch.cuda()
				reward_batch = reward_batch.cuda()
				done_batch = done_batch.cuda()

			state_action_values = model(state_batch)
			state_action_values_g = state_action_values.gather(1,action_batch)

			############################
			next_state_values = model_(next_state_batch)
			next_state_values = Variable(next_state_values.cpu().data.max(1)[0]).view((-1,1))
			############################
			# Compute the expected Q values
			gamma_next = (next_state_values * GAMMA).type(FloatTensor)
			expected_state_action_values = done_batch*gamma_next + reward_batch

			# Compute Huber loss
			#loss = F.smooth_l1_loss(state_action_values_g, expected_state_action_values)
			diff = state_action_values_g - expected_state_action_values
			diff_squared = diff*diff
			is_diff_squared = importanceSamplingWeights * diff_squared
			loss = torch.mean( is_diff_squared)
			#loss = nn.MSELoss(state_action_values_g, expected_state_action_values)
			
			# Optimize the model
			# we do not zero the worker's model's gradient since it is used as an accumulator for gradient.
			# The worker's model's gradient accumulator is being zero-ed after being applied to the model,
			# when the function from_worker2model is called.
			loss.backward()
			
			
		except Exception as e :
			#TODO : find what is the reason of this error in backward...
			#"leaf variable was used in an inplace operation."
			bashlogger.exception('Error in optimizer_model : {}'.format(e) )
			
		for param in model.parameters():
		    if param.grad is not None :
		    	if param.grad.data is not None :
		    		param.grad.data.clamp_(-1, 1)

		
		#UPDATE THE PR :
		loss_np = loss.cpu().data.numpy()
		for (idx, new_error) in zip(batch.idx,loss_np) :
			new_priority = memory.priority(new_error)
			#print( 'prior = {} / {}'.format(new_priority,self.rBuffer.total()) )
			memory.update(idx,new_priority)

		del batch 
		del next_state_batch
		del state_batch
		del action_batch
		del reward_batch
		

		return loss_np


	def train(self,model,env,memory,optimizer,logger=None,preprocess=T.ToTensor(),path=None,frompath=None,num_episodes=1000,epsend=0.05,epsstart=0.9,epsdecay=10): 
		try :
			episode_durations = []
			episode_reward = []
			episode_loss = []
			global rendering
			global use_cuda
			global MAX_STEPS
			#exploration counter ;
			steps_done = [0]
			
			#Double Network initialization :
			savemodel(model,path+'.save')
			#model_ = DuelingDQN(model.nbr_actions)
			model_ = copy.deepcopy(model)
			hard_update(model_,model)
			model_.eval()
			
			if use_cuda :
				model_ = model_.cuda()
				
			for i in range(num_episodes) :
				bashlogger.info('Episode : {} : memory : {}/{}'.format(i,len(memory),memory.capacity) )
				cumul_reward = 0.0
				last_screen = get_screen_reset(env,preprocess=preprocess)
				current_screen, reward, done, info = get_screen(env,env.action_space.sample(),preprocess=preprocess )
				state = current_screen - last_screen
				
				episode_buffer = []
				meanfreq = 0
				episode_loss_buffer = []
				episode_qsa_buffer = []

				
				showcount = 0
				for t in count() :
					
					action,qsa = select_action(model,state,steps_done=steps_done,epsend=epsend,epsstart=epsstart,epsdecay=epsdecay)
					episode_qsa_buffer.append(qsa)
					last_screen = current_screen
					current_screen, reward, done, info = get_screen(env,action[0,0],preprocess=preprocess)
					cumul_reward += reward

					if rendering :
						if showcount >= 10 :
							showcount = 0
							render(current_screen)#env.render()
						else :
							showcount +=1
					
					reward = FloatTensor([reward])

					if not done :
						next_state = current_screen#-last_screen
					else :
						next_state = torch.zeros(current_screen.size())

					episode_buffer.append( EXP(state,action,next_state,reward,done) )

					state = next_state

					# OPTIMIZE MODEL :
					since = time.time()		
					lossnp = self.optimize_model(model,model_,memory,optimizer)
					if lossnp is not None :
						episode_loss_buffer.append(  np.mean(lossnp) )
					else :
						episode_loss_buffer.append(0)
						
					# SOFT UPDATE :
					soft_update(model_,model,self.TAU)
				
					elt = time.time() - since
					f = 1.0/elt
					meanfreq = (meanfreq*(t+1) + f)/(t+2)
					
					if done or t > MAX_STEPS:
						self.from_worker2model()

						'''
						nbrTrain = 2
						for tr in range(nbrTrain) :
							since = time.time()		
							lossnp = optimize_model(model,model_,memory,optimizer)
							if lossnp is not None :
								episode_loss_buffer.append(  np.mean(lossnp) )
							else :
								episode_loss_buffer.append(0)
								
							elt = time.time() - since
							f = 1.0/elt
							meanfreq = (meanfreq*(tr+1) + f)/(tr+2)
							#print('{} Hz ; {} seconds.'.format(f,elt) )
						'''	
						episode_durations.append(t+1)
						episode_reward.append(cumul_reward)
						meanloss = np.mean(episode_loss_buffer)
						episode_loss.append(meanloss)
						meanqsa = np.mean(episode_qsa_buffer)


						log = 'Episode duration : {}'.format(t+1) +'---' +'Cum Reward : {} // Mean Loss : {} // QSA : {}'.format(cumul_reward,meanloss,meanqsa) +'---'+' {}Hz'.format(meanfreq)
						bashlogger.info(log)
						if logger is not None :
							new = {'episodes':[i],'duration':[t+1],'cum reward':[cumul_reward],'mean frequency':[meanfreq],'loss':[meanloss],'meanQSA':[meanqsa]}
							logger.append(new)

						if path is not None :
							# SAVE THE MAIN MODEL :
							self.model.lock()
							savemodel(self.model,path+'.save')
							self.model.unlock()
							bashlogger.info('Model saved : {}'.format(path) )
						#plot_durations()
						break


				#Let us add this episode_buffer to the replayBuffer :
				for el in episode_buffer :
					init_priority = memory.priority( torch.abs(el.reward).cpu().numpy() )
					memory.add(el,init_priority)
				del episode_buffer

			bashlogger.info('Complete')
			if path is not None :
				savemodel(model,path+'.save')
				bashlogger.info('Model saved : {}'.format(path) )
			
			env.close()
		except Exception as e :
			bashlogger.exception(e)


def savemodel(model,path='./modelRL.save') :
	torch.save( model.state_dict(), path)

def loadmodel(model,path='./modelRL.save') :
	model.load_state_dict( torch.load(path) )


def main():
	global nbr_actions
	#env = 'SpaceInvaders-v0'#gym.make('SpaceInvaders-v0')#.unwrapped
	#nbr_actions = 6
	#env = 'Breakout-v0'#gym.make('Breakout-v0')#.unwrapped
	#nbr_actions = 4
	#useCNN = {'use':True,'dim':3}

	#env = 'Acrobot-v1'
	env = 'CartPole-v1'
	#env = 'MountainCar-v0'
	task = gym.make(env)
	nbr_actions = task.action_space.n
	useCNN = {'use':False,'dim':task.observation_space.shape[0]}

	task.reset()
	#task.render()
	task.close()
	
	if useCNN['use']:
		preprocess = T.Compose([T.ToPILImage(),
					T.Scale(64, interpolation=Image.CUBIC),
					T.ToTensor() ] )
	else :
		preprocess = T.Compose([
					T.ToTensor() ] )

	last_sync = 0
	
	numep = 1000
	global BATCH_SIZE
	BATCH_SIZE = 256
	global GAMMA
	GAMMA = 0.999
	TAU = 1e-2
	global MIN_MEMORY
	MIN_MEMORY = 1e3
	EPS_START = 0.999
	EPS_END = 0.3
	EPS_DECAY = 10000
	#alphaPER = 0.5
	alphaPER = 0.8
	global lr
	lr = 1e-3
	memoryCapacity = 25e3
	#num_worker = 8
	num_worker = 4
	#num_worker = 2
	#num_worker = 1

	#model_path = './'+env+'::CNN+DuelingDoubleDQN+PR-alpha'+str(alphaPER)+'-w'+str(num_worker)+'-lr'+str(lr)+'-b'+str(BATCH_SIZE)+'-m'+str(memoryCapacity)+'/'
	model_path = './'+env+'::CNN+DuelingDoubleDQN+WithZG+GAMMA{}+TAU{}'.format(GAMMA,TAU)\
	+'+IS+PER-alpha'+str(alphaPER) \
	+'+RewardScaler{}'.format(REWARD_SCALER)\
	+'-w'+str(num_worker)+'-lr'+str(lr)+'-b'+str(BATCH_SIZE)+'-m'+str(memoryCapacity)+'/'
	
	#+'+PER-alpha'+str(alphaPER) \
	#+'+TruePR-alpha'+str(alphaPER)\
	

	#mkdir :
	if not os.path.exists(model_path) :
		os.mkdir(model_path)
	path=model_path+env
	frompath = None

	savings =  [ p for p in os.listdir(model_path) if ('save' in p)==True ]
	if len(savings) :
		frompath = os.path.join(model_path,savings[0])


	#model = DQN(nbr_actions)
	model = DuelingDQN(nbr_actions,useCNN=useCNN)
	model.share_memory()
	bashlogger.info('Model : created.')
	if frompath is not None :
		loadmodel(model,frompath)
		bashlogger.info('Model loaded: {}'.format(frompath))

	if use_cuda :
		bashlogger.info('Model : CUDA....')
		model = model.cuda()
		bashlogger.info('Model : CUDA : ok.')

	memory = PrioritizedReplayBuffer(capacity=memoryCapacity,alpha=alphaPER)
	bashlogger.info('Memory : ok.')


	evaluation = False
	training = True

	if training :
		workers = []
		for i in range(num_worker) :
			worker = Worker(i,model,env,memory,lr=lr,preprocess=preprocess,path=path,frompath=frompath,num_episodes=numep,epsend=EPS_END,epsstart=EPS_START,epsdecay=EPS_DECAY,TAU=TAU)
			workers.append(worker)
			time.sleep(1)
			worker.start()
		
		
	if evaluation :
		task = gym.make(env)
		task.reset()
		for ep in range(numep) :
			dummyaction = 0
			cumr = 0.0
			done = False
			last_screen = get_screen_reset(task,preprocess=preprocess)
			current_screen, reward, done, info = get_screen(task,dummyaction,preprocess=preprocess)
			state = current_screen -last_screen
			
			nbrsteps = 0

			while not done :
				action, qsa = exploitation(model,state)
				last_screen = current_screen
				current_screen, reward, done, info = get_screen(task,action[0,0],preprocess=preprocess)
				reward = reward/REWARD_SCALER
				cumr += reward

				task.render()
				print('QSA : {}'.format(qsa))

				if done or (nbrsteps >= MAX_STEPS/5) :
					done = True
					task.reset()
					next_state = torch.zeros(current_screen.size())
					print('EVALUATION : EPISODE {} : cum reward = {} // steps = {} // DONE : {}'.format(ep,cumr, nbrsteps,done))
				else :
					next_state = current_screen -last_screen
					#print('step {}/{} : action = {}'.format(nbrsteps,MAX_STEPS, action))
				
				state = next_state
				nbrsteps +=1 


	if training :
		for i in range(num_worker) :
			try :
				workers[i].join()
			except Exception as e :
				bashlogger.info(e)

	

if __name__ == "__main__":
	main()


