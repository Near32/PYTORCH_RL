import gym
import math
import random
import os
import numpy as np
import matplotlib
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
'''
def Variable(data, volatile=False):
	if USE_CUDA:
		return autograd.Variable(data.cuda(),volatile=volatile)
	else:
		return autograd.Variable(data, volatile=volatile)
'''
import torchvision.transforms as T

use_cuda = False#torch.cuda.is_available()
rendering = False


FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

Transition = namedtuple('Transition', ('state','action','next_state', 'reward') )

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

class DQN(nn.Module) :
	def __init__(self,nbr_actions=2) :
		super(DQN,self).__init__()
		self.nbr_actions = nbr_actions

		self.conv1 = nn.Conv2d(3,16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)
		#self.head = nn.Linear(448,self.nbr_actions)
		self.head = nn.Linear(192,self.nbr_actions)

	def forward(self, x) :
		x = F.relu( self.bn1(self.conv1(x) ) )
		x = F.relu( self.bn2(self.conv2(x) ) )
		x = F.relu( self.bn3(self.conv3(x) ) )
		x = x.view( x.size(0), -1)
		x = self.head( x )
		return x


def get_screen(env,action,preprocess) :
	screen, reward, done, info = env.step(action)
	screen = screen.transpose( (2,0,1) )
	screen = np.ascontiguousarray( screen, dtype=np.float32) / 255.0
	screen = torch.from_numpy(screen)
	screen = preprocess(screen)
	screen = screen.unsqueeze(0)
	#screen = screen.type(Tensor)
	return screen, reward, done, info

def get_screen_reset(env,preprocess) :
	screen = env.reset()
	screen = screen.transpose( (2,0,1) )
	screen = np.ascontiguousarray( screen, dtype=np.float32) / 255.0
	screen = torch.from_numpy(screen)
	screen = preprocess(screen)
	screen = screen.unsqueeze(0)
	return screen

def test1() :
	plt.figure()
	plt.imshow( get_screen(env.action_space.sample()).cpu().squeeze(0).permute( 1, 2 ,0 ).numpy(), interpolation='none')
	plt.title('Example extracted screen')
	plt.show()


def select_action(model,state,epsend=0.05,epsstart=0.9,epsdecay=200) :
	global steps_done
	global nbr_actions
	sample = random.random()
	eps_threshold = epsend + (epsstart-epsend) * math.exp(-1.0 * steps_done / epsdecay )
	steps_done +=1

	if sample > eps_threshold :
		return model( Variable(state, volatile=True).type(FloatTensor) ).data.max(1)[1].view(1,1)
	else :
		return LongTensor( [[random.randrange(nbr_actions) ] ] )


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


def optimize_model(model,memory,optimizer) :
	global last_sync
	global use_cuda
	
	if len(memory) < BATCH_SIZE :
		return
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions) )

	#non_final_mask = ByteTensor( tuple(map(lambda s: s is not None, batch.next_state) ) )
	# We don't want to backprop through the expected action values and volatile
	# will save us on temporarily changing the model parameters'
	# requires_grad to False!
	#non_final_next_states = Variable(torch.cat([s for s in batch.next_state
	#                                        if s is not None]),
	#                             volatile=True)
	try :
		next_state_batch = Variable(torch.cat( batch.next_state))
	except Exception as e :
		print(e)
	state_batch = Variable( torch.cat( batch.state) )
	action_batch = Variable( torch.cat( batch.action) )
	reward_batch = Variable( torch.cat( batch.reward ) )
	
	if use_cuda :
		#non_final_mask = non_final_mask.cuda()
		#non_final_next_states = non_final_next_states.cuda()
		next_state_batch = next_state_batch.cuda()
		state_batch = state_batch.cuda()
		action_batch = action_batch.cuda()
		reward_batch = reward_batch.cuda()

	state_action_values = model(state_batch)
	#print(state_action_values.size())
	state_action_values = state_action_values.gather(1,action_batch)

	# Compute V(s_{t+1}) for all next states.
	'''
	next_state_values = Variable(torch.zeros(BATCH_SIZE))#.type(Tensor))
	if use_cuda :
		nex_state_values = next_state_values.cuda()

	next_state_values[non_final_mask.cpu()] = model(non_final_next_states).cpu()
	print(next_state_values)
	next_state_values = next_state_values.max(1)[0]
	'''
	#next_state_values = Variable(torch.zeros(BATCH_SIZE))#.type(Tensor))
	#if use_cuda :
	#	nex_state_values = next_state_values.cuda()

	#next_state_values = model(non_final_next_states)
	next_state_values = model(next_state_batch)
	next_state_values = next_state_values.max(1)[0]
	
	# Now, we don't want to mess up the loss with a volatile flag, so let's
	# clear it. After this, we'll just end up with a Variable that has
	# requires_grad=False
	#next_state_values.volatile = False
	# Compute the expected Q values
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	# Compute Huber loss
	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

	# Optimize the model
	optimizer.zero_grad()
	loss.backward()
	for param in model.parameters():
	    param.grad.data.clamp_(-1, 1)
	optimizer.step()

	del batch 


def render(frame) :
	if use_cuda :
		plt.imshow( frame.cpu().squeeze(0).permute( 1, 2 ,0 ).numpy(), interpolation='none')
	else :
		plt.imshow( frame.squeeze(0).permute( 1, 2 ,0 ).numpy(), interpolation='none')
	plt.title('Current Frame')
	plt.pause(1e-4)

def train(model,env,memory,optimizer,preprocess=T.ToTensor(),path=None,frompath=None,num_episodes=1000,epsend=0.05,epsstart=0.9,epsdecay=200): 
	episode_durations = []
	episode_reward = []
	global rendering
	#exploration counter ;
	global steps_done
	steps_done = 0
	
	if frompath is not None :
		loadmodel(model,frompath)
		print('Model loaded: {}'.format(frompath))

	
	for i in range(num_episodes) :
		print('Episode : {} : memory : {}/{}'.format(i,len(memory),memory.capacity) )
		cumul_reward = 0.0
		last_screen = get_screen_reset(env,preprocess=preprocess)
		current_screen, reward, done, info = get_screen(env,env.action_space.sample(),preprocess=preprocess )
		state = current_screen - last_screen
		for t in count() :
			action = select_action(model,state,epsend=epsend,epsstart=epsstart,epsdecay=epsdecay)
			last_screen = current_screen
			current_screen, reward, done, info = get_screen(env,action[0,0],preprocess=preprocess)
			cumul_reward += reward

			if rendering :
				env.render()
			reward = Tensor([reward])

			if not done :
				next_state = current_screen -last_screen
			else :
				next_state = torch.zeros(current_screen.size())

			memory.push( state, action, next_state, reward)

			state = next_state

			since = time.time()
			optimize_model(model,memory,optimizer)
			elt = time.time() - since
			f = 1.0/elt
			#print('{} Hz ; {} seconds.'.format(f,elt) )
			
			if done :
				episode_durations.append(t+1)
				episode_reward.append(cumul_reward)
				print('Epoch duration : {}'.format(t+1) )
				print('Cumulative Reward : {}'.format(cumul_reward) )
				if path is not None :
					savemodel(model,path)
					print('Model saved : {}'.format(path) )
				#plot_durations()
				break

	print('Complete')
	if path is not None :
		savemodel(model,path)
		print('Model saved : {}'.format(path) )
	
	env.close()
	plt.ioff()
	plt.show()

def savemodel(model,path='./modelRL.save') :
	torch.save( model.state_dict(), path)

def loadmodel(model,path='./modelRL.save') :
	model.load_state_dict( torch.load(path) )


def main():
	#env = gym.make('SpaceInvaders-v0')#.unwrapped
	env = gym.make('Breakout-v0')#.unwrapped
	global nbr_actions
	nbr_actions = 4
	env.reset()

	resize = T.Compose([T.ToPILImage(),
					T.Scale(40, interpolation=Image.CUBIC),
					T.ToTensor() ] )

	last_sync = 0
	path='modelRL.save'
	frompath = None

	if path in os.listdir('./') :
		frompath = path

	numep = 2000
	global BATCH_SIZE
	BATCH_SIZE = 128
	global GAMMA
	GAMMA = 0.999
	EPS_START = 0.9
	EPS_END = 0.05
	EPS_DECAY = 200

	model = DQN(nbr_actions)
	print('Model : created.')
	if use_cuda :
		print('Model : CUDA....')
		model = model.cuda()
		print('Model : CUDA : ok.')

	optimizer = optim.RMSprop(model.parameters() )
	print('Optimizer : ok.')
	memory = ReplayMemory(1e4)
	print('Memory : ok.')

	train(model,env,memory,optimizer,preprocess=resize,path=path,frompath=frompath,num_episodes=numep,epsend=EPS_END,epsstart=EPS_START,epsdecay=EPS_DECAY)


if __name__ == "__main__":
	main()


