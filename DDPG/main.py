import os
import time

from worker import Worker
from utils.replayBuffer import EXP,PrioritizedReplayBuffer
from utils.statsLogger import statsLogger

from model import Model,Model2
from NN import ActorNN,CriticNN,ActorCriticNN,ActorCriticNN2

import torch
import torchvision.transforms as T

import logging
bashlogger = logging.getLogger("bash logger")
bashlogger.setLevel(logging.DEBUG)
FORMAT = '[%(asctime)-15s][%(threadName)s][%(levelname)s][%(funcName)s] %(message)s'
logging.basicConfig(format=FORMAT)



use_cuda = True#torch.cuda.is_available()


'''
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
'''

def main():
	#env = 'SpaceInvaders-v0'#gym.make('SpaceInvaders-v0')#.unwrapped
	#nbr_actions = 6
	#env = 'Breakout-v0'#gym.make('Breakout-v0')#.unwrapped
	#nbr_actions = 4
	#input_size = 3
	
	env = 'Pendulum-v0'#gym.make('Breakout-v0')#.unwrapped
	action_dim = 1
	input_dim = 3
	action_scaler = 2.0
	'''
	env = 'MountainCarContinuous-v0'#gym.make('Breakout-v0')#.unwrapped
	action_dim = 1
	input_dim = 2
	action_scaler = 100.0
	'''

	'''
	preprocess = T.Compose([T.ToPILImage(),
					#T.Scale(84, interpolation=Image.CUBIC),
					#T.Scale(50, interpolation=Image.CUBIC),
					T.Scale(30, interpolation=Image.CUBIC),
					T.ToTensor() ] )
	'''
	preprocess = T.Compose([T.ToTensor() ] )
	
	last_sync = 0
	
	numep = 20000
	BATCH_SIZE = 128
	GAMMA = 0.9
	TAU = 1e-3
	MIN_MEMORY = 1e3
	use_cnn = False

	CNN = {'use_cnn':use_cnn, 'input_size':input_dim}

	alphaPER = 0.5
	
	lr = 1e-4
	memoryCapacity = 2e4
	
	num_worker = 1
	renderings = [False]*num_worker
	#renderings[0] = True
	
	#Dueling :
	dueling = False
	
	#HER :
	k = 2
	strategy = 'future'
	use_her = False
	singlegoal = False
	HER = {'k':k, 'strategy':strategy,'use_her':use_her,'singlegoal':singlegoal}

	global use_cuda

	envpath = './'+env+'/'
	model_path = envpath+env+'::'

	if dueling :
		model_path +='Dueling'
	
	model_path += 'DDPG+PR+'

	if HER['use_her'] :
		model_path += 'HER-alpha'+str(alphaPER)+'-k'+str(k)+strategy

	model_path += '-w'+str(num_worker)+'-lr'+str(lr)+'-b'+str(BATCH_SIZE)+'-m'+str(memoryCapacity)+'/'
	
	#mkdir :
	if not os.path.exists(envpath) :
		os.mkdir(envpath)
	if not os.path.exists(model_path) :
		os.mkdir(model_path)
	
	path=model_path+env
	frompath = None

	savings =  [ p for p in os.listdir(model_path) if ('save' in p)==True ]
	if len(savings) :
		frompath = os.path.join(model_path,savings[0])


	memory = PrioritizedReplayBuffer(capacity=memoryCapacity,alpha=alphaPER)
	bashlogger.info('Memory : ok.')

	
	#################################################
	#################################################
	'''
	#actorcritic = ActorCriticNN(state_dim=input_dim,action_dim=action_dim,action_scaler=action_scaler,dueling=dueling,CNN=CNN,HER=HER['use_her'])
	actorcritic = ActorCriticNN2(state_dim=input_dim,action_dim=action_dim,action_scaler=action_scaler,dueling=dueling,CNN=CNN,HER=HER['use_her'])
	bashlogger.info('Models : creation ...')
	actorcritic.share_memory()

	model = Model(NN=actorcritic,memory=memory,GAMMA=GAMMA,LR=lr,TAU=TAU,use_cuda=use_cuda,BATCH_SIZE=BATCH_SIZE)
	bashlogger.info('Models : created.')
	if frompath is not None :
		model.load(frompath)
		bashlogger.info('Models loaded: {}'.format(frompath))
	'''
	#################################################
	#################################################
	
	actor = ActorNN(state_dim=input_dim,action_dim=action_dim,action_scaler=action_scaler,CNN=CNN,HER=HER['use_her'])
	actor.share_memory()
	critic = CriticNN(state_dim=input_dim,action_dim=action_dim,dueling=dueling,CNN=CNN,HER=HER['use_her'])
	critic.share_memory()

	model = Model2(actor=actor,critic=critic,memory=memory,GAMMA=GAMMA,LR=lr,TAU=TAU,use_cuda=use_cuda,BATCH_SIZE=BATCH_SIZE)
	
	bashlogger.info('Models : created.')
	if frompath is not None :
		if 'actor' in frompath :
			frompath = frompath[:-6]
		elif 'critic' in frompath :
			frompath = frompath[:-7]
		model.load(frompath)
		bashlogger.info('Models loaded: {}'.format(frompath))
	
	#################################################
	#################################################
	
	workers = []
	for i in range(num_worker) :
		worker = Worker(i,model,env,memory,preprocess=preprocess,path=path,frompath=frompath,num_episodes=numep,HER=HER,use_cuda=use_cuda,rendering=renderings[i])
		workers.append(worker)
		time.sleep(1)
		worker.start()

	for i in range(num_worker) :
		try :
			workers[i].join()
		except Exception as e :
			bashlogger.info(e)

if __name__ == "__main__":
	main()