import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import copy
import logging
bashlogger = logging.getLogger("bash logger")
bashlogger.setLevel(logging.DEBUG)
FORMAT = '[%(asctime)-15s][%(threadName)s][%(levelname)s][%(funcName)s] %(message)s'
logging.basicConfig(format=FORMAT)


from NN import ActorCriticNN
from utils.utils import soft_update, hard_update, OrnsteinUhlenbeckNoise


TAU = 1e-3
GAMMA = 0.99
LR = 1e-3
USE_CUDA = True
BATCH_SIZE = 256


class Model :
	def __init__(self, NN, memory, algo='ddpg',GAMMA=GAMMA,LR=LR,TAU=TAU,use_cuda=USE_CUDA,BATCH_SIZE=BATCH_SIZE ) :
		self.NN = NN
		self.target_NN = copy.deepcopy(NN)

		self.use_cuda = use_cuda
		if self.use_cuda :
			self.NN = self.NN.cuda()
			self.target_NN = self.target_NN.cuda()

		self.memory = memory

		self.gamma = GAMMA
		self.lr = LR
		self.tau = TAU
		self.batch_size = BATCH_SIZE

		self.optimizer = optim.Adam(self.NN.parameters(), self.lr)

		self.noise = OrnsteinUhlenbeckNoise(self.action_dim)

		hard_update(self.NN, self.target_NN)

		self.algo = algo

	
	def act(self, x,exploitation=False) :
		state = Variable( torch.from_numpy(x), volatile=True )
		if self.use_cuda :
			state = state.cuda()
		action = self.NN.actor( state).detach()
		
		if exploitation :
			return action.cpu().data.numpy()
		else :
			# exploration :
			new_action = action.cpu().data.numpy() + self.noise.sample()*self.NN.action_scaler
			return new_action

	def optimize(self,MIN_MEMORY=1e3) :

		if self.algo == 'ddpg' :
			try :
				if len(self.memory) < MIN_MEMORY :
					return
				
				#Create Batch with PR :
				prioritysum = self.memory.total()
				randexp = np.random.random(size=self.batch_size)*prioritysum
				batch = list()
				for i in range(self.batch_size):
					try :
						el = self.memory.get(randexp[i])
						batch.append(el)
					except TypeError as e :
						continue
						#print('REPLAY BUFFER EXCEPTION...')
				
				# Create Batch with replayMemory :
				batch = TransitionPR( *zip(*batch) )
				next_state_batch = Variable(torch.cat( batch.next_state), requires_grad=False)
				state_batch = Variable( torch.cat( batch.state) , requires_grad=False)
				action_batch = Variable( torch.cat( batch.action) , requires_grad=False)
				reward_batch = Variable( torch.cat( batch.reward ), requires_grad=False ).view((-1,1))
				
				if self.use_cuda :
					next_state_batch = next_state_batch.cuda()
					state_batch = state_batch.cuda()
					action_batch = action_batch.cuda()
					reward_batch = reward_batch.cuda()

				#before optimization :
				self.optimizer.zero_grad()
				
				# Critic :
				# sample action from next_state, without gradient repercusion :
				next_action = self.NN.act(next_state_batch).detach()
				# evaluate the next state action over the target, without repercusion (faster...) :
				next_qsa = torch.squeeze( self.target_NN( next_state_batch, next_action).detach() )
				# Supervise loss :
				## y_true :
				y_true = reward_batch + self.gamma*next_qsa 
				## y_pred :
				y_pred = torch.squeeze( self.NN.critic(state_batch,action_batch) )
				## loss :
				critic_loss = F.smooth_l1_loss(y_true,y_pred)
				critic_loss.backward()
				#self.optimizer.step()

				# Actor :
				pred_action = self.NN.actor(state_batch)
				pred_qsa = torch.squeeze( self.NN.critic(state_batch, pred_action) )
				# loss :
				actor_loss = -1.0*torch.sum( pred_qsa)
				actor_loss.backward()
				#self.optimizer.step()

				# optimize both pathway :
				self.optimizer.step()

			except Exception as e :
				bashlogger.debug(e)


			# soft update :
			soft_update(self.target_NN, self.NN, self.tau)

			return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()

		else :
			raise NotImplemented

	def save(self,path) :
		torch.save( self.NN.state_dict(), path)

	def load(self, path) :
		self.NN.load_state_dict( torch.load(path) )
		hard_update(self.target_NN, self.NN)