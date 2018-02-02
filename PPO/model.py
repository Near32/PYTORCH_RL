import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

from threading import Lock
import numpy as np

import copy
import logging
bashlogger = logging.getLogger("bash logger")
bashlogger.setLevel(logging.DEBUG)
FORMAT = '[%(asctime)-15s][%(threadName)s][%(levelname)s][%(funcName)s] %(message)s'
logging.basicConfig(format=FORMAT)


from NN import ActorCriticNN
from utils.utils import soft_update, hard_update, OrnsteinUhlenbeckNoise
from utils.replayBuffer import EXP,TransitionPR,PrioritizedReplayBuffer, ReplayMemory

TAU = 1e-3
GAMMA = 0.99
LR = 1e-3
USE_CUDA = False
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

		self.noise = OrnsteinUhlenbeckNoise(self.NN.action_dim)

		hard_update(self.target_NN,self.NN )

		self.algo = algo

	
	def act(self, x,exploitation=False) :
		#state = Variable( torch.from_numpy(x), volatile=True )
		state = Variable( x, volatile=True )
		if self.use_cuda :
			state = state.cuda()
		action = self.NN.actor( state).detach()
		
		if exploitation :
			return action.cpu().data.numpy()
		else :
			# exploration :
			new_action = action.cpu().data.numpy() + self.noise.sample()*self.NN.action_scaler
			return new_action

	def evaluate(self, x,a) :
		state = Variable( x, volatile=True )
		action = Variable( a, volatile=True )
		if self.use_cuda :
			state = state.cuda()
			action = action.cuda()

		qsa = self.NN.critic( state, action).detach()
		
		return qsa.cpu().data.numpy()

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
				next_taction = self.target_NN.actor(next_state_batch).detach()
				# evaluate the next state action over the target, without repercusion (faster...) :
				next_tqsa = torch.squeeze( self.target_NN.critic( next_state_batch, next_taction).detach() )
				# Supervise loss :
				## y_true :
				y_true = reward_batch + self.gamma*next_tqsa 
				## y_pred :
				y_pred = torch.squeeze( self.NN.critic(state_batch,action_batch) )
				## loss :
				critic_loss = F.smooth_l1_loss(y_pred,y_true)
				#critic_loss.backward()
				#self.optimizer.step()

				# Actor :
				pred_action = self.NN.actor(state_batch)
				pred_qsa = torch.squeeze( self.target_NN.critic(state_batch, pred_action) )
				# loss :
				actor_loss = -1.0*torch.sum( pred_qsa)
				#actor_loss.backward()
				#self.optimizer.step()

				# optimize both pathway :
				scalerA = 0.1
				scalerV = 10.0
				total_loss = scalerA*actor_loss + scalerV*critic_loss
				total_loss.backward()
				self.optimizer.step()

			except Exception as e :
				bashlogger.debug('error : {}',format(e) )
				

			# soft update :
			soft_update(self.target_NN, self.NN, self.tau)

			return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()

		else :
			raise NotImplemented


	def optimizeSEPARATED(self,MIN_MEMORY=1e3) :

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
				'''
				next_state_batch = Variable(torch.cat( batch.next_state) )
				state_batch = Variable( torch.cat( batch.state) )
				action_batch = Variable( torch.cat( batch.action) )
				reward_batch = Variable( torch.cat( batch.reward ) ).view((-1,1))
				'''
				
				if self.use_cuda :
					next_state_batch = next_state_batch.cuda()
					state_batch = state_batch.cuda()
					action_batch = action_batch.cuda()
					reward_batch = reward_batch.cuda()

				
				# Critic :
				# sample action from next_state, without gradient repercusion :
				next_taction = self.target_NN.actor(next_state_batch).detach()
				# evaluate the next state action over the target, without repercusion (faster...) :
				next_tqsa = torch.squeeze( self.target_NN.critic( next_state_batch, next_taction).detach() )
				# Supervise loss :
				## y_true :
				y_true = reward_batch + self.gamma*next_tqsa 
				## y_pred :
				y_pred = torch.squeeze( self.NN.critic(state_batch,action_batch) )
				## loss :
				critic_loss = F.smooth_l1_loss(y_true,y_pred)
				#before optimization :
				self.optimizer.zero_grad()
				critic_loss.backward()
				self.optimizer.step()

				
				
				# Actor :
				pred_action = self.NN.actor(state_batch)
				pred_qsa = torch.squeeze( self.NN.critic(state_batch, pred_action) )
				# loss :
				actor_loss = -1.0*torch.sum( pred_qsa)
				#before optimization :
				self.optimizer.zero_grad()
				actor_loss.backward()
				self.optimizer.step()
				
			except Exception as e :
				bashlogger.debug('error : {}',format(e) )


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


class ModelDistributed :
	def __init__(self, NN, memory, algo='ddpg',GAMMA=GAMMA,LR=LR,TAU=TAU,use_cuda=USE_CUDA,BATCH_SIZE=BATCH_SIZE,MIN_MEMORY=1e3 ) :
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
		self.MIN_MEMORY = MIN_MEMORY

		self.mutex = Lock()
		
		self.noise = OrnsteinUhlenbeckNoise(self.NN.action_dim)

		hard_update(self.target_NN, self.NN )

		self.algo = algo

	def generate_optimizers(self) :
		optimizer = optim.Adam(self.NN.parameters(), self.lr)
		
		return {'critic':optimizer,'actor':optimizer}


	def act(self, x,exploitation=False) :
		#state = Variable( torch.from_numpy(x), volatile=True )
		state = Variable( x, volatile=True )
		if self.use_cuda :
			state = state.cuda()
		action = self.NN.actor( state).detach()
		
		if exploitation :
			return action.cpu().data.numpy()
		else :
			# exploration :
			new_action = action.cpu().data.numpy() + self.noise.sample()*self.NN.action_scaler
			return new_action

	def evaluate(self, x,a) :
		state = Variable( x, volatile=True )
		action = Variable( a, volatile=True )
		if self.use_cuda :
			state = state.cuda()
			action = action.cuda()

		qsa = self.NN.critic( state, action).detach()
		
		return qsa.cpu().data.numpy()


	def optimizeSEPARATED(self,MIN_MEMORY=1e3) :

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
				'''
				next_state_batch = Variable(torch.cat( batch.next_state) )
				state_batch = Variable( torch.cat( batch.state) )
				action_batch = Variable( torch.cat( batch.action) )
				reward_batch = Variable( torch.cat( batch.reward ) ).view((-1,1))
				'''
				
				if self.use_cuda :
					next_state_batch = next_state_batch.cuda()
					state_batch = state_batch.cuda()
					action_batch = action_batch.cuda()
					reward_batch = reward_batch.cuda()

				
				# Critic :
				# sample action from next_state, without gradient repercusion :
				next_taction = self.target_NN.actor(next_state_batch).detach()
				# evaluate the next state action over the target, without repercusion (faster...) :
				next_tqsa = torch.squeeze( self.target_NN.critic( next_state_batch, next_taction).detach() )
				# Supervise loss :
				## y_true :
				y_true = reward_batch + self.gamma*next_tqsa 
				## y_pred :
				y_pred = torch.squeeze( self.NN.critic(state_batch,action_batch) )
				## loss :
				critic_loss = F.smooth_l1_loss(y_true,y_pred)
				#before optimization :
				self.optimizer.zero_grad()
				critic_loss.backward()
				self.optimizer.step()

				
				
				# Actor :
				pred_action = self.NN.actor(state_batch)
				pred_qsa = torch.squeeze( self.NN.critic(state_batch, pred_action) )
				# loss :
				actor_loss = -1.0*torch.sum( pred_qsa)
				#before optimization :
				self.optimizer.zero_grad()
				actor_loss.backward()
				self.optimizer.step()
				
			except Exception as e :
				bashlogger.debug('error : {}',format(e) )


			# soft update :
			soft_update(self.target_NN, self.NN, self.tau)

			return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()

		else :
			raise NotImplemented


	def optimize(self,optimizer_critic,optimizer_actor) :
		'''
		self.target_critic.eval()
		self.target_actor.eval()
		self.critic.train()
		self.actor.train()
		'''

		if self.algo == 'ddpg' :
			try :
				if len(self.memory) < self.MIN_MEMORY :
					return
				
				#Create Batch 
				self.mutex.acquire()
				
				if isinstance(self.memory, PrioritizedReplayBuffer) :
					#with PR :
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
				else :
					#with random RB :
					batch = self.memory.sample(self.batch_size)

				self.mutex.release()

				if len(batch) == 0 :
					return

				# Create Batch with replayMemory :
				batch = TransitionPR( *zip(*batch) )
				next_state_batch = Variable(torch.cat( batch.next_state), requires_grad=False)
				state_batch = Variable( torch.cat( batch.state) , requires_grad=False)
				action_batch = Variable( torch.cat( batch.action) , requires_grad=False)
				reward_batch = Variable( torch.cat( batch.reward ), requires_grad=False ).view((-1))
				'''
				next_state_batch = Variable(torch.cat( batch.next_state) )
				state_batch = Variable( torch.cat( batch.state) )
				action_batch = Variable( torch.cat( batch.action) )
				reward_batch = Variable( torch.cat( batch.reward ) ).view((-1,1))
				'''
				
				if self.use_cuda :
					next_state_batch = next_state_batch.cuda()
					state_batch = state_batch.cuda()
					action_batch = action_batch.cuda()
					reward_batch = reward_batch.cuda()

				
				# Critic :
				#before optimization :
				optimizer_critic.zero_grad()
				# sample action from next_state, without gradient repercusion :
				next_taction = self.target_NN.actor(next_state_batch).detach()
				# evaluate the next state action over the target, without repercusion (faster...) :
				next_tqsa = torch.squeeze( self.target_NN.critic( next_state_batch, next_taction).detach() ).view((-1))
				# Supervise loss :
				## y_true :
				y_true = reward_batch + self.gamma*next_tqsa 
				## y_pred :
				y_pred = torch.squeeze( self.NN.critic(state_batch,action_batch) )
				## loss :
				critic_loss = F.smooth_l1_loss(y_pred,y_true)
				#criterion = nn.MSELoss()
				#critic_loss = criterion(y_pred,y_true)
				critic_loss.backward()
				#clamping :
				torch.nn.utils.clip_grad_norm(self.NN.parameters(),50)				
				optimizer_critic.step()
				
				# Actor :
				#before optimization :
				optimizer_actor.zero_grad()
				#predict action :
				pred_action = self.NN.actor(state_batch)
				#predict associated qvalues :
				pred_qsa = self.NN.critic(state_batch, pred_action)
				# loss :
				actor_loss = -1.0*torch.mean(torch.sum( pred_qsa) )
				actor_loss.backward()
				#clamping :
				clampactor = 1.0/torch.min( [ 0.5, torch.max( [1e-4, torch.abs(torch.mean(critic_loss.cpu().data) ) ] ) ] )
				torch.nn.utils.clip_grad_norm(self.NN.parameters(),clampactor)				
				optimizer_actor.step()

				
				'''
				critic_grad = 0.0
				for p in self.critic.parameters() :
					critic_grad += np.mean(p.grad.cpu().data.numpy())
				print( 'Mean Critic Grad : {}'.format(critic_grad) )
				'''
				
				actor_grad = 0.0
				for p in self.NN.parameters() :
					actor_grad += np.max( np.abs(p.grad.cpu().data.numpy() ) )
				#print( 'Mean Actor Grad : {}'.format(actor_grad) )
				

				#UPDATE THE PR :
				if isinstance(self.memory, PrioritizedReplayBuffer) :
					self.mutex.acquire()
					loss = torch.abs(actor_loss) + torch.abs(critic_loss)
					#loss = torch.abs(actor_loss) #+ torch.abs(critic_loss)
					loss_np = loss.cpu().data.numpy()
					for (idx, new_error) in zip(batch.idx,loss_np) :
						new_priority = self.memory.priority(new_error)
						#print( 'prior = {} / {}'.format(new_priority,self.rBuffer.total()) )
						self.memory.update(idx,new_priority)
					self.mutex.release()

			except Exception as e :
				bashlogger.debug('error : {}',format(e) )
				print(len(batch) )
				print(batch[0])
				raise e

			# soft update :
			soft_update(self.target_NN, self.NN, self.tau)
			
			del batch
			del next_state_batch 
			del state_batch 
			del action_batch 
			del reward_batch 

			closs = critic_loss.cpu()
			aloss = actor_loss.cpu()
			del actor_loss
			del critic_loss

			return closs.data.numpy(), aloss.data.numpy(), actor_grad

		else :
			raise NotImplemented

	def save(self,path) :
		torch.save( self.NN.state_dict(), path)

	def load(self, path) :
		self.NN.load_state_dict( torch.load(path) )
		hard_update(self.target_NN, self.NN)



class Model2 :
	def __init__(self, actor, critic, memory, algo='ddpg',GAMMA=GAMMA,LR=LR,TAU=TAU,use_cuda=USE_CUDA,BATCH_SIZE=BATCH_SIZE,MIN_MEMORY=1e3 ) :
		self.actor = actor
		self.critic = critic
		self.target_actor = copy.deepcopy(actor)
		self.target_critic = copy.deepcopy(critic)

		self.use_cuda = use_cuda
		if self.use_cuda :
			self.actor = self.actor.cuda()
			self.target_actor = self.target_actor.cuda()
			self.critic = self.critic.cuda()
			self.target_critic = self.target_critic.cuda()


		self.memory = memory

		self.gamma = GAMMA
		self.lr = LR
		self.tau = TAU
		self.batch_size = BATCH_SIZE
		self.MIN_MEMORY = MIN_MEMORY

		self.optimizer_actor = optim.Adam(self.actor.parameters(), self.lr)
		self.optimizer_critic = optim.Adam(self.critic.parameters(), self.lr*1e1)

		self.noise = OrnsteinUhlenbeckNoise(self.actor.action_dim)

		hard_update(self.target_actor, self.actor)
		hard_update(self.target_critic, self.critic)

		self.algo = algo

	
	def act(self, x,exploitation=False) :
		#self.actor.eval()
		
		#state = Variable( torch.from_numpy(x), volatile=True )
		state = Variable( x, volatile=True )
		if self.use_cuda :
			state = state.cuda()
		action = self.actor( state).detach()
		
		if exploitation :
			return action.cpu().data.numpy()
		else :
			# exploration :
			new_action = action.cpu().data.numpy() + self.noise.sample()*self.actor.action_scaler
			return new_action

	def evaluate(self, x,a) :
		#self.critic.eval()
		
		state = Variable( x, volatile=True )
		action = Variable( a, volatile=True )
		if self.use_cuda :
			state = state.cuda()
			action = action.cuda()

		qsa = self.critic( state, action).detach()
		
		return qsa.cpu().data.numpy()

	def optimize(self) :
		'''
		self.target_critic.eval()
		self.target_actor.eval()
		self.critic.train()
		self.actor.train()
		'''

		if self.algo == 'ddpg' :
			try :
				if len(self.memory) < self.MIN_MEMORY :
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
				reward_batch = Variable( torch.cat( batch.reward ), requires_grad=False ).view((-1))
				'''
				next_state_batch = Variable(torch.cat( batch.next_state) )
				state_batch = Variable( torch.cat( batch.state) )
				action_batch = Variable( torch.cat( batch.action) )
				reward_batch = Variable( torch.cat( batch.reward ) ).view((-1,1))
				'''
				
				if self.use_cuda :
					next_state_batch = next_state_batch.cuda()
					state_batch = state_batch.cuda()
					action_batch = action_batch.cuda()
					reward_batch = reward_batch.cuda()

				
				# Actor :
				#before optimization :
				self.optimizer_actor.zero_grad()
				#predict action :
				pred_action = self.actor(state_batch)
				#predict associated qvalues :
				pred_qsa = self.critic(state_batch, pred_action)
				# loss :
				actor_loss = -1.0*torch.mean(torch.sum( pred_qsa) )
				actor_loss.backward()
				#clamping :
				torch.nn.utils.clip_grad_norm(self.actor.parameters(),0.05)				
				self.optimizer_actor.step()

				# Critic :
				#before optimization :
				self.optimizer_critic.zero_grad()
				# sample action from next_state, without gradient repercusion :
				next_taction = self.target_actor(next_state_batch).detach()
				# evaluate the next state action over the target, without repercusion (faster...) :
				next_tqsa = torch.squeeze( self.target_critic( next_state_batch, next_taction).detach() ).view((-1))
				# Supervise loss :
				## y_true :
				y_true = reward_batch + self.gamma*next_tqsa 
				## y_pred :
				y_pred = torch.squeeze( self.critic(state_batch,action_batch) )
				## loss :
				#critic_loss = F.smooth_l1_loss(y_pred,y_true)
				criterion = nn.MSELoss()
				critic_loss = criterion(y_pred,y_true)
				critic_loss.backward()
				#clamping :
				torch.nn.utils.clip_grad_norm(self.critic.parameters(),0.5)				
				self.optimizer_critic.step()
				
				'''
				critic_grad = 0.0
				for p in self.critic.parameters() :
					critic_grad += np.mean(p.grad.cpu().data.numpy())
				print( 'Mean Critic Grad : {}'.format(critic_grad) )
				'''
				
				actor_grad = 0.0
				for p in self.actor.parameters() :
					actor_grad += np.max( np.abs(p.grad.cpu().data.numpy() ) )
				#print( 'Mean Actor Grad : {}'.format(actor_grad) )
				

				#UPDATE THE PR :
				loss = torch.abs(actor_loss) + torch.abs(critic_loss)
				#loss = torch.abs(actor_loss) #+ torch.abs(critic_loss)
				loss_np = loss.cpu().data.numpy()
				for (idx, new_error) in zip(batch.idx,loss_np) :
					new_priority = self.memory.priority(new_error)
					#print( 'prior = {} / {}'.format(new_priority,self.rBuffer.total()) )
					self.memory.update(idx,new_priority)
			
			except Exception as e :
				bashlogger.debug('error : {}',format(e) )
				

			# soft update :
			soft_update(self.target_critic, self.critic, self.tau)
			soft_update(self.target_actor, self.actor, self.tau)

			del batch
			del next_state_batch 
			del state_batch 
			del action_batch 
			del reward_batch 

			closs = critic_loss.cpu()
			aloss = actor_loss.cpu()
			del actor_loss
			del critic_loss

			return closs.data.numpy(), aloss.data.numpy(), actor_grad

		else :
			raise NotImplemented


	def save(self,path) :
		torch.save( self.actor.state_dict(), path+'.actor')
		torch.save( self.critic.state_dict(), path+'.critic')

	def load(self, path) :
		self.actor.load_state_dict( torch.load(path+'.actor') )
		hard_update(self.target_actor, self.actor)
		self.critic.load_state_dict( torch.load(path+'.critic') )
		hard_update(self.target_critic, self.critic)



class Model2Distributed :
	def __init__(self, actor, critic, memory, algo='ddpg',GAMMA=GAMMA,LR=LR,TAU=TAU,use_cuda=USE_CUDA,BATCH_SIZE=BATCH_SIZE,MIN_MEMORY=1e3 ) :
		self.actor = actor
		self.critic = critic
		self.target_actor = copy.deepcopy(actor)
		self.target_critic = copy.deepcopy(critic)

		self.use_cuda = use_cuda
		if self.use_cuda :
			self.actor = self.actor.cuda()
			self.target_actor = self.target_actor.cuda()
			self.critic = self.critic.cuda()
			self.target_critic = self.target_critic.cuda()


		self.memory = memory

		self.gamma = GAMMA
		self.lr = LR
		self.tau = TAU
		self.batch_size = BATCH_SIZE
		self.MIN_MEMORY = MIN_MEMORY

		self.mutex = Lock()
		
		self.noise = OrnsteinUhlenbeckNoise(self.actor.action_dim)

		hard_update(self.target_actor, self.actor)
		hard_update(self.target_critic, self.critic)

		self.algo = algo
		if self.algo == 'ppo' :
			self.previous_actor = copy.deepcopy(self.actor)
			self.epsilon = 0.2
			if self.use_cuda :
				self.previous_actor = self.previous_actor.cuda()

	
	def generate_optimizers(self) :
		optimizer_actor = optim.Adam(self.actor.parameters(), self.lr*1e0)
		optimizer_critic = optim.Adam(self.critic.parameters(), self.lr*1e2)

		return {'critic':optimizer_critic,'actor':optimizer_actor}


	def act(self, x,exploitation=True) :
		#self.actor.eval()
		
		#state = Variable( torch.from_numpy(x), volatile=True )
		state = Variable( x)
		if self.use_cuda :
			state = state.cuda()
		action = self.actor( state).detach()
		#action = self.target_actor( state).detach()
		
		try:
			if action.size()[0] != state.size()[0] :
				raise
		except Exception as e :
			print(action.size(),state.size(),x)
			raise e 
			
		if self.algo == 'ppo' or exploitation :
			return action.cpu().data.numpy()
		else :
			# exploration :
			new_action = action.cpu().data.numpy() + self.noise.sample()*self.actor.action_scaler
			return new_action

	def evaluate(self, x,a) :
		#self.critic.eval()
		
		state = Variable( x )
		action = Variable( a )
		if self.use_cuda :
			state = state.cuda()
			action = action.cuda()

		qsa = self.critic( state, action).detach()
		#qsa = self.target_critic( state, action).detach()
		
		return qsa.cpu().data.numpy()

	def optimize(self,optimizer_critic,optimizer_actor) :
		'''
		self.target_critic.eval()
		self.target_actor.eval()
		self.critic.train()
		self.actor.train()
		'''

		if self.algo == 'ppo' :
			self.mutex.acquire()
			
			if len(self.memory) < self.MIN_MEMORY :
						return
			
			rollout_size = len(self.memory)
			iterations = int(rollout_size/self.batch_size)

			acc_closs = 0.0
			acc_aloss = 0.0
			acc_actor_grad = 0.0

			for i in range(iterations) :					
				try :					
					#Create Batch 
					#self.mutex.acquire()
					
					if isinstance(self.memory, PrioritizedReplayBuffer) :
						#with PR :
						prioritysum = self.memory.total()
						randexp = np.random.random(size=self.batch_size)*prioritysum
						batch = list()
						for j in range(self.batch_size):
							try :
								el = self.memory.get(randexp[j])
								batch.append(el)
							except TypeError as e :
								continue
								#print('REPLAY BUFFER EXCEPTION...')
					else :
						#with random RB :
						batch = self.memory.sample(self.batch_size)

					#self.mutex.release()

					if len(batch) == 0 :
						continue

					# Create Batch with replayMemory :
					batch = TransitionPR( *zip(*batch) )
					next_state_batch = Variable(torch.cat( batch.next_state))#, requires_grad=False)
					state_batch = Variable( torch.cat( batch.state) )#, requires_grad=False)
					action_batch = Variable( torch.cat( batch.action) )#, requires_grad=False)
					reward_batch = Variable( torch.cat( batch.reward ) )#, requires_grad=False ).view((-1))
					
					if self.use_cuda :
						next_state_batch = next_state_batch.cuda()
						state_batch = state_batch.cuda()
						action_batch = action_batch.cuda()
						reward_batch = reward_batch.cuda()

					
					# Critic :
					# sample action from next_state, without gradient repercusion :
					#next_taction = self.target_actor(next_state_batch).detach()
					# evaluate the next state action over the target, without repercusion (faster...) :
					#next_tqsa = torch.squeeze( self.target_critic( next_state_batch, next_taction).detach() ).view((-1))
					
					# Supervise loss :
					## y_true :
					#y_true = reward_batch + self.gamma*next_tqsa 
					y_true = reward_batch
					## y_pred :
					y_pred = torch.squeeze( self.critic(state_batch,action_batch) )
					## loss :
					#critic_loss = F.smooth_l1_loss(y_pred,y_true)
					criterion = nn.MSELoss()
					critic_loss = criterion(y_pred,y_true)
					#before optimization :
					#self.critic.zero_grad()
					#self.actor.zero_grad()
					optimizer_critic.zero_grad()
					#optimizer_actor.zero_grad()
					critic_loss.backward()
					#clamping :
					#torch.nn.utils.clip_grad_norm(self.critic.parameters(),50)				
					optimizer_critic.step()
					
					
					###################################
					
					# Actor :
					#transfer current actor's weights to previous actor :
					hard_update(self.previous_actor, self.actor)
					
					#predict action with old weights :
					pred_old_action = self.previous_actor(state_batch)
					pred_old_dist = self.previous_actor.dist
					pred_old_log_prob_batch = pred_old_dist.log_prob(action_batch)
					#predict action with current weights :
					pred_action = self.actor(state_batch) 
					pred_dist = self.actor.dist
					pred_log_prob_batch = pred_dist.log_prob(action_batch)
					
					#helper vars :
					clipped_m = (1.0-self.epsilon)#*torch.ones(ratio.size())
					clipped_p = (1.0+self.epsilon)#*torch.ones(ratio.size())
					
					#compute advantage :
					qsa = self.critic( state_batch, action_batch)
					v = self.critic.Vvalue
					v = v.view((-1))
					advantages = reward_batch-v
					# center and reduice advantage :
					advantages = ( advantages-advantages.mean() ) / ( advantages.std() + 1e-10)
					advantages = advantages.view((-1,1) )
					advantages = Variable( advantages.data )
					
					#compute ratios :
					ratio = torch.exp(pred_log_prob_batch - Variable( pred_old_log_prob_batch.data) )
					clipped_ratio = ratio.clamp(clipped_m,clipped_p)
					surr = ratio * advantages
					clipped_surr = clipped_ratio * advantages
					actor_loss = -torch.min(surr,clipped_surr).mean()
					
					#before optimization :
					#self.actor.zero_grad()
					optimizer_actor.zero_grad()
					actor_loss.backward()
					
					#clamping :
					#clampactor = 5e1#np.max( [ 0.25, 1.0/np.max( [ 5e-1, np.abs( np.mean(critic_loss.cpu().data.numpy() ) ) ] ) ] )
					#torch.nn.utils.clip_grad_norm(self.actor.parameters(),clampactor)				
					
					#update of the weights :
					optimizer_actor.step()
					
					###################################

					
					'''
					critic_grad = 0.0
					for p in self.critic.parameters() :
						critic_grad += np.mean(p.grad.cpu().data.numpy())
					print( 'Mean Critic Grad : {}'.format(critic_grad) )
					'''
					
					actor_grad = 0.0
					for p in self.actor.parameters() :
						actor_grad += np.max( np.abs(p.grad.cpu().data.numpy() ) )
					#print( 'Mean Actor Grad : {}'.format(actor_grad) )
					
				except Exception as e :
					bashlogger.debug('error : {}',format(e) )
					raise e

				# soft update :
				soft_update(self.target_critic, self.critic, self.tau)
				#soft_update(self.target_actor, self.actor, self.tau)
				# hard update :
				#hard_update(self.target_critic, self.critic)
				hard_update(self.target_actor, self.actor)


				
				closs = critic_loss.cpu()
				aloss = actor_loss.cpu()
				
				acc_closs += closs
				acc_aloss += aloss
				acc_actor_grad += actor_grad

			self.mutex.release()

			#UPDATE THE PR :
			#self.memory.reset()

			return acc_closs.data.numpy(), acc_aloss.data.numpy(), acc_actor_grad

		else :
			raise NotImplemented


	def compute_returns(self,exp_buffer) :
		if self.algo == 'pg' :
			cumr = 0.0 
			for i in reversed(range(len(exp_buffer)) ) :
				cumr = exp_buffer[i].reward + self.gamma*cumr
				exp_buffer[i] = EXP(exp_buffer[i].state,exp_buffer[i].action,exp_buffer[i].next_state,cumr,exp_buffer[i].done)
			return exp_buffer
		elif self.algo == 'ddpg' :
			return exp_buffer
		elif self.algo == 'ppo' :
			cumr = 0.0 
			for i in reversed(range(len(exp_buffer)) ) :
				cumr = exp_buffer[i].reward + self.gamma*cumr
				exp_buffer[i] = EXP(exp_buffer[i].state,exp_buffer[i].action,exp_buffer[i].next_state,cumr,exp_buffer[i].done)
			return exp_buffer
		else :
			return exp_buffer


	def save(self,path) :
		torch.save( self.actor.state_dict(), path+'.actor')
		torch.save( self.critic.state_dict(), path+'.critic')

	def load(self, path) :
		self.actor.load_state_dict( torch.load(path+'.actor') )
		hard_update(self.target_actor, self.actor)
		self.critic.load_state_dict( torch.load(path+'.critic') )
		hard_update(self.target_critic, self.critic)



class ActorCriticModel2Distributed :
	def __init__(self, actor_critic, memory, algo='ddpg',GAMMA=GAMMA,LR=LR,TAU=TAU,use_cuda=USE_CUDA,BATCH_SIZE=BATCH_SIZE,MIN_MEMORY=1e3 ) :
		self.actor_critic = actor_critic
		self.target_actor_critic = copy.deepcopy(actor_critic)
		
		self.use_cuda = use_cuda
		if self.use_cuda :
			self.actor_critic = self.actor_critic.cuda()
			self.target_actor_critic = self.target_actor_critic.cuda()
			

		self.memory = memory

		self.gamma = GAMMA
		self.lr = LR
		self.tau = TAU
		self.batch_size = BATCH_SIZE
		self.MIN_MEMORY = MIN_MEMORY

		self.mutex = Lock()
		
		self.noise = OrnsteinUhlenbeckNoise(self.actor_critic.action_dim)

		hard_update(self.target_actor_critic, self.actor_critic)
		
		self.algo = algo
		if self.algo == 'ppo' :
			self.previous_actor_critic = copy.deepcopy(self.actor_critic)
			self.epsilon = 0.2
			if self.use_cuda :
				self.previous_actor_critic = self.previous_actor_critic.cuda()

	
	def generate_optimizers(self) :
		optimizer = optim.Adam(self.actor_critic.parameters(), self.lr*1e0)
		
		return {'critic':optimizer,'actor':optimizer}


	def act(self, x,exploitation=True) :
		self.actor_critic.eval()
		
		#state = Variable( torch.from_numpy(x), volatile=True )
		state = Variable( x, volatile=True )
		if self.use_cuda :
			state = state.cuda()
		action = self.actor_critic.act( state).detach()
		#action = self.target_actor( state).detach()
		
		if self.algo == 'ppo' or exploitation :
			return action.cpu().data.numpy()
		else :
			# exploration :
			new_action = action.cpu().data.numpy() + self.noise.sample()*self.actor.action_scaler
			return new_action

	def evaluate(self, x, a) :
		self.actor_critic.eval()
		
		state = Variable( x, volatile=True )
		action = Variable( a, volatile=True )
		if self.use_cuda :
			state = state.cuda()
			action = action.cuda()

		qsa = self.actor_critic.evaluate( state, action).detach()
		#qsa = self.target_critic( state, action).detach()
		
		return qsa.cpu().data.numpy()

	def evaluateV(self, x) :
		self.actor_critic.eval()
		
		state = Variable( x, volatile=True )
		action = Variable( a, volatile=True )
		if self.use_cuda :
			state = state.cuda()
			action = action.cuda()

		V = self.actor_critic.evaluateV( state).detach()
		#qsa = self.target_critic( state, action).detach()
		
		return V.cpu().data.numpy()

	def optimize(self,optimizer_critic,optimizer_actor) :
		self.target_actor_critic.eval()
		self.actor_critic.train()
		
		optimizer = optimizer_actor

		if self.algo == 'ppo' :
			if len(self.memory) < self.MIN_MEMORY :
						return
			
			rollout_size = len(self.memory)
			iterations = int(rollout_size/self.batch_size)

			acc_closs = 0.0
			acc_aloss = 0.0
			acc_actor_grad = 0.0

			for i in range(iterations) :					
				try :					
					#Create Batch 
					self.mutex.acquire()
					
					if isinstance(self.memory, PrioritizedReplayBuffer) :
						#with PR :
						prioritysum = self.memory.total()
						randexp = np.random.random(size=self.batch_size)*prioritysum
						batch = list()
						for j in range(self.batch_size):
							try :
								el = self.memory.get(randexp[j])
								batch.append(el)
							except TypeError as e :
								continue
								#print('REPLAY BUFFER EXCEPTION...')
					else :
						#with random RB :
						batch = self.memory.sample(self.batch_size)

					self.mutex.release()

					if len(batch) == 0 :
						continue

					# Create Batch with replayMemory :
					batch = TransitionPR( *zip(*batch) )
					next_state_batch = Variable(torch.cat( batch.next_state))#, requires_grad=False)
					state_batch = Variable( torch.cat( batch.state) )#, requires_grad=False)
					action_batch = Variable( torch.cat( batch.action) )#, requires_grad=False)
					reward_batch = Variable( torch.cat( batch.reward ) )#, requires_grad=False ).view((-1))
					
					if self.use_cuda :
						next_state_batch = next_state_batch.cuda()
						state_batch = state_batch.cuda()
						action_batch = action_batch.cuda()
						reward_batch = reward_batch.cuda()

					#transfer current actor's weights to previous actor :
					hard_update(self.previous_actor_critic, self.actor_critic)
					optimizer.zero_grad()
					
					# Critic :
					# Supervise loss :
					## y_true :
					y_true = reward_batch
					## y_pred :
					y_pred = torch.squeeze( self.actor_critic.evaluateV(state_batch) )
					## loss :
					#critic_loss = F.smooth_l1_loss(y_pred,y_true)
					criterion = nn.MSELoss()
					critic_loss = criterion(y_pred,y_true)
					
					
					###################################
					
					# Actor :
					
					#predict action with old weights :
					pred_old_action = self.previous_actor_critic.act(state_batch)
					pred_old_dist = self.previous_actor_critic.dist
					pred_old_log_prob_batch = pred_old_dist.log_prob(action_batch)
					#predict action with current weights :
					pred_action = self.actor_critic.act( state_batch) 
					pred_dist = self.actor_critic.dist
					pred_log_prob_batch = pred_dist.log_prob(action_batch)
					
					#helper vars :
					clipped_m = (1.0-self.epsilon)#*torch.ones(ratio.size())
					clipped_p = (1.0+self.epsilon)#*torch.ones(ratio.size())
					
					#compute advantage :
					v = self.actor_critic.evaluateV( state_batch)
					v = v.view((-1))
					advantages = reward_batch-v
					# center and reduice advantage :
					advantages = ( advantages-advantages.mean() ) / ( advantages.std() + 1e-10)
					advantages = advantages.view((-1,1) )
					advantages = Variable( advantages.data )
					
					#compute ratios :
					ratio = torch.exp(pred_log_prob_batch - Variable( pred_old_log_prob_batch.data) )
					clipped_ratio = ratio.clamp(clipped_m,clipped_p)
					surr = ratio * advantages
					clipped_surr = clipped_ratio * advantages
					actor_loss = -torch.min(surr,clipped_surr).mean()
					
					(critic_loss*0.1+actor_loss).backward()
					
					#clamping :
					clampactor = 5e-1#np.max( [ 0.25, 1.0/np.max( [ 5e-1, np.abs( np.mean(critic_loss.cpu().data.numpy() ) ) ] ) ] )
					torch.nn.utils.clip_grad_norm(self.actor_critic.parameters(),clampactor)				
					
					#update of the weights :
					optimizer.step()
					
					###################################

					
					'''
					critic_grad = 0.0
					for p in self.critic.parameters() :
						critic_grad += np.mean(p.grad.cpu().data.numpy())
					print( 'Mean Critic Grad : {}'.format(critic_grad) )
					'''
					
					actor_grad = 0.0
					for p in self.actor_critic.parameters() :
						if p.grad is not None :
							actor_grad += np.max( np.abs(p.grad.cpu().data.numpy() ) )
					#print( 'Mean Actor Grad : {}'.format(actor_grad) )
					
				except Exception as e :
					bashlogger.debug('error : {}',format(e) )
					raise e

				# soft update :
				#soft_update(self.target_critic, self.critic, self.tau)
				#soft_update(self.target_actor, self.actor, self.tau)
				# hard update :
				#hard_update(self.target_critic, self.critic)
				hard_update(self.target_actor_critic, self.actor_critic)


				
				closs = critic_loss.cpu()
				aloss = actor_loss.cpu()
				
				acc_closs += closs
				acc_aloss += aloss
				acc_actor_grad += actor_grad

			#UPDATE THE PR :
			self.memory.reset()

			return acc_closs.data.numpy(), acc_aloss.data.numpy(), acc_actor_grad

		else :
			raise NotImplemented


	def compute_returns(self,exp_buffer) :
		if self.algo == 'pg' :
			cumr = 0.0 
			for i in reversed(range(len(exp_buffer)) ) :
				cumr = exp_buffer[i].reward + self.gamma*cumr
				exp_buffer[i] = EXP(exp_buffer[i].state,exp_buffer[i].action,exp_buffer[i].next_state,cumr,exp_buffer[i].done)
			return exp_buffer
		elif self.algo == 'ddpg' :
			return exp_buffer
		elif self.algo == 'ppo' :
			cumr = 0.0 
			for i in reversed(range(len(exp_buffer)) ) :
				cumr = exp_buffer[i].reward + self.gamma*cumr
				exp_buffer[i] = EXP(exp_buffer[i].state,exp_buffer[i].action,exp_buffer[i].next_state,cumr,exp_buffer[i].done)
			return exp_buffer
		else :
			return exp_buffer


	def save(self,path) :
		torch.save( self.actor_critic.state_dict(), path+'.actor_critic')
		
	def load(self, path) :
		self.actor_critic.load_state_dict( torch.load(path+'.actor_critic') )
		hard_update(self.target_actor_critic, self.actor_critic)
		
