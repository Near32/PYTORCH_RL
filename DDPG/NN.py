import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


EPS = 3e-3

def init_weights(size):
	v = 1. / np.sqrt(size[0])
	return torch.Tensor(size).uniform_(-v, v)

class ActorNN(nn.Module) :
	def __init__(self,state_dim=3,action_dim=2,action_scaler=2.0,CNN={'use_cnn':False,'input_size':3},HER=True) :
		super(ActorNN,self).__init__()
		
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_scaler = action_scaler

		self.CNN = CNN
		# dictionnary with :
		# - 'input_size' : int
		# - 'use_cnn' : bool
		#
		if self.CNN['use_cnn'] :
			self.state_dim = self.CNN['input_size']

		self.HER = HER
		if self.HER :
			self.state_dim *= 2

		#Features :
		if self.CNN['use_cnn'] :
			self.conv1 = nn.Conv2d(self.state_dim,16, kernel_size=5, stride=2)
			self.bn1 = nn.BatchNorm2d(16)
			self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
			self.bn2 = nn.BatchNorm2d(32)
			self.conv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
			self.bn3 = nn.BatchNorm2d(32)
			#self.featx = nn.Linear(448,self.nbr_actions)
			self.featx = nn.Linear(192,128)
		else :
			self.fc1 = nn.Linear(self.state_dim,256)
			self.fc1.weight.data = init_weights(self.fc1.weight.data.size())
			#self.bn1 = nn.BatchNorm1d(512)
			self.fc2 = nn.Linear(256,128)
			self.fc2.weight.data = init_weights(self.fc2.weight.data.size())	
			#self.bn2 = nn.BatchNorm1d(256)
			#self.featx = nn.Linear(448,self.nbr_actions)
			self.featx = nn.Linear(128,64)
			

		self.featx.weight.data = init_weights(self.featx.weight.data.size())

		# Actor network :
		self.actor_final = nn.Linear(64,self.action_dim)
		self.actor_final.weight.data.uniform_(-EPS,EPS)


	def features(self,x) :
		if self.CNN['use_cnn'] :
			x1 = F.relu( self.bn1(self.conv1(x) ) )
			x2 = F.relu( self.bn2(self.conv2(x1) ) )
			x3 = F.relu( self.bn3(self.conv3(x2) ) )
			x4 = x3.view( x3.size(0), -1)
			
			fx = F.relu( self.featx( x4) )
			# batch x 128 
		else :
			#x1 = F.relu( self.bn1(self.fc1(x) ) )
			x1 = F.relu( self.fc1(x) )
			#x2 = F.relu( self.bn2(self.fc2(x1) ) )
			x2 = F.relu( self.fc2(x1)  )
			fx = F.relu( self.featx( x2) )
			#fx = F.relu( self.featx( x1) )
			# batch x 128
	
		return fx

	def forward(self, x) :
		
		fx = self.features(x)

		xx = self.actor_final( fx )
		
		#scale the actions :
		unscaled = F.tanh(xx)
		scaled = unscaled * self.action_scaler

		return scaled

class VCriticNN(nn.Module) :
	def __init__(self,HER=False) :
		super(VCriticNN,self).__init__()
		self.HER = HER
		nbrchannel = 3
		if self.HER :
			nbrchannel *= 2

		self.conv1 = nn.Conv2d(nbrchannel,16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)
		#self.head = nn.Linear(448,self.nbr_actions)
		self.head = nn.Linear(192,1)

	def forward(self, x) :
		x1 = F.relu( self.bn1(self.conv1(x) ) )
		x2 = F.relu( self.bn2(self.conv2(x1) ) )
		x3 = F.relu( self.bn3(self.conv3(x2) ) )
		x4 = x3.view( x3.size(0), -1)
		x5 = self.head( x4 )
		
		return x5


class CriticNN(nn.Module) :
	def __init__(self,state_dim=3,action_dim=2,CNN={'use_cnn':False,'input_size':3},dueling=True,HER=True) :
		super(CriticNN,self).__init__()
		
		self.state_dim = state_dim
		self.action_dim = action_dim
		
		self.dueling = dueling

		self.CNN = CNN
		# dictionnary with :
		# - 'input_size' : int
		# - 'use_cnn' : bool
		#
		if self.CNN['use_cnn'] :
			self.state_dim = self.CNN['input_size']

		self.HER = HER
		if self.HER :
			self.state_dim *= 2

		#Features :
		if self.CNN['use_cnn'] :
			self.conv1 = nn.Conv2d(self.state_dim,16, kernel_size=5, stride=2)
			self.bn1 = nn.BatchNorm2d(16)
			self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
			self.bn2 = nn.BatchNorm2d(32)
			self.conv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
			self.bn3 = nn.BatchNorm2d(32)
			#self.featx = nn.Linear(448,self.nbr_actions)
			self.featx = nn.Linear(192,128)
		else :
			self.fc1 = nn.Linear(self.state_dim,256)
			self.fc1.weight.data = init_weights(self.fc1.weight.data.size())
			#self.bn1 = nn.BatchNorm1d(256)
			#self.fc2 = nn.Linear(512,256)
			#self.bn2 = nn.BatchNorm1d(256)
			#self.featx = nn.Linear(448,self.nbr_actions)
			self.featx = nn.Linear(256,128)

		
		self.featx.weight.data = init_weights(self.featx.weight.data.size())

		# Critic network :
		## state value path :
		if self.dueling :
			self.critic_Vhead = nn.Linear(128,1)
		else :
			self.critic_Vhead = nn.Linear(128,64)
		
		self.critic_Vhead.weight.data = init_weights(self.critic_Vhead.weight.data.size())
		
		## action value path :
		self.critic_afc1 = nn.Linear(self.action_dim,128)
		self.critic_afc1.weight.data.uniform_(-1e-1,1e-1)
		#self.critic_afc1.weight.data = init_weights(self.critic_afc1.weight.data.size())
		#self.critic_afc2 = nn.Linear(256,128)
		#self.critic_afc2.weight.data.uniform_(-EPS,EPS)

		if self.dueling :
			self.critic_ahead = nn.Linear(256,1)
			self.critic_ahead.weight.data = init_weights(self.critic_ahead.weight.data.size())
		else :
			self.critic_ahead = nn.Linear(256,128)
			self.critic_ahead.weight.data = init_weights(self.critic_ahead.weight.data.size())
			#linear layer, after the concatenation of ahead and vhead :
			'''
			self.critic_final = nn.Linear(128,1)
			self.critic_final.weight.data.uniform_(-EPS,EPS) 
			'''
			self.critic_final1 = nn.Linear(128,64)
			self.critic_final1.weight.data = init_weights(self.critic_final1.weight.data.size())
			self.critic_final2 = nn.Linear(64,1)
			self.critic_final2.weight.data.uniform_(-EPS,EPS) 


	def features(self,x) :
		if self.CNN['use_cnn'] :
			x1 = F.relu( self.bn1(self.conv1(x) ) )
			x2 = F.relu( self.bn2(self.conv2(x1) ) )
			x3 = F.relu( self.bn3(self.conv3(x2) ) )
			x4 = x3.view( x3.size(0), -1)
			
			fx = F.relu( self.featx( x4) )
			# batch x 128 
		else :
			#x1 = F.relu( self.bn1(self.fc1(x) ) )
			x1 = F.relu( self.fc1(x) )
			#x2 = F.relu( self.bn2(self.fc2(x1) ) )
			#fx = F.relu( self.featx( x2) )
			fx = F.relu( self.featx( x1) )
			# batch x 128
	
		return fx

	def forward(self, x,a) :
		
		fx = self.features(x)

		#V value :
		v = self.critic_Vhead( fx )
		

		a1 = F.relu( self.critic_afc1(a) )
		#a2 = F.relu( self.critic_afc2(a1) )
		# batch x 128
		#afx = torch.cat([ fx, a2], dim=1)
		afx = torch.cat([ fx, a1], dim=1)
		# batch x 256

		if self.dueling :
			advantage = self.critic_ahead(afx)
			out = advantage + v
		else :
			advantage = self.critic_ahead(afx)
			#concat = torch.cat( [ v,advantage], dim=1)
			'''
			out = self.critic_final(advantage)
			'''
			preout = self.critic_final1(advantage)
			out = self.critic_final2(preout)

		return out

class ActorCriticNN(nn.Module) :
	def __init__(self,state_dim=3,action_dim=2,action_scaler=2.0,CNN={'use_cnn':False,'input_size':3},dueling=True,HER=True) :
		super(ActorCriticNN,self).__init__()
		
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_scaler = action_scaler

		self.dueling = dueling

		self.CNN = CNN
		# dictionnary with :
		# - 'input_size' : int
		# - 'use_cnn' : bool
		#
		if self.CNN['use_cnn'] :
			self.state_dim = self.CNN['input_size']

		self.HER = HER
		if self.HER :
			self.state_dim *= 2

		#Features :
		if self.CNN['use_cnn'] :
			self.conv1 = nn.Conv2d(self.state_dim,16, kernel_size=5, stride=2)
			self.bn1 = nn.BatchNorm2d(16)
			self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
			self.bn2 = nn.BatchNorm2d(32)
			self.conv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
			self.bn3 = nn.BatchNorm2d(32)
			#self.featx = nn.Linear(448,self.nbr_actions)
			self.featx = nn.Linear(192,128)
		else :
			self.fc1 = nn.Linear(self.state_dim,512)
			self.bn1 = nn.BatchNorm1d(512)
			self.fc2 = nn.Linear(512,256)
			self.bn2 = nn.BatchNorm1d(256)
			#self.featx = nn.Linear(448,self.nbr_actions)
			self.featx = nn.Linear(256,128)

		self.featx.weight.data.uniform_(-EPS,EPS)

		# Critic network :
		## state value path :
		if self.dueling :
			self.critic_Vhead = nn.Linear(128,1)
		else :
			self.critic_Vhead = nn.Linear(128,64)
		self.critic_Vhead.weight.data.uniform_(-EPS,EPS)

		## action value path :
		self.critic_afc1 = nn.Linear(self.action_dim,256)
		self.critic_afc1.weight.data.uniform_(-EPS,EPS)
		self.critic_afc2 = nn.Linear(256,128)
		self.critic_afc2.weight.data.uniform_(-EPS,EPS)

		if self.dueling :
			self.critic_ahead = nn.Linear(256,128)
			self.critic_ahead.weight.data.uniform_(-EPS,EPS)
		else :
			self.critic_ahead = nn.Linear(256,64)
			self.critic_ahead.weight.data.uniform_(-EPS,EPS)
			#linear layer, after the concatenation of ahead and vhead :
			self.critic_final = nn.Linear(128,1)
			self.critic_final.weight.data.uniform_(-EPS,EPS)

		# Actor network :
		self.actor_final = nn.Linear(128,self.action_dim)
		self.actor_final.weight.data.uniform_(-EPS,EPS)


	def features(self,x) :
		if self.CNN['use_cnn'] :
			x1 = F.relu( self.bn1(self.conv1(x) ) )
			x2 = F.relu( self.bn2(self.conv2(x1) ) )
			x3 = F.relu( self.bn3(self.conv3(x2) ) )
			x4 = x3.view( x3.size(0), -1)
			
			fx = F.relu( self.featx( x4) )
			# batch x 128 
		else :
			x1 = F.relu( self.bn1(self.fc1(x) ) )
			x2 = F.relu( self.bn2(self.fc2(x1) ) )
			fx = F.relu( self.featx( x2) )
			# batch x 128
	
		return fx

	def critic(self, x,a) :
		
		fx = self.features(x)

		#V value :
		v = self.critic_Vhead( fx )
		

		a1 = F.relu( self.critic_afc1(a) )
		a2 = F.relu( self.critic_afc2(a1) )
		# batch x 128
		afx = torch.cat([ fx, a2], dim=1)
		# batch x 256

		if self.dueling :
			advantage = self.critic_ahead(afx)
			out = advantage + v
		else :
			advantage = self.critic_ahead(afx)
			concat = torch.cat( [ v,advantage], dim=1)
			out = self.critic_final(concat)

		return out

	def actor(self, x) :
		
		fx = self.features(x)

		xx = self.actor_final( fx )
		
		#scale the actions :
		unscaled = F.tanh(xx)
		scaled = unscaled * self.action_scaler

		return scaled



class ActorCriticNN2(nn.Module) :
	def __init__(self,state_dim=3,action_dim=2,action_scaler=2.0,CNN={'use_cnn':False,'input_size':3},dueling=True,HER=True) :
		super(ActorCriticNN2,self).__init__()
		
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_scaler = action_scaler

		self.dueling = dueling

		self.CNN = CNN
		# dictionnary with :
		# - 'input_size' : int
		# - 'use_cnn' : bool
		#
		if self.CNN['use_cnn'] :
			self.state_dim = self.CNN['input_size']

		self.HER = HER
		if self.HER :
			self.state_dim *= 2

		# Critic network :
		#Features :
		if self.CNN['use_cnn'] :
			self.cconv1 = nn.Conv2d(self.state_dim,16, kernel_size=5, stride=2)
			self.cbn1 = nn.BatchNorm2d(16)
			self.cconv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
			self.cbn2 = nn.BatchNorm2d(32)
			self.cconv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
			self.cbn3 = nn.BatchNorm2d(32)
			#self.featx = nn.Linear(448,self.nbr_actions)
			self.cfeatx = nn.Linear(192,128)
		else :
			self.cfc1 = nn.Linear(self.state_dim,512)
			self.cbn1 = nn.BatchNorm1d(512)
			self.cfc2 = nn.Linear(512,256)
			self.cbn2 = nn.BatchNorm1d(256)
			#self.featx = nn.Linear(448,self.nbr_actions)
			self.cfeatx = nn.Linear(256,128)

		self.cfeatx.weight.data.uniform_(-EPS,EPS)

		## state value path :
		if self.dueling :
			self.critic_Vhead = nn.Linear(128,1)
		else :
			self.critic_Vhead = nn.Linear(128,64)
		self.critic_Vhead.weight.data.uniform_(-EPS,EPS)

		## action value path :
		self.critic_afc1 = nn.Linear(self.action_dim,256)
		self.critic_afc1.weight.data.uniform_(-EPS,EPS)
		self.critic_afc2 = nn.Linear(256,128)
		self.critic_afc2.weight.data.uniform_(-EPS,EPS)

		if self.dueling :
			self.critic_ahead = nn.Linear(256,128)
			self.critic_ahead.weight.data.uniform_(-EPS,EPS)
		else :
			self.critic_ahead = nn.Linear(256,64)
			self.critic_ahead.weight.data.uniform_(-EPS,EPS)
			#linear layer, after the concatenation of ahead and vhead :
			self.critic_final = nn.Linear(128,1)
			self.critic_final.weight.data.uniform_(-EPS,EPS)

		# Actor network :
		#Features :
		if self.CNN['use_cnn'] :
			self.aconv1 = nn.Conv2d(self.state_dim,16, kernel_size=5, stride=2)
			self.abn1 = nn.BatchNorm2d(16)
			self.aconv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
			self.abn2 = nn.BatchNorm2d(32)
			self.aconv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
			self.abn3 = nn.BatchNorm2d(32)
			#self.featx = nn.Linear(448,self.nbr_actions)
			self.afeatx = nn.Linear(192,128)
		else :
			self.afc1 = nn.Linear(self.state_dim,512)
			self.abn1 = nn.BatchNorm1d(512)
			self.afc2 = nn.Linear(512,256)
			self.abn2 = nn.BatchNorm1d(256)
			#self.featx = nn.Linear(448,self.nbr_actions)
			self.afeatx = nn.Linear(256,128)

		self.afeatx.weight.data.uniform_(-EPS,EPS)

		self.actor_final = nn.Linear(128,self.action_dim)
		self.actor_final.weight.data.uniform_(-EPS,EPS)


	def featuresCritic(self,x) :
		if self.CNN['use_cnn'] :
			x1 = F.relu( self.cbn1(self.cconv1(x) ) )
			x2 = F.relu( self.cbn2(self.cconv2(x1) ) )
			x3 = F.relu( self.cbn3(self.cconv3(x2) ) )
			x4 = x3.view( x3.size(0), -1)
			
			fx = F.relu( self.cfeatx( x4) )
			# batch x 128 
		else :
			x1 = F.relu( self.cbn1(self.cfc1(x) ) )
			x2 = F.relu( self.cbn2(self.cfc2(x1) ) )
			fx = F.relu( self.cfeatx( x2) )
			# batch x 128
	
		return fx

	def featuresActor(self,x) :
		if self.CNN['use_cnn'] :
			x1 = F.relu( self.abn1(self.aconv1(x) ) )
			x2 = F.relu( self.abn2(self.aconv2(x1) ) )
			x3 = F.relu( self.abn3(self.aconv3(x2) ) )
			x4 = x3.view( x3.size(0), -1)
			
			fx = F.relu( self.afeatx( x4) )
			# batch x 128 
		else :
			x1 = F.relu( self.abn1(self.afc1(x) ) )
			x2 = F.relu( self.abn2(self.afc2(x1) ) )
			fx = F.relu( self.afeatx( x2) )
			# batch x 128
	
		return fx

	
	def critic(self, x,a) :
		
		fx = self.featuresCritic(x)

		#V value :
		self.v = self.critic_Vhead( fx )
		

		a1 = F.relu( self.critic_afc1(a) )
		a2 = F.relu( self.critic_afc2(a1) )
		# batch x 128
		afx = torch.cat([ fx, a2], dim=1)
		# batch x 256

		if self.dueling :
			self.advantage = self.critic_ahead(afx)
			self.out = self.advantage + self.v
		else :
			self.advantage = self.critic_ahead(afx)
			concat = torch.cat( [ self.v,afx], dim=1)
			self.out = self.critic_final(concat)

		return self.out

	def actor(self, x) :
		
		fx = self.featuresActor(x)

		xx = self.actor_final( fx )
		
		#scale the actions :
		self.unscaled = F.tanh(xx)
		self.scaled = self.unscaled * self.action_scaler

		return self.scaled