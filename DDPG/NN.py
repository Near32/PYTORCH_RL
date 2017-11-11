import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ActorNN(nn.Module) :
	def __init__(self,action_scaler,nbr_actions=2,CNN={'use_cnn':False,'input_size':3},HER=True) :
		super(ActorNN,self).__init__()
		self.nbr_actions = nbr_actions
		self.action_scaler = action_scaler

		self.CNN = CNN
		# dictionnary with :
		# - 'input_size' : int
		# - 'use_cnn' : bool
		#
		nbrchannel = self.CNN['input_size']
		
		self.HER = HER
		if self.HER :
			nbrchannel *= 2

		if self.CNN['use_cnn'] :
			self.conv1 = nn.Conv2d(nbrchannel,16, kernel_size=5, stride=2)
			self.bn1 = nn.BatchNorm2d(16)
			self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
			self.bn2 = nn.BatchNorm2d(32)
			self.conv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
			self.bn3 = nn.BatchNorm2d(32)
			#self.head = nn.Linear(448,self.nbr_actions)
			self.head = nn.Linear(192,self.nbr_actions)
		else :
			self.fc1 = nn.Linear(nbrchannel,256)
			self.bn1 = nn.BatchNorm1d(256)
			self.fc2 = nn.Linear(256,128)
			self.bn2 = nn.BatchNorm1d(128)
			self.head = nn.Linear(128,self.nbr_actions)

		
	def forward(self, x) :
		
		if self.CNN['use_cnn'] :
			x1 = F.relu( self.bn1(self.conv1(x) ) )
			x2 = F.relu( self.bn2(self.conv2(x1) ) )
			x3 = F.relu( self.bn3(self.conv3(x2) ) )
			fx = x3.view( x3.size(0), -1)
		else :
			x1 = F.relu( self.bn1(self.fc1(x) ) )
			fx = F.relu( self.bn2(self.fc2(x1) ) )
				
		xx = self.head( fx )
		
		#scale the actions :
		self.unscaled = F.tanh(xx)
		self.scaled = self.unscaled * self.action_scaler

		return self.scaled


class VCriticNN(nn.Module) :
	def __init__(self,HER=False) :
		super(CriticNN,self).__init__()
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


class QCriticNN(nn.Module) :
	def __init__(self,nbr_actions=2,CNN={'use_cnn':False,'input_size':3},dueling=True,HER=True) :
		super(CriticNN,self).__init__()
		self.dueling = dueling

		self.CNN = CNN
		# dictionnary with :
		# - 'input_size' : int
		# - 'use_cnn' : bool
		#
		nbrchannel = self.CNN['input_size']

		self.HER = HER
		if self.HER :
			nbrchannel *= 2

		if self.CNN['use_cnn'] :
			self.conv1 = nn.Conv2d(nbrchannel,16, kernel_size=5, stride=2)
			self.bn1 = nn.BatchNorm2d(16)
			self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
			self.bn2 = nn.BatchNorm2d(32)
			self.conv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
			self.bn3 = nn.BatchNorm2d(32)
			#self.featx = nn.Linear(448,self.nbr_actions)
			self.featx = nn.Linear(192,128)
		else :
			else :
			self.fc1 = nn.Linear(nbrchannel,512)
			self.bn1 = nn.BatchNorm1d(512)
			self.fc2 = nn.Linear(512,256)
			self.bn2 = nn.BatchNorm1d(256)
			#self.featx = nn.Linear(448,self.nbr_actions)
			self.featx = nn.Linear(256,128)

		
		if self.dueling :
			self.Vhead = nn.Linear(128,1)
		else :
			self.Vhead = nn.Linear(128,64)

		#action path :
		self.nbr_actions = nbr_actions
		self.afc1 = nn.Linear(self.nbr_actions,256)
		self.afc2 = nn.Linear(256,128)

		if self.dueling :
			self.ahead = nn.Linear(256,128)
		else :
			self.ahead = nn.Linear(256,64)
			#linear layer, after the concatenation of ahead and vhead :
			self.final = nn.Linear(128,1)

	def forward(self, x,a) :
		
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

		#V value :
		self.v = self.Vhead( fx )
		

		a1 = F.relu( self.afc1(a) )
		a2 = F.relu( self.afc2(a1) )
		# batch x 128
		afx = torch.cat([ fx, a2], dim=1)
		# batch x 256

		if self.dueling :
			self.advantage = self.ahead(afx)
			self.out = self.advantage + self.v
		else :
			self.advantage = self.ahead(afx)
			concat = torch.cat( [ self.v,afx], dim=1)
			self.out = self.final(concat)

		return out

class ActorCriticNN(nn.Module) :
	def __init__(self,state_dim=3,action_dim=2,action_scaler=2.0,CNN={'use_cnn':False,'input_size':3},dueling=True,HER=True) :
		super(ActorCriticNN,self).__init__()
		
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
			else :
			self.fc1 = nn.Linear(self.state_dim,512)
			self.bn1 = nn.BatchNorm1d(512)
			self.fc2 = nn.Linear(512,256)
			self.bn2 = nn.BatchNorm1d(256)
			#self.featx = nn.Linear(448,self.nbr_actions)
			self.featx = nn.Linear(256,128)

		# Critic network :
		## state value path :
		if self.dueling :
			self.critic_Vhead = nn.Linear(128,1)
		else :
			self.critic_Vhead = nn.Linear(128,64)

		## action value path :
		self.critic_afc1 = nn.Linear(self.action_dim,256)
		self.critic_afc2 = nn.Linear(256,128)

		if self.dueling :
			self.critic_ahead = nn.Linear(256,128)
		else :
			self.critic_ahead = nn.Linear(256,64)
			#linear layer, after the concatenation of ahead and vhead :
			self.critic_final = nn.Linear(128,1)

		# Actor network :
		self.actor_final = nn.Linear(128,self.action_dim)


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

		return out

	def act(self, x) :
		
		fx = self.features(x)

		xx = self.actor_final( fx )
		
		#scale the actions :
		self.unscaled = F.tanh(xx)
		self.scaled = self.unscaled * self.action_scaler

		return self.scaled