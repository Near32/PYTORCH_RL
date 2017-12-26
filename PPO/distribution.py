import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Distribution(object) :
	def sample(self) :
		raise NotImplementedError

	def log_prob(self,values) :
		raise NotImplementedError

class Bernoulli(Distribution) :
	def __init__(self, probs) :
		self.probs = probs

	def sample(self) :
		return torch.bernoulli(self.probs)

	def log_prob(self,values) :
		log_pmf = ( torch.stack( [1-self.probs, self.probs] ) ).log()

		return log_pmf.gather( 0, values.unsqueeze(0).long() ).squeeze(0)

class Normal(Distribution) :
	def __init__(self, mean, std=None, log_var=None) :
		super(Normal,self).__init__()
		self.mean = mean
		
		if std is not None :
			self.std = std
			self.log_var = None
		else :
			if log_var is None :
				raise 
			self.log_var = log_var
			self.std = None
	def sample(self) :
		if self.std is not None :
			return torch.normal(self.mean, self.std)
		else :
			return torch.normal(self.mean, torch.exp( self.log_var/2.0 ) ) 

	def log_prob(self,values) :
		if self.log_var is None :
			var = self.std**2
			#log_std = math.log(self.std)
			log_std = self.std.log()
		else :
			log_std = self.log_var/2
			var = torch.exp( self.log_var)

		return -((values - self.mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))


def test_normal_std() :
	mean = torch.randn((8,1) )
	dist = Normal( mean, std=1e-3*torch.ones((8,1) ) )

	sample = dist.sample()
	print( torch.cat( [mean,sample], dim=1) )

	log_prob = dist.log_prob(sample)
	print(log_prob)

def test_normal_logvar() :
	mean = torch.randn((8,1) )
	dist = Normal( mean, log_var=-1e2*torch.ones((8,1) ) )

	sample = dist.sample()
	print( torch.cat( [mean,sample], dim=1) )

	log_prob = dist.log_prob(sample)
	print(log_prob)

if __name__ == '__main__' :
	#test_normal_std()
	test_normal_logvar()





