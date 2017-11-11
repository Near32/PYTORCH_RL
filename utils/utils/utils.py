import torch

import numpy as np

def hard_update(fromm, to) :
	for fp, tp in zip( fromm.parameters(), toparameters() ) :
		fp.cpu().data.copy_( tp.cpu().data )

def soft_update(fromm, to, tau) :
	for fp, tp in zip( fromm.parameters(), toparameters() ) :
		fp.cpu().data.copy_( (1.0-tau)*fp.cpu().data + tau*tp.cpu().data ) 

class OrnsteinUhlenbeckNoise :
	def __init__(self, dim,mu=0.0, theta=0.15, sigma=0.2) :
		self.dim = dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma

		self.X = np.ones(self.dim)*self.mu

	def sample(self) :
		dx = self.theta * ( self.mu - self.X)
		dx += self.sigma *  np.random.randn( self.dim )
		self.X += dx
		return self.X





