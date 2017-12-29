# Deep Reinforcement Learning with PyTorch

## Deep Q-Network (DQN) 
This implementation of the Deep Q-Network (["Human-level control through deep reinforcement learning"][https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf]) can be augmented with the following features :
	* ["Prioritized Experience Replay"][https://arxiv.org/pdf/1511.05952.pdf]
	* ["Dueling Deep Q-Network"][https://arxiv.org/pdf/1511.06581.pdf]
	* ["Double Deep Q-Network"][https://arxiv.org/pdf/1509.06461.pdf]
	* a multi-threaded ["Distributed Architecture"][https://arxiv.org/pdf/1508.04186.pdf] with a unique replay memory though.
	* ["Hindsight Experience Replay"][https://arxiv.org/pdf/1707.01495.pdf]
	
Experiment : CartPole :
	* Adam
	* learning rate : 1e-4
	* minibatch size : 128
	* replay memory capacity : 25e3
	*	prioritized experience replay exponent $\alpha$ : 0.5
	* number of thread/worker : 1
	* hindsight experience replay : [ ]
	
![resultDQN1](/results/DQN/result.png)

## Deep Deterministic Policy Gradient (DDPG)
 
