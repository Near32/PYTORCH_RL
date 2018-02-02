# Deep Reinforcement Learning with PyTorch

## Deep Q-Network (DQN) 
This implementation of the Deep Q-Network (["Human-level control through deep reinforcement learning"](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)) can be augmented with the following features :

* ["Prioritized Experience Replay"](https://arxiv.org/pdf/1511.05952.pdf)
* ["Dueling Deep Q-Network"](https://arxiv.org/pdf/1511.06581.pdf)
* ["Double Deep Q-Network"](https://arxiv.org/pdf/1509.06461.pdf)
* a multi-threaded ["Distributed Architecture"](https://arxiv.org/pdf/1508.04186.pdf) with a unique replay memory though.
* ["Hindsight Experience Replay"](https://arxiv.org/pdf/1707.01495.pdf)

Experiment : **CartPole-v1** :

* Adam
* learning rate : 1e-4
* minibatch size : 128
* replay memory capacity : 25e3
*	prioritized experience replay exponent $\alpha$ : 0.5
* number of thread/worker : 1
* double DQN : [x]
* hindsight experience replay : [ ]

![resultDQN1](/results/DQN/result.png)

## Deep Deterministic Policy Gradient (DDPG)
This implementation of the Deep Deterministic Policy Gradient (["Continuous Control with Deep Reinforcement Learning"](https://arxiv.org/pdf/1509.02971.pdf)) can be augmented with the following features :

* ["Prioritized Experience Replay"](https://arxiv.org/pdf/1511.05952.pdf)
* ["Dueling Deep Q-Network"](https://arxiv.org/pdf/1511.06581.pdf)
* a multi-threaded architecture (["A2C"/"A3C"](https://arxiv.org/pdf/1602.01783.pdf)).
* ["Hindsight Experience Replay"](https://arxiv.org/pdf/1707.01495.pdf)

Experiment : **Pendulum-v0** :

* Adam
* learning rate : 1e-4
* minibatch size : 128
* soft update $\tau$ : 1e-3
* replay memory capacity : 1e6
*	prioritized experience replay exponent $\alpha$ : 0.0 (no priority)
* number of thread/worker : 1
* hindsight experience replay : [ ]

![resultDDPG1](/results/DDPG/result.png)

## Proximal Policy Optimization (PPO)
This implementation of the ["Proximal Policy Optimization Algorithm"](https://arxiv.org/pdf/1707.06347.pdf) can be augmented with the following features :

* ["Prioritized Experience Replay"](https://arxiv.org/pdf/1511.05952.pdf)
* ["Dueling Deep Q-Network"](https://arxiv.org/pdf/1511.06581.pdf)
* a multi-threaded architecture (["A2C"/"A3C"](https://arxiv.org/pdf/1602.01783.pdf)).
* ["Hindsight Experience Replay"](https://arxiv.org/pdf/1707.01495.pdf)

Experiment : **Pendulum-v0** :

* Adam
* learning rate : 1e-6
* minibatch size : 64
* soft update $\tau$ : 1e-3
* replay memory capacity : 25e3
*	prioritized experience replay exponent $\alpha$ : 0.0 (no priority)
* number of thread/worker : 1
* hindsight experience replay : [ ]

![resultPPO1](/results/PPO/100000/result.png)


