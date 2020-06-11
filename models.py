import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
	fan_in = layer.weight.data.size()[0]
	lim = 1. / np.sqrt(fan_in)
	return (-lim, lim)

class actor_network(nn.Module):
	def __init__(self, state_size, hidden1_size, hidden2_size, action_size):
		super(actor_network, self).__init__()
		self.fc1 = nn.Linear(state_size,hidden1_size)
		self.fc2 = nn.Linear(hidden1_size,hidden2_size)
		self.fc3 = nn.Linear(hidden2_size,action_size)
		self.reset_parameters()
	
	def reset_parameters(self):
		self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
		self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
		self.fc3.weight.data.uniform_(-3e-3,3e-3)
		
	def forward(self,state):
		x = F.elu(self.fc1(state))
		x = F.elu(self.fc2(x))
		x = F.tanh(self.fc3(x))
		return(x)

class critic_network(nn.Module):
	def __init__(self, state_size, hidden1_size, hidden2_size, action_size):
		super(critic_network, self).__init__()
		self.fc1 = nn.Linear(state_size,hidden1_size)
		self.fc2 = nn.Linear(hidden1_size+action_size,hidden2_size)
		self.fc3 = nn.Linear(hidden2_size,1)
		self.reset_parameters()
		
	def reset_parameters(self):
		self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
		self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
		self.fc3.weight.data.uniform_(-3e-3,3e-3)
		
	def forward(self,state,action):
		#x = torch.cat((state,action),dim=1) 
		#here the action has been included from the 1st layer, unlike DDPG paper which adds it in at 2nd.
		x = F.elu(self.fc1(state))
		x = torch.cat((x,action),dim=1)
		x = F.elu(self.fc2(x))
		x = self.fc3(x)
		return(x)
		
