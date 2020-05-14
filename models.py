import torch.nn as nn
import torch
import torch.nn.functional as F

class actor_network(nn.Module):
    def __init__(self, state_size, hidden1_size, hidden2_size, action_size):
        super(actor_network, self).__init__()
        self.fc1 = nn.Linear(state_size,hidden1_size)
        self.fc2 = nn.Linear(hidden1_size,hidden2_size)
        self.fc3 = nn.Linear(hidden2_size,action_size)
        
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return(x)

class critic_network(nn.Module):
    def __init__(self, observation_size, hidden1_size, hidden2_size, action_size, n_agents):
        #assume every agent has same action size and observation size
        super(critic_network, self).__init__()
        self.fc1 = nn.Linear((observation_size+action_size)*n_agents,hidden1_size)
        self.fc2 = nn.Linear(hidden1_size,hidden2_size)
        self.fc3 = nn.Linear(hidden2_size,action_size)
        
    def forward(self,all_observations,all_actions):
        
        x = torch.cat((all_observations,all_actions),dim=1) 
        #here the action has been included from the 1st layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return(x)
		
