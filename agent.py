import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import copy
import random
import pickle
from models import *
import os


class replay_buffer():
	def __init__(self,max_size,batch_size):
		self.memory = deque(maxlen=int(max_size))
		self.batch_size = batch_size
		self.experience = namedtuple("Experience", field_names=["states", "actions_taken", "rewards", "next_states", "dones"])

	
	def add(self,state,action,reward,next_state,done):
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)
		#memory 1st index is length, 2nd index is 0:state, 1:action, 2:reward, 3:next_state, 4:done
		
	def sample(self):
		experiences = random.sample(self.memory,k=self.batch_size)
		states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float()
		actions_taken = torch.from_numpy(np.vstack([e.actions_taken for e in experiences if e is not None])).float()
		rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float()
		next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float()
		dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float()
		
		return(states,actions_taken,rewards,next_states,dones)
	
	def __len__(self):
		return(len(self.memory))

class OU_Noise():
	#Generate Ornstein-Uhlenbeck Noise
	#mu = 0, sigma constant
	def __init__(self,action_size,num_agents,theta,sigma):
		self.action_size = action_size
		self.num_agents = num_agents
		self.theta = theta
		self.sigma = sigma
		self.reset()
	def reset(self):
		self.x = np.zeros((self.num_agents,self.action_size))
	def get_noise(self):
		dx = -self.theta * self.x + self.sigma * np.random.randn(self.num_agents,self.action_size)
		self.x = self.x+dx
		return self.x
		
class ddpg_Agent():
	def __init__(self, state_size, action_size, num_agents, hidden1_size, hidden2_size, actor_lr, critic_lr, gamma, tau):
		self.critic_local = critic_network(state_size*num_agents,hidden1_size,hidden2_size,action_size*num_agents)
		self.critic_target = copy.deepcopy(self.critic_local)
		self.actor_local = actor_network(state_size,hidden1_size,hidden2_size,action_size)
		self.actor_target = copy.deepcopy(self.actor_local)

		self.gamma, self.tau = gamma, tau
		
		self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=actor_lr)
		self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=critic_lr)
	
	def get_actions(self,states):
		s = torch.from_numpy(states).float()
		a = self.actor_local(s).detach().numpy()
		return(a)
	
			
	def learn(self,experience_expanded):
		states,actions,rewards,next_states,dones,next_actions,actor_actions = experience_expanded
		
		
		Q_vals = self.critic_local(state=states,action=actions)
		next_Q = self.critic_target(next_states,next_actions)
		Q_target = rewards + self.gamma * next_Q
		
		#update local critic network
		critic_loss = F.mse_loss(Q_vals, Q_target)
		self.critic_optim.zero_grad()
		critic_loss.backward(retain_graph=True)
		self.critic_optim.step()
		
		#update local actor network
		actor_loss = -self.critic_local(states, actor_actions).mean()
		self.actor_optim.zero_grad()
		actor_loss.backward()
		self.actor_optim.step()
		
		#update target networks
		self.actor_target = self.soft_update(self.actor_local, self.actor_target)
		self.critic_target = self.soft_update(self.critic_local, self.critic_target)
		
	def soft_update(self,local,target):
		for name, param in target.state_dict().items():
			l, t = local.state_dict()[name], target.state_dict()[name] #local,target
			new = l * self.tau + t * (1.0-self.tau)
			target.state_dict()[name].copy_(new)
		return(target)
		
class MADDPG():
	def __init__(self,state_size, action_size, num_agents, hidden1_size, hidden2_size, 
				 max_replay_size, batch_size, actor_lr, critic_lr, gamma, tau):
		
		self.num_agents = num_agents
		self.action_size, self.state_size = action_size, state_size
		
		self.agents = []
		for _ in range(self.num_agents):
			#create each agent
			self.agents.append(ddpg_Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, 
									 hidden1_size=hidden1_size, hidden2_size=hidden2_size, 
									 actor_lr=actor_lr, critic_lr=critic_lr, 
									 gamma=gamma, tau=tau))
		
		self.replay_memory = replay_buffer(max_size=max_replay_size, batch_size=batch_size)
		self.batch_size = batch_size
		
	def get_actions(self, all_obs):
		all_actions = []
		for i in range(self.num_agents):
			#use only that agents observations to choose an action
			all_actions.append(self.agents[i].get_actions(all_obs[i,:]))
		all_actions = np.array(all_actions) #size (num_agents,action_size)
		return(all_actions)
	
	def step(self,states,actions,rewards,next_states,dones,updates_per_step):
		#each agent has its own replay buffer, but the states stored include observations from all agents
		
		#flatten states and actions from all agents, ready to be input into critic
		all_states = states.reshape(self.num_agents*self.state_size)
		all_next_states = next_states.reshape(self.num_agents*self.state_size)
		all_actions = actions.reshape(self.num_agents*self.action_size)
		
		self.replay_memory.add(all_states,all_actions,rewards,all_next_states,dones)
		
		#If enough memory, take a random sample from the memory, and learn from it
		if len(self.replay_memory) >= self.batch_size:
			#experience_batch = self.replay_memory.sample()
			for _ in range(updates_per_step):
				for agent_i in range(self.num_agents):
					experience_batch = self.replay_memory.sample()
					self.learn(experience_batch, agent_i)
		
	def learn(self, experiences, agent_no):
		
		all_states,all_actions,rewards,all_next_states,dones = experiences
		#all_states and all_next have shape (batch_size, num_agents*state_size)
		
		next_actions = []
		#get next actions using the target actor of each agent
		for i in range(self.num_agents):  
			ns = all_next_states[:,i:i+self.state_size] #next state (observation) for that agent as a torch float. all batches
			a = self.agents[i].actor_target(ns)
			next_actions.append(a)
		
		next_actions = torch.cat(next_actions,dim=1)
		
		s = all_states[:,agent_no:agent_no+self.state_size] #state (observation) for that agent as a torch float. all batches
		modified_actions = all_actions
		modified_actions[:,agent_no:agent_no+self.action_size] = self.agents[agent_no].actor_local(s) 
		agent_dones = dones[:,agent_no].view(self.batch_size,1)
		agent_rewards = rewards[:,agent_no].view(self.batch_size,1)
		experience_expanded = (all_states,all_actions,agent_rewards,all_next_states,agent_dones,next_actions,modified_actions)
		self.agents[agent_no].learn(experience_expanded)
		
	def checkpoint(self, name, location, ep_no, scores=None):
		
		if not(os.path.exists(location+'/'+name)):
			os.mkdir(location+'/'+name)
			print('created directory ',location+'/'+name)
		loc = location+'/'+name+'/'
		
		for agent_no in range(self.num_agents):
			torch.save(self.agents[agent_no].actor_local.state_dict(),loc+'ep'+str(ep_no)+'_actor_local_agent'+str(agent_no)+'.pth')
			torch.save(self.agents[agent_no].critic_local.state_dict(),loc+'ep'+str(ep_no)+'_critic_local_agent'+str(agent_no)+'.pth')
		if scores:
			with open(loc+'ep'+str(ep_no)+'_scores.pkl', 'wb') as f:
				pickle.dump(scores, f)
	
	def load_checkpoint(self, name, location, ep_no, scores=False):
		loc = location+'/'+name+'/'
		for agent_no in range(self.num_agents):
			self.agents[agent_no].actor_local.load_state_dict(torch.load(loc+'ep'+str(ep_no)+'_actor_local_agent'+str(agent_no)+'.pth'))
			self.agents[agent_no].actor_target.load_state_dict(torch.load(loc+'ep'+str(ep_no)+'_actor_local_agent'+str(agent_no)+'.pth'))
			self.agents[agent_no].critic_local.load_state_dict(torch.load(loc+'ep'+str(ep_no)+'_critic_local_agent'+str(agent_no)+'.pth'))
			self.agents[agent_no].critic_target.load_state_dict(torch.load(loc+'ep'+str(ep_no)+'_critic_local_agent'+str(agent_no)+'.pth'))
		if scores:
			with open(loc+'ep'+str(ep_no)+'_scores.pkl', 'rb') as f:
				scores = pickle.load(f)
			return(scores)
		
