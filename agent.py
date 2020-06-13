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
		self.experience = namedtuple("Experience", field_names=["all_states","states","actions","rewards","all_next_states","next_states","dones"])

	
	def add(self,all_states,states,actions,rewards,all_next_states,next_states,dones):
		#all_states,states,actions,rewards,all_next_states,next_states,dones
		#print(next_states.shape,' add func')
		e = self.experience(all_states,states,actions,rewards,all_next_states,next_states,dones)
		self.memory.append(e)
		#memory 1st index is length, 2nd index is 0:state, 1:action, 2:reward, 3:next_state, 4:done
		
	def sample(self):
		experiences = random.sample(self.memory,k=self.batch_size)
		
		all_states = torch.from_numpy(np.array([e.all_states for e in experiences if e is not None])).float()
		states = torch.from_numpy(np.array([e.states for e in experiences if e is not None])).float()
		actions = torch.from_numpy(np.array([e.actions for e in experiences if e is not None])).float()
		rewards = torch.from_numpy(np.array([e.rewards for e in experiences if e is not None])).float()
		all_next_states = torch.from_numpy(np.array([e.all_next_states for e in experiences if e is not None])).float()
		next_states = torch.from_numpy(np.array([e.next_states for e in experiences if e is not None])).float()
		dones = torch.from_numpy(np.array([e.dones for e in experiences if e is not None]).astype(np.uint8)).float()
		#print(next_states.shape,' sample_func')
		return(all_states,states,actions,rewards,all_next_states,next_states,dones)
	
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
		
class gaussian_noise():
	def __init__(self,action_size,num_agents,mean,stdev):
		self.action_size = action_size
		self.num_agents = num_agents
		self.stdev = stdev
		self.mean = mean
		self.reset()
	def reset(self):
		#for consistency with OU noise function
		self.x = self.mean + self.stdev * np.random.randn(self.num_agents,self.action_size)
	def get_noise(self):
		self.x = self.mean + self.stdev * np.random.randn(self.num_agents,self.action_size)
		return(self.x)
		
		
class ddpg_Agent():
	def __init__(self, state_size, action_size, num_agents, hidden1_size, hidden2_size, actor_lr, critic_lr, gamma, tau, batch_norm=True):
		self.critic_local = critic_network(state_size*num_agents,hidden1_size,hidden2_size,action_size*num_agents)
		self.critic_target = copy.deepcopy(self.critic_local)
		self.actor_local = actor_network(state_size,hidden1_size,hidden2_size,action_size)
		self.actor_target = copy.deepcopy(self.actor_local)
		#could try replacing these with a hard update
		self.gamma, self.tau = gamma, tau
		
		self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=actor_lr)
		self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=critic_lr)
		
		self.BatchNorm=batch_norm
	
	def get_actions(self,states):
		s = torch.from_numpy(states).float()
		self.actor_local.eval()
		with torch.no_grad():
			a = self.actor_local(s).detach().numpy()
		self.actor_local.train()
		return(a)
	
			
	def learn(self,experience_expanded):
		all_states, actor_all_actions, all_actions_taken, reward_agent, done_agent, all_next_states, all_next_actions = experience_expanded
		#states,actions,rewards,next_states,dones,next_actions,actor_actions = experience_expanded
		
		
		next_Q = self.critic_target(all_next_states,all_next_actions)

		if self.BatchNorm:
			reward_mean = torch.mean(reward_agent)
			reward_std  = torch.std(reward_agent)
			if reward_std>0:
				reward_agent_norm = (reward_agent-reward_mean)/reward_std
			else:
				reward_agent_norm = (reward_agent-reward_mean)
			Q_target = reward_agent_norm + self.gamma * next_Q * (1-done_agent)
			
		else:
			Q_target = reward_agent + self.gamma * next_Q * (1-done_agent)
		
		Q_vals = self.critic_local(state=all_states,action=all_actions_taken)
		
		
		#update local critic network
		critic_loss = F.mse_loss(input=Q_vals, target=Q_target)
		self.critic_optim.zero_grad()
		critic_loss.backward()
		self.critic_optim.step()
		
		#update local actor network
		actor_loss = -self.critic_local(all_states, actor_all_actions).mean()
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
				 max_replay_size, batch_size, actor_lr, critic_lr, gamma, tau, batch_norm=True):
		
		self.num_agents = num_agents
		self.action_size, self.state_size = action_size, state_size
		self.all_action_size = self.action_size*self.num_agents
		self.agents = []
		for _ in range(self.num_agents):
			#create each agent
			single_ddpg_agent = ddpg_Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, 
									 hidden1_size=hidden1_size, hidden2_size=hidden2_size, 
									 actor_lr=actor_lr, critic_lr=critic_lr, 
									 gamma=gamma, tau=tau, batch_norm=batch_norm)
			self.agents.append(single_ddpg_agent)
		
		self.replay_memory = replay_buffer(max_size=max_replay_size, batch_size=batch_size)
		self.batch_size = batch_size
		
		self.learn_in_step = True 
		self.learn_step_count = 0
		#True= a learn step is taken when step is called
		#False = learn step not taken, (but experience added to replay memory)
		
	def get_actions(self, all_obs):
		all_actions = []
		for i in range(self.num_agents):
			#use only that agents observations to choose an action
			all_actions.append(self.agents[i].get_actions(all_obs[i,:]))
		all_actions = np.array(all_actions) #size (num_agents,action_size)
		return(all_actions)
	
	def step(self,states,actions,rewards,next_states,dones,updates_per_step):
		
		
		#flatten states and actions from all agents, ready to be input into critic
		all_states = np.reshape(states,newshape=(-1))
		all_next_states = np.reshape(next_states,newshape=(-1))
		#all_actions = actions.reshape(self.num_agents*self.action_size)
		
		self.replay_memory.add(all_states,states,actions,rewards,all_next_states,next_states,dones)
		
		#If enough memory, take a random sample from the memory, and learn from it
		if len(self.replay_memory) >= self.batch_size and self.learn_in_step:
			for _ in range(updates_per_step):
				self.learn_step_count +=1
				for agent_i in range(self.num_agents):
					experience_batch = self.replay_memory.sample()
					self.learn(experience_batch, agent_i)
		
	def learn(self, experiences, agent_no):
		
		all_states,states,actions,rewards,all_next_states,next_states,dones = experiences
		#all_states and all_next have shape (batch_size, num_agents*state_size)
		all_next_actions = torch.zeros(states.shape[:2] + (self.action_size,),dtype=torch.float)
		#get next actions using the target actor of each agent
		for i in range(self.num_agents):  
			ns = next_states[:,i,:] #next state (observation) for that agent as a torch float. all batches
			all_next_actions[:,i,:] = self.agents[i].actor_target.forward(ns)
		all_next_actions = all_next_actions.view(-1,self.all_action_size)
		
		agent = self.agents[agent_no]
		agent_state = states[:,agent_no,:]
		
		#get actions with this agent's modified to be that given by the actor network
		actor_all_actions = copy.deepcopy(actions)
		actor_all_actions[:,agent_no,:] = agent.actor_local(agent_state) 
		actor_all_actions = actor_all_actions.view(-1,self.all_action_size)
		
		all_actions_taken = actions.view(-1,self.all_action_size)
		
		done_agent = dones[:,agent_no].view(self.batch_size,1)
		reward_agent = rewards[:,agent_no].view(self.batch_size,1)
		experience_expanded = (all_states, actor_all_actions, all_actions_taken, reward_agent, done_agent, all_next_states, all_next_actions)
		agent.learn(experience_expanded)
		
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
		
