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

def add_noise(arr, magnitude):
    '''
    Returns array with added noise. Noise is gaussian noise with mean 0 and standard deviation of magnitude
    '''
    noise = np.random.randn(*arr.shape)
    new = arr + magnitude*noise
    return new

class replay_buffer():
    def __init__(self, max_size, batch_size):
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)
        
        self.memory = deque(maxlen=self.max_size)
        self.experiences = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        
    def add(self, states, actions, rewards, next_states, dones):
        e = self.experiences(states, actions, rewards, next_states, dones)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        s, ns, a, r, d = [], [], [], [], []
        for e in experiences:
            s.append(torch.from_numpy(e.states).float())
            a.append(torch.from_numpy(e.actions).float())
            r.append(torch.from_numpy(np.array(e.rewards)).float())
            ns.append(torch.from_numpy(e.next_states).float())
            d.append(torch.from_numpy(np.array(e.dones,dtype=np.float64)).float())
        return(torch.stack(s), torch.stack(a), torch.stack(r), torch.stack(ns), torch.stack(d)) #stack adds new dimension in at 0 (left)
    
    def __len__(self):
        return(len(self.memory))
		
class agent:
    def __init__(self, num_agents, observation_size, action_size,
                 actor_hidden1_size, actor_hidden2_size,
                 critic_hidden1_size, critic_hidden2_size,
                 noise_start, noise_decay, noise_min,
                 replay_buffer_size, replay_batch_size,
                 gamma, critic_LR, actor_LR, tau):
        
        self.num_agents = num_agents
        self.action_size = action_size
        self.observation_size = observation_size
        
        self.gamma = gamma
        self.tau = tau
        
        self.actors_local = []
        self.critics_local = []
        self.actor_optims = []
        self.critic_optims = []
        for i in range(self.num_agents):
            self.actors_local.append(actor_network(observation_size, actor_hidden1_size, actor_hidden2_size, action_size))
            self.critics_local.append(critic_network(observation_size, critic_hidden1_size, critic_hidden2_size, action_size, self.num_agents))
            self.actor_optims.append(optim.Adam(self.actors_local[i].parameters(), lr=actor_LR))
            self.critic_optims.append(optim.Adam(self.critics_local[i].parameters(), lr=critic_LR))
        
        self.actors_target = copy.deepcopy(self.actors_local)
        self.critics_target = copy.deepcopy(self.critics_local)
        
        self.noise_magnitude = noise_start
        self.noise_decay, self.noise_min = noise_decay, noise_min
        self.step_count = 0
        
        self.replay_batch_size = replay_batch_size
        self.replay_buffer = replay_buffer(replay_buffer_size, replay_batch_size)
        

    
    def get_actions(self, states, noise=True):
        '''
        Given array of all observations (states) for all agents (n_agents,state_size), return actions using each agent's policy
        '''
        actions = []
        for i in range(self.num_agents):
            state = states[i,:]
            state = torch.from_numpy(state).float()
            action = self.actors_local[i](state).detach().numpy()
            actions.append(action)
        
        actions = np.array(actions) 
        #shape(n_agents,action_size)
        
        if noise:
            actions = add_noise(actions,self.noise_magnitude)
            
        return(np.array(actions))
    
    def decay_noise(self):
        self.noise_magnitude = max(self.noise_magnitude*self.noise_decay, self.noise_min)
    
    def step(self, states, actions, rewards, next_states, dones):
        self.replay_buffer.add(states, actions, rewards, next_states, dones)
        if len(self.replay_buffer) >= self.replay_batch_size:
            for i in range(self.num_agents):
                e = self.replay_buffer.sample()
                self.learn(e, i)
                
    def learn(self, experience, agent_no):
        states, actions, rewards, next_states, dones = experience
        #states and next states: (batch_size, n_agents, observation_size)
        #actions:                (batch_size, n_agents, action_size)
        #rewards and dones       (batch_size, n_agents)

        next_actions = []
        for k in range(self.num_agents):
            local_obs = next_states[:,k,:]
            #local_obs = torch.from_numpy(local_obs).float()
            na = self.actors_target[k](local_obs)
            #size (batch_size,action_size)
            next_actions.append(na)
            
        #update local critic
        states_flat = states.view(self.replay_batch_size,self.num_agents*self.observation_size)
        actions_flat = actions.view(self.replay_batch_size,self.num_agents*self.action_size)
        Q_vals = self.critics_local[agent_no](states_flat,actions_flat) #size (batch_size, action_size)
        
        next_actions = torch.stack(next_actions,dim=1).view(self.replay_batch_size,self.num_agents*self.action_size)
        flat_next_states = next_states.view(self.replay_batch_size,self.num_agents*self.observation_size)
        Q_next = self.critics_target[agent_no](flat_next_states, next_actions) #size (batch_size, action_size)
        Q_target = rewards[:,agent_no].view(self.replay_batch_size,1) + self.gamma * Q_next
        
        critic_loss = F.mse_loss(Q_vals,Q_target)
        self.critic_optims[agent_no].zero_grad
        critic_loss.backward()
        self.critic_optims[agent_no].step()

        #update local actor
        actions_to_critic = actions #actions of every agent
        actions_to_critic[:,agent_no,:] = self.actors_local[agent_no](states[:,agent_no,:]) #replace values for this agent with value from policy
        actions_to_critic = actions_to_critic.view(self.replay_batch_size,self.num_agents*self.action_size)

        actor_loss = -self.critics_local[agent_no](states_flat,actions_to_critic).mean()
        self.actor_optims[agent_no].zero_grad()
        actor_loss.backward()
        self.actor_optims[agent_no].step()
        
        #soft update target networks
        self.actors_target[agent_no] = self.soft_update(self.actors_local[agent_no], self.actors_target[agent_no], self.tau)
        self.critics_target[agent_no] = self.soft_update(self.critics_local[agent_no], self.critics_target[agent_no], self.tau)
        
        self.step_count +=1
        
    def soft_update(self,local,target,tau):
        for name, param in target.state_dict().items():
            l, t = local.state_dict()[name], target.state_dict()[name] #local,target
            new = l * tau + t * (1.0-tau)
            target.state_dict()[name].copy_(new)
        return(target)
    
    def save_checkpoints(self, root, run_name, episode_number, scores=None):
        '''
        In the directory root, creates file run_name if it doesn't exist already. Saves all the model weights (state_dict) 
        for all the actor and critic local networks (i.e. one of each per agent).
        Optional scores argument will also save a scores file in same location as a pickle object
        '''
        
        #create sub-folder for this run if it doesn't exist already
        if not(os.path.exists(root+'/'+run_name)):
            os.mkdir(root+'/'+run_name)
        
        for i in range(self.num_agents):
            torch.save(self.actors_local[i].state_dict(), root+'/'+run_name+'/actor-agent_%s-episode_%s.pth'%(i,episode_number))
            torch.save(self.critics_local[i].state_dict(), root+'/'+run_name+'/critic-agent_%s-episode_%s.pth'%(i,episode_number))
        
        if scores!=None:
            with open(root+'/'+run_name+'/scores-episode_'+episode_number+'.pkl', 'wb') as f:
                pickle.dump(scores,f)
    
    
    def load_checkpoints(self, root, run_name, episode_number, scores_bool=False):
        for i in range(self.num_agents):
            self.actors_local[i].load_state_dict(torch.load(root+'/'+run_name+'/actor-agent_%s-episode_%s.pth'%(i,episode_number)))
            self.critics_local[i].load_state_dict(torch.load(root+'/'+run_name+'/critic-agent_%s-episode_%s.pth'%(i,episode_number)))
        if scores_bool:
            scores = pickle.load(open(root+'/'+run_name+'/scores-episode_'+episode_number+'.pkl','rb'))
            return scores