
# Udacity DRLND Project 3: Collaboration and Competition

### Introduction
This repository contains code to train a pair of agents to play tennis. It also contains trained agents. The environment used is from [Unity](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis).

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play. The task is episodic, and the score for each episode is the maximum score of the two agents. The aim is an average score of +0.5 over 100 episodes.

The action space for each agent is 2 dimensional, and corresponds to the agent moving forward/backward and jumping. The state space consists of 8 variables detailing the position and velocity of the ball and racket.

The agents are both collaborative and competitive. The algorithm used to train is a Multi Agent Deep Deterministic Policy Gradient algorithm (MADDPG). In this algorithm both agents have an actor and critic, the actor is a neural network used to choose an action based on the observation that agent recieves. The critic predicts the future rewards given the state and action taken. The actor uses only local information available to the agent, whilst the critic uses extra information, in this case the state observed and action taken by the other agent. This extra information is available during training, but at execution time the agents chooses actions based only on its' own observations.


### Getting Started with the environment

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the same folder as this repository, and unzip (or decompress) the file. 


### Repository structure

Two python files (agent.py and models.py) contain the bulk of the code used to complete this task. To train an agent from scratch, run the code in the notebook Tennis.ipynb. A trained model is saved in the checkpoints folder, and in the report notebook this model is loaded and tested.

Note that although the training process saves checkpoints of the model at regular intervals, only one set of weights is svaed in this repository.

