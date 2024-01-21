import importlib
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# layer_init from CleanRL
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        
        #TODO: Dict spaces
        
        self.obs_layer = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(1920, 512)),
            nn.ReLU(),
        )
        
        self.actor = layer_init( nn.Linear(512, action_space.n))
        self.critic = layer_init(nn.Linear(512, 1))
        
    def get_value(self, x):
        hidden = self.obs_layer(x / 255.0)#torch.permute(obs, (0, 3, 1, 2)))
        value = self.critic(hidden)
        return value
        
    def get_action_and_value(self, x, action=None):
        hidden = self.obs_layer(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

#deprecated
def discount_rewards(rewards, gamma=0.99):
    print(rewards)
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])

#deprecated 
def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]
    
    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])
        
    return np.array(gaes[::-1])

class PPOTrainer():
    def __init__(self,
                policy_network,
                ppo_clip_val=0.2,
                target_kl_div=0.01,
                max_policy_train_iters=80,
                value_train_iters=80,
                policy_lr=3e-4,
                value_lr=1e-2):
        
        self.policy = policy_network
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters
        
        policy_params = list(self.policy.obs_layer.parameters()) + \
            list(self.policy.actor.parameters())
        self.policy_optimizer = optim.Adam(policy_params, lr=policy_lr)
        
        value_params = list(self.policy.obs_layer.parameters()) + \
            list(self.policy.critic.parameters())
        self.value_optimizer = optim.Adam(value_params, lr=value_lr)
        
    def train_policy(self, obs, acts, old_log_probs, gaes):
        for _ in range(self.max_policy_train_iters):
            self.policy_optimizer.zero_grad()
            
            new_logits = self.policy.action(obs)
            new_logits = Categorical(logits=new_logits)
            new_log_probs = new_logits.log_prob(acts)
            
            policy_ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = policy_ratio.clamp(1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
            
            clipped_loss = clipped_ratio * gaes
            full_loss = policy_ratio * gaes
            policy_loss = -torch.min(full_loss, clipped_loss).mean()
            
            policy_loss.backward()
            self.policy_optimizer.step()
            
            kl_div = (old_log_probs - new_log_probs).mean()
            if kl_div >= self.target_kl_div: break
        
        
    def train_value(self, obs, returns):
        for _ in range(self.value_train_iters):
            self.value_optimizer.zero_grad()
            
            values = self.policy.value(obs)
            value_loss = (returns - values) ** 2
            value_loss = value_loss.mean()
            
            value_loss.backward()
            self.value_optimizer.step()

def obs_encoder(obs):
    new_obs = {}
    encoding = {}
    
    for i in range(len(obs)):
        for agent in obs[i].keys():
            if agent not in new_obs:
                new_obs[agent] = []
                encoding[agent] = []
            
            new_obs[agent].append(obs[i][agent] / 255.0)
            encoding[agent] += [i]
            
    for agent in new_obs:
        new_obs[agent] = torch.tensor(np.array(new_obs[agent]), dtype=torch.float32)
        
    return new_obs, encoding
    

def rollout(policies, env, max_steps=10):
    #train_data = [[], [], [], [], []]
    train_data = {} #This needs it's own class
    ep_rewards = {}
    
    obs, _ = env.reset()
    
    nobs, encoding = obs_encoder(obs)
    
    for _ in range(max_steps):
        actions = [{}] * env.num_envs
        for agent in nobs.keys():
            if agent not in train_data: 
                train_data[agent] = [[], [], [], [], []]
                ep_rewards[agent] = 0
        
            #ob = obs[agent] / 255.0
            #logits, val = policies[agent](torch.tensor(np.array([ob]), dtype=torch.float32, device='cuda'))
            logits, vals = policies[agent](nobs[agent])
            act_distribution = Categorical(logits=logits)
            acts = act_distribution.sample()
            act_log_probs = act_distribution.log_prob(acts)#.item()
            
            #act = act.item()
            #val = val.item()
            # Decode the actions
            actsi = acts.tolist()
            assert len(actsi) == len(encoding[agent])
            for i in encoding[agent]:
                actions[i][agent] = actsi[i] 
            
            train_data[agent][0].append(nobs[agent].tolist())
            train_data[agent][1].append(acts.tolist())
            #train_data[agent][2].append(rew)
            train_data[agent][3].append(vals.tolist())
            train_data[agent][4].append(act_log_probs.tolist())
            
        next_obs, rewards, done, _, _ = env.step(actions)
        
        arewards = {}
        for rew in rewards:
            for agent in rew.keys():
                if agent not in arewards:
                    arewards[agent] = []
                arewards[agent].append(rew[agent])
                
        for agent in arewards.keys():
            train_data[agent][2].append(arewards[agent])
            ep_rewards[agent] += sum(arewards[agent])
            
        #obs = next_obs
        nobs, encoding = obs_encoder(next_obs)
        #print(done)
        #if done: break
        
    #for agent in env.possible_agents:
    #train_data = [np.asarray(x) for x in train_data]
    
    ### DO FILTERING!!!!
    for agent in train_data.keys():
        train_data[agent][3] = calculate_gaes(train_data[agent][2], train_data[agent][3])
    
    return train_data, ep_rewards

#deprecated
def train(policies, env, n_episodes=2048, train_agents=None):
    ppos = {}
    ep_rewards = {}
    
    if train_agents is None: train_agents = env.agents
    
    for agent in train_agents:
        #ep_rewards[agent] = []
        ppos[agent] = PPOTrainer(
            policies[agent],
            policy_lr=1e-5,
            value_lr=1e-3,
            target_kl_div=0.02,
            max_policy_train_iters=40,
            value_train_iters=40)

    for episode_idx in range(n_episodes):
        train_data, rewards = rollout(policies, env)
        
        for agent in train_data.keys(): #For every agent that was actually trained on
            if agent not in ep_rewards: ep_rewards[agent] = []
            ep_rewards[agent].append(rewards[agent])
            
            # Shuffle
            permute_idxs = np.random.permutation(len(train_data[agent][0]))
            
            print(np.array(train_data[agent][1]))
            
            # Policy data
            obs = torch.tensor(np.array(train_data[agent][0])[permute_idxs], dtype=torch.float32, device='cuda')
            acts = torch.tensor(np.array(train_data[agent][1])[permute_idxs], dtype=torch.int32, device='cuda')
            gaes = torch.tensor(np.array(train_data[agent][3])[permute_idxs], dtype=torch.float32, device='cuda')
            act_log_probs = torch.tensor(np.array(train_data[agent][4])[permute_idxs], dtype=torch.float32, device='cuda')
            
            # Value data
            returns =  (np.array(train_data[agent][2]))[permute_idxs]
            returns = torch.tensor(returns, dtype=torch.float32, device='cuda')
            
            # Train model
            ppos[agent].train_policy(obs, acts, act_log_probs, gaes)
            ppos[agent].train_value(obs, returns)
            
        if (episode_idx + 1) % 1 == 0:
            
            print(f'~~~~~ Episode {episode_idx} ~~~~~')
            
            for agent in train_agents:
                if agent not in ep_rewards: continue
                rewards = np.mean(ep_rewards[agent][-1])
                print(f'{agent} - Avg Reward = {rewards:.1f}')
                
            print(f'~~~~~~~~~~~~~~~~~~~~')
        
    
    