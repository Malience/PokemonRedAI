import time
import numpy as np
import torch
from torch.distributions.categorical import Categorical

#TODO: Move somewhere else
def discount_rewards(rewards, gamma=0.99):
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])

def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])

class Rollout():
    def __init__(self, agent_id, max_steps, obs_shape, action_shape=(1,)):
        self.agent_id = agent_id
        self.max_steps = max_steps

        self.obs = torch.zeros((max_steps,) + (3, 72, 80)) #TODO make the shape better
        self.actions = torch.zeros((max_steps)) #TODO Add action shapes
        self.logprobs = torch.zeros((max_steps))
        self.rewards = torch.zeros((max_steps))
        self.values = torch.zeros((max_steps))

        self.size = 0
        
        self.total_reward = 0
        self.next_value = -1
        
        self.advantages = None
        self.returns = None
        
        self.term = False
        self.trun = False
        self.success = False
        
        self.ended = False
        self.calculated = False
        
    def step(self, obs, action, logprob, value):
        self.obs[self.size] = obs
        self.actions[self.size] = action
        self.logprobs[self.size] = logprob
        self.values[self.size] = value

        self.size += 1
        
    def add_reward(self, reward):
        self.rewards[self.size - 1] = reward
        self.total_reward += reward
        
    def end(self, next_value, term=False, trun=False, success=False):
        self.term = term
        self.trun = trun
        self.success = success
        
        self.next_value = next_value

        self.ended = True
     
    def calculate(self, gamma=0.99, gae_lambda=0.95):
        with torch.no_grad():
            #TODO make all of this run on torch, and not a method
            self.advantages = torch.zeros_like(self.rewards)
            lastgaelam = 0
            next_done = 1.0 if self.term or self.trun else 0.0
            for t in reversed(range(self.size)):
                if t == self.size - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = self.next_value
                else:
                    nextnonterminal = 1.0
                    nextvalues = self.values[t + 1]
                
                delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
                self.advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            
            self.returns = self.advantages + self.values

    def get(self):
        return (self.obs[:self.size],
                self.actions[:self.size],
                self.logprobs[:self.size],
                self.values[:self.size],
                self.advantages[:self.size],
                self.returns[:self.size])
    
    def __str__(self):
        return f"Rollout - {self.agent_id} - Reward = {self.total_reward}, Success = {self.success}"
            

def generate_rollouts(policies, env, target, count=1, collect=None, max_steps=200, verbose=False):
    '''
        policies: Dict of agent_id to policy
        env: env
        target: Primary policy to collect rollouts for. Will collect rollouts until this policy has received an amount specified by count.
        count:  How many rollouts to collect for the primary target
        collect: Set of Agents to collect, if left as None only collects rollouts for the primary target policy
        max_steps:
    '''

    if collect is None:
        collect = {target}

    rollouts = {}
    for agent in collect:
        rollouts[agent] = []
    
    while True:
        obs, infos = env.reset()
        
        cur_rollouts = {}
        for agent in collect:
            cur_rollouts[agent] = Rollout(agent, max_steps, env.observation_space.shape)
            
        reset = False
        for _ in range(max_steps):
            actions = {}
            for agent in obs.keys():
                agent_obs = obs[agent]
                #TODO Get rid of nparray, optimize, might only work with vecs
                agent_obs = torch.tensor(np.array([agent_obs]), dtype=torch.float32)

                with torch.no_grad():
                    action, log_prob, _, value = policies[agent].get_action_and_value(agent_obs)

                actions[agent] = action.item()
                
                if agent in collect:
                    cur_rollouts[agent].step(agent_obs, action, log_prob, value.flatten())
            
            next_obs, rewards, terms, truns, infos = env.step(actions)
            
            for agent in obs.keys():
                if agent in collect:
                    cur_rollouts[agent].add_reward(rewards[agent])
                    
                    if terms[agent]:
                        succ = True if 'success' in infos[agent] and infos[agent]['success'] else False
                        #TEMP temporary hack to get the next obs if the agent isn't used next frame
                        if agent not in next_obs:
                            temp_obs = next_obs.values()[0]
                        else:
                            temp_obs = next_obs[agent]
                        temp_obs = torch.tensor(np.array([temp_obs]), dtype=torch.float32)


                        with torch.no_grad():
                            next_value = policies[agent].get_value(temp_obs)

                        cur_rollouts[agent].end(next_value, terms[agent], truns[agent], succ)
                        
                        rollouts[agent].append(cur_rollouts[agent])
                        
                        if verbose: print(cur_rollouts[agent])

                        cur_rollouts[agent] = Rollout(agent, max_steps, env.observation_space.shape)
                        
                        if agent == target:
                            reset = True
                            continue
                            
            obs = next_obs
                        
            if reset: break

        for agent in cur_rollouts.keys():
            if cur_rollouts[agent].size > 0:
                cur_rollouts[agent].end(False, False, False)
                rollouts[agent].append(cur_rollouts[agent])
                if verbose: print(cur_rollouts[agent])
                    

        if len(rollouts[target]) >= count: break
    return rollouts

def obs_encoder(obs):
    new_obs = {}
    encoding = {}
    
    for i in range(len(obs)):
        for agent in obs[i].keys():
            if agent not in new_obs:
                new_obs[agent] = []
                encoding[agent] = []
            
            new_obs[agent].append(obs[i][agent])
            encoding[agent] += [i]
            
    for agent in new_obs:
        new_obs[agent] = torch.tensor(np.array(new_obs[agent]), dtype=torch.float32)
        
    return new_obs, encoding

def generate_rollouts_vec(policies, env, target, count=1, collect=None, max_steps=200, verbose=False):
    '''
        policies: Dict of agent_id to policy
        env: vector env
        target: Primary policy to collect rollouts for. Will collect rollouts until this policy has received an amount specified by count.
        count:  How many rollouts to collect for the primary target
        collect: Set of Agents to collect, if left as None only collects rollouts for the primary target policy
        max_steps:
    '''

    if collect is None:
        collect = {target}

    rollouts = {}
    for agent in collect:
        rollouts[agent] = []

    cur_rollouts = [{} for i in range(env.num_envs)]
    for i in range(env.num_envs):
        for agent in collect:
            cur_rollouts[i][agent] = Rollout(agent, max_steps, env.observation_space.shape)
            
    cur_steps = np.zeros((env.num_envs,))

    obs, infos = env.reset()

    time_spent_stepping = 0

    while True:
        new_obs, encoding = obs_encoder(obs)

        actions = [{} for i in range(env.num_envs)]
        for agent in new_obs.keys():
            agent_obs = new_obs[agent]
            #agent_obs = torch.tensor(np.array([agent_obs]), dtype=torch.float32)
            with torch.no_grad():
                action, log_prob, _, value = policies[agent].get_action_and_value(agent_obs)

            for i in range(len(encoding[agent])):
                e = encoding[agent][i]
                actions[e][agent] = action[i].item()
                if agent in collect:
                    cur_rollouts[e][agent].step(agent_obs[i], action[i], log_prob[i], value[i].flatten())

        start = time.time()

        next_obs, rewards, terms, truns, infos = env.step(actions)

        time_spent_stepping += time.time() - start

        cur_steps += 1

        for i in range(env.num_envs):
            def end_rollout(succ, agent):
                if agent not in next_obs[i]:
                    temp_obs = next_obs[i].values()[0] #TEMP
                else:
                    temp_obs = next_obs[i][agent]
                temp_obs = torch.tensor(np.array([temp_obs]), dtype=torch.float32)

                with torch.no_grad():
                    next_value = policies[agent].get_value(temp_obs)

                cur_rollouts[i][agent].end(next_value, terms[i][agent], truns[i][agent], succ)
                
                rollouts[agent].append(cur_rollouts[i][agent])
                
                if verbose: print(cur_rollouts[i][agent])

                cur_rollouts[i][agent] = Rollout(agent, max_steps, env.observation_space.shape)
                env.reset(envs=[i])
                cur_steps[i] = 0

            for agent in next_obs[i].keys():
                if agent in collect:
                    cur_rollouts[i][agent].add_reward(rewards[i][agent])

                    if terms[i][agent]:
                        succ = True if 'success' in infos[i][agent] and infos[i][agent]['success'] else False
                        end_rollout(succ, agent)
                        
            
            if cur_steps[i] >= max_steps - 1:
                for agent in cur_rollouts[i].keys():
                    if agent in collect:
                        end_rollout(False, agent)
                        
        obs = next_obs
                    
        if len(rollouts[target]) >= count:
            break
        
    if verbose:
        print(f"Time spent stepping: {time_spent_stepping}")

    return rollouts
    
def compose_rollouts(rollouts):
    length = 0
    
    for rollout in rollouts:
        if rollout.size > length: length = rollout.size
    
    obs, actions, action_log_probs, values, gaes, returns = [], [], [], [], [], []
    
    for rollout in rollouts:
        rollout.calculate()
        r_obs, r_actions, r_action_log_probs, r_values, r_gaes, r_returns = rollout.get()

        obs.append(r_obs)
        actions.append(r_actions)
        action_log_probs.append(r_action_log_probs)
        values.append(r_values)
        gaes.append(r_gaes)
        returns.append(r_returns)

    obs = torch.cat(obs, 0)
    actions = torch.cat(actions, 0)
    action_log_probs = torch.cat(action_log_probs, 0)
    values = torch.cat(values, 0)
    gaes = torch.cat(gaes, 0)
    returns = torch.cat(returns, 0)

    return obs, actions, action_log_probs, values, gaes, returns