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
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.obs = []
        self.actions = []
        self.action_log_probs = []
        self.values = []
        self.rewards = []
        self.size = 0
        
        self.total_reward = 0
        self.next_value = -1
        
        #self.gaes = None
        self.advantages = None
        self.returns = None
        
        self.term = False
        self.trun = False
        self.success = False
        
        self.ended = False
        self.calculated = False
        
    def step(self, obs, action, action_log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.values.append(value)
        self.size += 1
        
    def add_reward(self, reward):
        self.rewards.append(reward)
        self.total_reward += reward
        
    def end(self, next_value, term=False, trun=False, success=False):
        self.term = term
        self.trun = trun
        self.success = success
        
        self.next_value = next_value

        self.ended = True
        
    #deprecated
    def resize(self, new_size):
        assert new_size > 0
        assert self.size > 0
        
        # No need to resize
        if new_size == self.size: return
        
        # Need to shrink
        if new_size < self.size:
            self.obs = self.obs[:new_size]
            self.actions = self.actions[:new_size]
            self.action_log_probs = self.action_log_probs[:new_size]
            self.values = self.values[:new_size]
            self.rewards = self.rewards[:new_size]
            
        # We need to pad!!!
        else:
            diff = new_size - self.size
            obs_shape = self.obs[0].shape
            self.obs += [np.zeros(obs_shape)] * diff
            self.actions += [-1] * diff
            self.action_log_probs += [0] * diff
            self.values += [0] * diff
            self.rewards += [0] * diff
        
        self.size = new_size
            
     
    def calculate(self, gamma=0.99, gae_lambda=0.95):
        with torch.no_grad():
            #TODO make all of this run on torch, and not a method
            #self.rewards = torch.tensor(self.rewards, dtype=torch.float32)
            self.advantages = [0] * len(self.rewards)#torch.zeros_like(self.rewards)
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

        #self.returns = discount_rewards(self.rewards, discount_gamma)
        #self.gaes = calculate_gaes(self.rewards, self.values, gae_gamma, gae_decay)

    def get(self):
        return self.obs, self.actions, self.action_log_probs, self.values, self.advantages, self.returns

    #deprecated    
    def permute(self, permute_idxs=None):
        if permute_idxs is None:
            permute_idxs = np.random.permutation(self.size)
            
        assert len(permute_idxs) == self.size
        
        arr = self.obs[list(permute_idxs)]

        return (np.array(self.obs)[permute_idxs],
                np.array(self.actions)[permute_idxs],
                np.array(self.action_log_probs)[permute_idxs],
                np.array(self.values)[permute_idxs],
                #self.rewards[permute_idxs],
                np.array(self.advantages)[permute_idxs],
                np.array(self.returns)[permute_idxs])
    
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
            cur_rollouts[agent] = Rollout(agent)
            
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

                        cur_rollouts[agent] = Rollout(agent)
                        
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
    
def compose_rollouts(rollouts):
    length = 0
    
    for rollout in rollouts:
        if rollout.size > length: length = rollout.size
    
    obs, actions, action_log_probs, values, gaes, returns = [], [], [], [], [], []
    
    for rollout in rollouts:
        #rollout.resize(length)
        rollout.calculate()
        r_obs, r_actions, r_action_log_probs, r_values, r_gaes, r_returns = rollout.get()

        # obs.append(r_obs)
        # actions.append(r_actions)
        # action_log_probs.append(r_action_log_probs)
        # values.append(r_values)
        # gaes.append(r_gaes)
        # returns.append(r_returns)

        obs += r_obs
        actions += r_actions
        action_log_probs += r_action_log_probs
        values += r_values
        gaes += r_gaes
        returns += r_returns
        
    #obs = np.concatenate(obs, axis=0)

    obs = torch.cat(obs, 0)
    actions = torch.cat(actions, 0)
    action_log_probs = torch.cat(action_log_probs, 0)
    values = torch.cat(values, 0)
    gaes = torch.cat(gaes, 0)
    returns = torch.cat(returns, 0)

    # obs = torch.tensor(np.concatenate(obs, axis=0), dtype=torch.float32)
    # actions = torch.tensor(np.concatenate(actions, axis=0), dtype=torch.int32)
    # action_log_probs = torch.tensor(np.concatenate(action_log_probs, axis=0), dtype=torch.float32)
    # values = torch.tensor(np.concatenate(values, axis=0), dtype=torch.float32)
    # gaes = torch.tensor(np.concatenate(gaes, axis=0), dtype=torch.float32)
    # returns = torch.tensor(np.concatenate(returns, axis=0), dtype=torch.float32)
        
    return obs, actions, action_log_probs, values, gaes, returns
        
        
        