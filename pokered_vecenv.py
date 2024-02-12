import sys
import uuid 
import os
from math import floor, sqrt
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

from gymnasium import Env, spaces
from pettingzoo.utils.env import ParallelEnv

from emulator import *
from pokered_env import PokeRedEnv

import threading

import random

def check_bit(data, bit):
    return data & (1 << bit) > 0



class PokeRedVecEnv(ParallelEnv):
    metadata = {
        "name": "pokemon_red_environment_v0",
        "render_mode": "headless"
    }

    def __init__(self, emulators, save_states):
        super().__init__()
        
        self.emulators = emulators
        self.save_states = save_states
        
        self.num_envs = len(emulators)

        self.envs = []
        
        for i in range(self.num_envs):
            self.envs += [PokeRedEnv(self.emulators[i], self.save_states)]

        self.observation_space = self.envs[0].observation_space
        
        self.reset()
        
    def register_agent(self, multiagent):
        for env in self.envs:
            env.register_agent(multiagent)
        
    def initial_agent(self, agent_id):
        for env in self.envs:
            env.initial_agent(agent_id)
        
    def reset(self, envs = None, seed=None, options=None, initial_agents=['default']):
        obs = []
        infos = []
        
        if envs is None:
            envs = self.envs
        elif envs is int:
            envs = [self.envs[envs]]
        else:
            envs = [self.envs[i] for i in envs]

        for env in envs:
            ob, inf = env.reset(seed, options, initial_agents)
            obs += [ob]
            infos += [inf]
            
        return obs, infos

    def step(self, actions):
        assert len(actions) == self.num_envs, f"Received incorrect number of actions = {len(actions)}"
    
        obs, rewards, term, trun, infos = [{}] * self.num_envs, [{}] * self.num_envs, [False] * self.num_envs, [False] * self.num_envs, [{}] * self.num_envs

        for i in range(self.num_envs):
            ob, rew, trm, trn, inf = self.envs[i].step(actions[i])
            obs[i] = ob
            rewards[i] = rew
            term[i] = trm
            trun[i] = trn
            infos[i] = inf
        
        return obs, rewards, term, trun, infos
    
    def render(self):
        return [env.render() for env in self.envs]
    



    ### Defunct threading code, incase I need it later, seems to only increase the step time v.v
# def step_threaded(i, env, actions, obs, rewards, term, trun, infos):
#     ob, rew, trm, trn, inf = env.step(actions[i])
#     obs[i] = ob
#     rewards[i] = rew
#     term[i] = trm
#     trun[i] = trn
#     infos[i] = inf
#         threads = [None] * self.num_envs

#         for i in range(self.num_envs):
#             threads[i] = threading.Thread(target=step_threaded, args=(i, self.envs[i], actions, obs, rewards, term, trun, infos,))
#             threads[i].start()
        
#         for i in range(self.num_envs):
#             threads[i].join()