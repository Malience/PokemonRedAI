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

import random

def check_bit(data, bit):
    return data & (1 << bit) > 0

class PokeRedEnv(ParallelEnv):
    metadata = {
        "name": "pokemon_red_environment_v0",
        "render_mode": "headless"
    }

    def __init__(self, emulator, save_states):
        super().__init__()
        
        self.emulator = emulator
        self.save_states = save_states
        
        self.reset_count = 0
        
        ### Agents ###
        self.agents = {'default': None}
        self.initial = None
        
        ### Gymnasium Env Setup ###
        R, C = self.emulator.screen.raw_screen_buffer_dims()
        self.obs_size = (R // 2, C // 2)
        
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_size, dtype=np.uint8)
        
        ### Reset ###
        self.reset()
        
    def register_agent(self, multiagent):
        self.agents[multiagent.name] = multiagent
        multiagent.reset(self.emulator, self.state)
        
    def initial_agent(self, agent_id):
        self.initial = agent_id
        
    def reset(self, seed=None, options=None, initial_agents=['default']):
        self.seed = seed
        
        self.emulator.reset(random.choice(self.save_states))
        self.steps = {}
        
        self.state = {}
        
        for agent in self.agents.values():
            if agent is None: continue
            agent.reset(self.emulator, self.state)
        
        if self.initial is not None: initial_agents = [self.initial]
        
        obs = {}
        for agent in initial_agents:
            if agent not in self.agents:
                print(f"ERROR: initial agent not registered - {agent}")
                return
            
            obs[agent] = self.render()
        
        return obs, {}
        
    def step(self, actions):
        render = self.render()
        
        obs, rewards, term, trun, infos = {}, {}, {}, {}, {}
        
        for actor, action in actions.items():
            if actor == 'default':
                self.emulator.run(action if action is not None else SIMPLE_ACTION_SPACE.sample())
                ob, reward, trm, trn, info = {'default': None}, 0, False, False, {}
            else:
                ob, reward, trm, trn, info = self.agents[actor].step(self.emulator, action, self.state)
            
            for o in ob: obs[o] = render
            rewards[actor] = reward
            term[actor] = trm
            trun[actor] = trn
            infos[actor] = info
            
            if actor not in self.steps: self.steps[actor] = 1
            else: self.steps[actor] += 1
        
        return obs, rewards, term, trun, infos
    
    def render(self):
        #return self.emulator.screen.screen_ndarray()[::2, ::2, 0]
        return self.emulator.frames
    
    def check_if_done(self):
        return self.step_count >= self.max_steps