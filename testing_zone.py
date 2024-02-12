import os
from os.path import exists
from pathlib import Path
import threading
import time
import uuid
from pokered_env import PokeRedEnv
#from tensorboard_callback import TensorboardCallback

import supersuit as ss
from gymnasium.spaces import Tuple, Box, Discrete

from explore_low_agent import ExploreLowAgent
from pokered_vecenv import PokeRedVecEnv
from explore_move_paint import ExploreMovePaintAgent
from basic_flee_agent import BasicFleeAgent

#from ppo import PPO
from ppo import PPOSettings, PPOTrainer
from policy import Policy
from training import train

from emulator import *

import torch

import torch.nn as nn
import torch.optim as optim

def gen(env):
    for i in range(200):
        actions = {'default': MOVEMENT_ACTION_SPACE.sample()}
        env.step(actions)

if __name__ == '__main__':
    torch.set_default_device('cuda') 

    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')
    sess_path.mkdir(exist_ok=True)
    
    states_path = Path('states')
    states_path.mkdir(exist_ok=True)
    
    gb_path = './PokemonRed.gb'
    init_state = './has_pokedex_nballs.state'
    init_state = 'states/11_5_0-0.state'
    
    main_emulator = Emulator(sess_path, gb_path, instance_id='main', headless=True)
    main_emulator2 = Emulator(sess_path, gb_path, instance_id='main', headless=True)
    main_emulator3 = Emulator(sess_path, gb_path, instance_id='main', headless=True)
    main_env = PokeRedEnv(main_emulator, [init_state])
    main_env2 = PokeRedEnv(main_emulator2, [init_state])
    main_env3 = PokeRedEnv(main_emulator3, [init_state])

    emulators = [Emulator(sess_path, gb_path, instance_id=f'main_{i}', headless=True) for i in range(10)]
    vec_env = PokeRedVecEnv(emulators, [init_state])
    

    flee_policy = Policy([], SIMPLE_ACTION_SPACE)
    flee_agent = BasicFleeAgent('flee_agent', SIMPLE_ACTION_SPACE)
    
    explore_policy = Policy([], MOVEMENT_ACTION_SPACE)
    explore_agent = ExploreLowAgent('explore_agent', MOVEMENT_ACTION_SPACE, 12)
    #explore_agent = ExploreMovePaintAgent('explore_agent', MOVEMENT_ACTION_SPACE, 12)
    
    
    main_env.register_agent(flee_agent)
    main_env.register_agent(explore_agent)
    main_env.initial_agent('explore_agent')
    
    main_env2.register_agent(flee_agent)
    main_env2.register_agent(explore_agent)
    main_env2.initial_agent('explore_agent')
    
    main_env3.register_agent(flee_agent)
    main_env3.register_agent(explore_agent)
    main_env3.initial_agent('explore_agent')

    vec_env.register_agent(flee_agent)
    vec_env.register_agent(explore_agent)
    vec_env.initial_agent('explore_agent')
    
    policies = {'explore_agent': explore_policy, 'flee_agent': flee_policy}

    pposettings = PPOSettings()
    ppo = PPOTrainer(policies['explore_agent'], pposettings)
    
    #temp variables
    num_envs = 10
    num_steps = 400
    num_iterations = 100

    train(policies, vec_env, num_envs, 'explore_agent', ppo, num_steps=num_steps, num_iterations=num_iterations, verbose=True)