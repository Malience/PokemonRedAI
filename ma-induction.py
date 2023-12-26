import os
from os.path import exists
from pathlib import Path
import uuid

from pokered_env import PokeRedEnv

import supersuit as ss
from gymnasium.spaces import Tuple, Box, Discrete

from explore_low_agent import ExploreLowAgent
from basic_flee_agent import BasicFleeAgent

from emulator import *
from ram_map import *

#from ppo import PPO
import ppo

import torch

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = PokeRedEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

def train_explore(emulator, target_id, states, combat_agent, combat_policy, name):
    env = PokeRedEnv(emulator, states)
    
    explore_policy = ppo.Policy([], SIMPLE_ACTION_SPACE).to('cuda')
    explore_agent = ExploreLowAgent(name, SIMPLE_ACTION_SPACE, target_id)
    
    env.register_agent(explore_agent)
    env.register_agent(combat_agent)
    
    env.initial_agent(name)
    
    policies = {combat_agent.name: combat_policy, name: explore_policy}
    
    ppo.train(policies, env, n_episodes=30, train_agents=[combat_agent.name, name])
    
    return explore_agent, explore_policy
    
#def test_agent(emulator,  

if __name__ == '__main__':
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')
    
    states_path = Path('states')
    states_path.mkdir(exist_ok=True)
    
    gb_path = './PokemonRed.gb'
    init_state = './has_pokedex_nballs.state'
    #init_state = 'states/11_5_0-0.state'
    
    main_emulator = Emulator(sess_path, gb_path, instance_id='main', headless=False, emulation_speed=2)
    main_env = PokeRedEnv(main_emulator, [init_state])
    
    train_emulator = Emulator(sess_path, gb_path, instance_id='train', headless=True)
    
    explore_low_policy = ppo.Policy([], SIMPLE_ACTION_SPACE).to('cuda')
    
    explore_agent = ExploreLowAgent('explore_low_agent', SIMPLE_ACTION_SPACE, 0)
    
    #policies = {'explore_low_agent': explore_low_policy, 'combat_agent': combat_policy}#, 'explore_med_agent': explore_med_policy}
    
    #env.initial_agent('explore_low_agent')
    #env.register_agent(explore_agent)
    #env.register_agent(flee_agent)
    
    #train_data, rewards = ppo.rollout(policies, env)
    
    #print(train_data['explore_low_agent'][0])
    #print(train_data['explore_med_agent'])
    #print(rewards)
    
    #ppo.train(policies, env)
    
    map_states = {}
    #TODO LOADING
    
    agents = {}
    
    policies = {}
    
    flee_policy = ppo.Policy([], SIMPLE_ACTION_SPACE).to('cuda')
    flee_agent = BasicFleeAgent('flee_agent', SIMPLE_ACTION_SPACE)
    combat_policies = {'flee_agent': flee_policy}
    
    
    main_env.register_agent(flee_agent)
    
    
    current_combat_agent = 'flee_agent'
    current_explore_agent = None
    
    request_action = True
    current_target = None
    obs, infos = main_env.reset()
    while True:
        #if agent_enabled: action, _states = model.predict(obs, deterministic=False)
        
        if request_action:  
            inp = input()
            current_target = int(inp)
            
            x, y, n = position(main_emulator.pyboy)
            
            if n == current_target: continue
            
            objective = f'{n}_to_{current_target}'
            agent_id = f'{objective}_agent'
            
            if agent_id in agents:
                #policy = policies[policy_id]
                current_explore_agent = agent_id
                
            else:
                if n not in map_states:
                    map_states[n] = {}
                
                if (x, y, n) not in map_states[n]:
                    state_path = states_path / Path(f'{x}_{y}_{n}-{n}.state')
                    main_emulator.save_state(state_path)
                    
                    map_states[n][(x, y, n)] = state_path
                    
                agent, policy = train_explore(train_emulator, current_target, list(map_states[n].values()), flee_agent, flee_policy, agent_id)
                
                agents[agent_id] = agent
                policies[agent_id] = policy
                
                main_env.register_agent(agents[agent_id])
                
                current_explore_agent = agent_id
            
            
            request_action = False
        
        if battle_type(main_emulator.pyboy) > 0:
            action = combat_policies[current_combat_agent].predict(main_env.render())
            actions = {current_combat_agent: action}
            
        else:
            x, y, n = position(main_emulator.pyboy)
            if n == current_target:
                request_action = True
                continue
                
            action = policies[current_explore_agent].predict(main_env.render())
            actions = {current_explore_agent: action}
        
        obs, rewards, terminated, truncated, info = main_env.step(actions)
        
        #main_emulator.run(act)
        
        
        
        
        #env.render()
        #frame+=1