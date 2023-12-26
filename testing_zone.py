import os
from os.path import exists
from pathlib import Path
import uuid
from pokered_env import PokeRedEnv
#from tensorboard_callback import TensorboardCallback

import supersuit as ss
from gymnasium.spaces import Tuple, Box, Discrete

from explore_low_agent import ExploreLowAgent
from basic_flee_agent import BasicFleeAgent

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

if __name__ == '__main__':


    use_wandb_logging = False
    ep_length = 2048 * 10
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')
    
    ## Absolutely dumb stupid pathing I have to do because HEAVEN FORBID I CAN JUST RUN MY PROGRAM IN A NORMAL FOLDER ##
    ## Note to all other developers, don't do this complete and total garbage ##
    folder = os.getcwd()
    
    env_config = {
                'headless': True,
                'action_freq': 24, 'init_state': f'{folder}/has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': f'{folder}/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': True, 'reward_scale': 4, 'extra_buttons': False,
                'explore_weight': 3 # 2.5
            }
    
    print(env_config)
    
    num_cpu = 8  # Also sets the number of episodes per training iteration
    #env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    env = PokeRedEnv(env_config)
    #obs, infos = env.reset(42, {})
    
    #x = torch.Tensor(obs['explore_low_agent'])
    
    #checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path, name_prefix='poke')
    
    callbacks = []#[checkpoint_callback]#, TensorboardCallback()]

    #env_checker.check_env(env)
    learn_steps = 40
    # put a checkpoint here you want to start from
    file_name = 'session_43b4837c/poke_327680_steps' 
    
    # Literally every single supersuit wrapper doesn't work, wow
    env = env
    #env = ss.max_observation_v0(env, 2)
    #env = ss.frame_skip_v0(env, 4)
    #env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    #env = ss.color_reduction_v0(env, mode="B")
    #env = ss.resize_v1(env, x_size=84, y_size=84)
    #env = ss.frame_stack_v1(env, 4)
    #env = ss.agent_indicator_v0(env, type_only=False)
    #env = ss.pettingzoo_env_to_vec_env_v1(env)
    #envs = ss.concat_vec_envs_v1(env, 16 // 2, num_cpus=0, base_class="gym")
    #envs.single_observation_space = envs.observation_space
    #envs.single_action_space = envs.action_space
    #envs.is_vector_env = True
    #envs = gym.wrappers.RecordEpisodeStatistics(envs)
    
    #model = PPO(env)
    #model.train(100)
    
    simple_space = Discrete(len(env.emulator.simple_actions))
    
    explore_low_policy = ppo.Policy([], simple_space).to('cuda')
    #explore_med_policy = ppo.Policy([], env.action_spaces['explore_med_agent']).to('cpu')
    combat_policy = ppo.Policy([], simple_space).to('cuda')
    
    explore_agent = ExploreLowAgent('explore_low_agent', simple_space, 0)
    flee_agent = BasicFleeAgent('combat_agent', simple_space)
    
    policies = {'explore_low_agent': explore_low_policy, 'combat_agent': combat_policy}#, 'explore_med_agent': explore_med_policy}
    
    env.initial_agent('explore_low_agent')
    env.register_agent(explore_agent)
    env.register_agent(flee_agent)
    
    #train_data, rewards = ppo.rollout(policies, env)
    
    #print(train_data['explore_low_agent'][0])
    #print(train_data['explore_med_agent'])
    #print(rewards)
    
    ppo.train(policies, env)