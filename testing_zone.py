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
import ppo
from rollout import *

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
    

    flee_policy = ppo.Policy([], SIMPLE_ACTION_SPACE)
    flee_agent = BasicFleeAgent('flee_agent', SIMPLE_ACTION_SPACE)
    
    explore_policy = ppo.Policy([], MOVEMENT_ACTION_SPACE)
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


    #ppo.train(policies, env)
    
    # start = time.time()

    # #rollouts = generate_rollouts(policies, main_env, 'explore_agent', count=10, max_steps=1000, verbose=True)

    # print(f"Non Vec Time: {time.time()-start:.4f}")

    # start = time.time()

    # rollouts = generate_rollouts_vec(policies, vec_env, 'explore_agent', count=10, max_steps=1000, verbose=True)

    # print(f"Vec Time: {time.time()-start:.4f}")
    #for _ in range(10): gen(main_env)
    #gen(main_env2)
    #gen(main_env3)

    # t = [threading.Thread(target=gen, args=[main_env]) for _ in range(10)]

    # for ti in t: ti.start()
    # for ti in t: ti.join()'

    ppo = ppo.PPOTrainer(
            policies['explore_agent'],
            policy_lr=1e-5,
            value_lr=1e-3,
            target_kl_div=0.02,
            max_policy_train_iters=80,
            value_train_iters=80)
    
    #temp variables
    num_minibatches = 4
    num_envs = 10
    num_steps = 400
    num_iterations = 100
    learning_rate = 2.5e-4
    update_epochs = 4
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = None

    anneal_lr = False
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True

    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches

    #setup
    agent = policies['explore_agent']
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    for iteration in range(1, num_iterations + 1):
        if anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        #rollouts = generate_rollouts(policies, main_env, 'explore_agent', count=num_envs, max_steps=num_steps, verbose=True)
        rollouts = generate_rollouts_vec(policies, vec_env, 'explore_agent', count=num_envs, max_steps=num_steps, verbose=True)
        #print(f"Time Elapsed: {time.time() - start}")

        successes = sum([1 if rollout.success else 0 for rollout in rollouts['explore_agent']])
        total = len(rollouts['explore_agent'])
        print(f"Success Rate: {successes/total*100}%")

        if successes == 0:
            print("Repeating without training!")
            #continue
        
        train_rollouts = []
        for rollout in rollouts['explore_agent']:
            #if rollout.success: 
                train_rollouts.append(rollout)

        obs, actions, action_log_probs, values, advantages, returns = compose_rollouts(rollouts['explore_agent'])

        #unnecessary since already permutating
        iter_size = obs.shape[0]
        b_inds = np.arange(iter_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, iter_size, minibatch_size):
                end = start + minibatch_size
                if end > iter_size:
                    end = iter_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs[mb_inds], actions.long()[mb_inds])
                logratio = newlogprob - action_log_probs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                
                mb_advantages = advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                    v_clipped = values[mb_inds] + torch.clamp(
                        newvalue - values[mb_inds],
                        -clip_coef,
                        clip_coef
                    )
                    v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                # Train
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
            
            if target_kl is not None and approx_kl > target_kl:
                break
        
        print(f"value_loss: {v_loss.item()}")
        print(f"policy_loss: {pg_loss.item()}")
        print(f"entropy: {entropy_loss.item()}")
        print(f"old_approx_kl: {old_approx_kl.item()}")
        print(f"approx_kl: {approx_kl.item()}")

        #ppo.train_policy(obs, actions, action_log_probs, gaes)
        #ppo.train_value(obs, returns)

    #print(rollout1)