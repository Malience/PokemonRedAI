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

from training import Trainer

# Anneal algorithm, I hate it
# if anneal_lr:
#             frac = 1.0 - (iteration - 1.0) / num_iterations
#             lrnow = frac * learning_rate
#             optimizer.param_groups[0]["lr"] = lrnow

class PPOSettings:
    def __init__(self, num_minibatches=4, learning_rate=2.5e-4, update_epochs=4, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=None, norm_adv=True, clip_coef=0.2, clip_vloss=True):
        self.num_minibatches = num_minibatches
        self.learning_rate = learning_rate
        self.update_epochs = update_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss    

class PPOTrainer(Trainer):
    def __init__(self, policy, settings: PPOSettings):
        
        self.policy = policy
        self.settings = settings

        self.optimizer = optim.Adam(policy.parameters(), lr=settings.learning_rate, eps=1e-5)
        
    def train(self, obs, actions, action_log_probs, values, advantages, returns, batch_size, verbose=False):
        
        batch_size = batch_size
        minibatch_size = batch_size // self.settings.num_minibatches
        
        #unnecessary since already permutating
        iter_size = obs.shape[0]
        b_inds = np.arange(iter_size)
        for epoch in range(self.settings.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, iter_size, minibatch_size):
                end = start + minibatch_size
                if end > iter_size:
                    end = iter_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(obs[mb_inds], actions.long()[mb_inds])
                logratio = newlogprob - action_log_probs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                
                mb_advantages = advantages[mb_inds]
                if self.settings.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.settings.clip_coef, 1 + self.settings.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.settings.clip_vloss:
                    v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                    v_clipped = values[mb_inds] + torch.clamp(
                        newvalue - values[mb_inds],
                        -self.settings.clip_coef,
                        self.settings.clip_coef
                    )
                    v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - self.settings.ent_coef * entropy_loss + v_loss * self.settings.vf_coef

                # Train
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.settings.max_grad_norm)
                self.optimizer.step()
            
            if self.settings.target_kl is not None and approx_kl > self.settings.target_kl:
                break
        
        if verbose:
            print(f"value_loss: {v_loss.item()}")
            print(f"policy_loss: {pg_loss.item()}")
            print(f"entropy: {entropy_loss.item()}")
            print(f"old_approx_kl: {old_approx_kl.item()}")
            print(f"approx_kl: {approx_kl.item()}")
