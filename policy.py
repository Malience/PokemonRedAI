import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

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