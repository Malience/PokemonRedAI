from multiagent import MultiAgent

from gymnasium import spaces

from ram_map import *

class BasicFleeAgent(MultiAgent):
    def step(self, game, action):
        reward = 0
        trun = False
        
        emulator.run(action)
        
        if battle_type == 0:
            reward = 1
            trun = True
            
        return [self.name], reward, False, trun, None