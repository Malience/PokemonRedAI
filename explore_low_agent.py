from multiagent import MultiAgent

from gymnasium import spaces

from ram_map import *

class ExploreLowAgent(MultiAgent):
    def __init__(self, name, action_space, target_n):
        super().__init__(name, action_space)
        self.target_n = target_n

    def step(self, emulator, action):
        
        reward = 0
        trun = False
        
        old_n = map_n(emulator.pyboy)
        
        emulator.run(action)
        
        new_n = map_n(emulator.pyboy)
        
        
        
        if new_n == self.target_n:
            reward = 5
            trun = True
        elif new_n != old_n:
            reward = -1
            trun = True
        
        #print(f'Target: {self.target_n}, Old: {old_n}, New: {new_n}, Rewards: {reward}')
        
        return [self.name], reward, False, trun, None