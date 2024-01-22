import math

from multiagent import MultiAgent

from gymnasium import spaces

from ram_map import *

class ExploreLowAgent(MultiAgent):
    def __init__(self, name, action_space, target_n):
        super().__init__(name, action_space)
        self.target_n = target_n
    
    #TODO for parallelization we will just give it state as a parameter
    def reset(self, emulator, state):
        '''
        returns obs, info
        '''
        
        state['init_n'] = -1
        state['places'] = set()
        state['farthest'] = 0.0
        
        return {}
    
    def step(self, emulator, action, state):
        
        reward = 0
        term = False
        info = {}
        
        old_x, old_y, old_n = position(emulator.pyboy)
        if state['init_n'] < 0:
            state['init_n']  = old_n
        
        init_n = state['init_n']


        emulator.run(action)
        
        x, y, new_n = position(emulator.pyboy)
        
        if (x, y, new_n) not in state['places']:
            state['places'].add((x, y, new_n))
            # if new_n == init_n:
            reward += 0.01
            
        dist = (x - old_x)**2 + (y - old_y)**2
        
        # if dist > state['farthest']:
        #     reward += (dist - state['farthest']) * 0.2
        
        if new_n == self.target_n:
            reward += 5
            term = True
            info['success'] = True
            
        elif new_n != old_n:
            reward = -0.001
            #term = True
            #info['success'] = False
        
        #print(f'Target: {self.target_n}, Old: {old_n}, New: {new_n}, Rewards: {reward}')
        
        return [self.name], reward, term, False, info