from multiagent import MultiAgent

from gymnasium import spaces

from ram_map import *

class BasicFleeAgent(MultiAgent):
    def reset(self, emulator, state):
        '''
        returns obs, info
        '''
        return {}

    def step(self, game, action, state):
        reward = 0
        term = False
        info = {}
        
        emulator.run(action)
        
        if battle_type == 0:
            reward = 1
            trun = True
            info = {'success': True}
            
        return [self.name], reward, term, False, info