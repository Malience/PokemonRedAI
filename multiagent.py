# priority?

class MultiAgent():
    def __init__(self, name, action_space):
        self.name = name
        self.action_space = action_space
        
        
        
    def step(self, emulator, action):
        '''
        obs - list of other MultiAgents that cant be triggered from this action.
        reward, term, trun, info
        '''
        pass