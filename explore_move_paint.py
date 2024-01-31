import math

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from multiagent import MultiAgent

from gymnasium import spaces

from ram_map import *

safety_offset = 10

class ExploreMovePaintAgent(MultiAgent):
    def __init__(self, name, action_space, target_n):
        super().__init__(name, action_space)
        self.target_n = target_n
    
    #TODO for parallelization we will just give it state as a parameter
    def reset(self, emulator, state):
        '''
        returns obs, info
        '''

        
        start_canvas = np.zeros((100+safety_offset*2, 100+safety_offset*2))
        x, y, n = position(emulator.pyboy)
        self.paint(start_canvas, x, y)

        state['start_n'] = n
        state['canvas'] = {n: start_canvas}
        state['step'] = 0
        
        return {}

    def paint(self, canvas, x, y, radius=5):
        added_val = 0.0
        sqradius = radius * radius

        x += safety_offset
        y += safety_offset

        for cur_x in range(x-radius, x+radius):
            if cur_x >= canvas.shape[0] or cur_x < 0:
                continue
            for cur_y in range(y-radius, y+radius):
                if cur_y >= canvas.shape[1] or cur_x < 0:
                    continue
                sqdist = (x - cur_x) ** 2 + (y - cur_y) ** 2
                if sqdist > sqradius:
                    continue
                
                cur_val = canvas[cur_x, cur_y]
                val = 1.0 - math.sqrt(sqdist) / radius

                # new_val = max(cur_val + val, 1.0)
                new_val = max(cur_val, val)

                added_val += new_val - cur_val

                canvas[cur_x, cur_y] = new_val

        return added_val

    def step(self, emulator, action, state):
        
        reward = 0
        term = False
        info = {}

        emulator.run(action)
        
        
        start_n = state['start_n']
        x, y, n = position(emulator.pyboy)

        if n not in state['canvas']:
            state['canvas'][n] = np.zeros((100, 100))
            self.paint(state['canvas'][n], x, y)

        canvas = state['canvas'][n]

        added_val = self.paint(canvas, x, y)
        if n != start_n:
            added_val *= 0.98

        reward = added_val * 0.002

        if n == self.target_n:
            reward += 5
            term = True
            info['success'] = True

        hpad = 5
        vpad = 4

        cropped_canvas = canvas[x+safety_offset-hpad:x+safety_offset+hpad , y+safety_offset-vpad:y+safety_offset+vpad]

        if state['step'] % 50 == 0:
            try:
                plt.imsave(emulator.s_path / Path(f'curframe_paint.jpeg'), state['canvas'][n])
                plt.imsave(emulator.s_path / Path(f'curframe_cropped.jpeg'), cropped_canvas)
            except:
                print("Windows hates you")
        state['step'] += 1
        
        return [self.name], reward, term, False, info