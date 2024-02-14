import sys
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from pyboy import PyBoy
from pyboy.logger import log_level
from pyboy.utils import WindowEvent

from gymnasium import spaces

MOVEMENT_ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
]

SIMPLE_ACTIONS = MOVEMENT_ACTIONS + [
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
]

EXTENDED_ACTIONS = SIMPLE_ACTIONS + [
    WindowEvent.PRESS_BUTTON_START,
    WindowEvent.PASS
]

RELEASE_ARROW = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP
]

RELEASE_BUTTON = [
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B
]

MOVEMENT_ACTION_SPACE = spaces.Discrete(len(MOVEMENT_ACTIONS))
SIMPLE_ACTION_SPACE = spaces.Discrete(len(SIMPLE_ACTIONS))
EXTENDED_ACTION_SPACE = spaces.Discrete(len(EXTENDED_ACTIONS))

class Emulator():
    def __init__(self, session_path, gb_path,
        instance_id='default', headless=True,
        emulation_speed=None, act_freq=24):
        
        self.s_path = session_path
        self.s_path.mkdir(exist_ok=True)
        
        self.headless = headless
        self.act_freq = act_freq
        self.emulation_speed = 6 if emulation_speed is None else emulation_speed
        
        self.instance_id = instance_id
        
        ### PyBoy Setup ###
        head = 'headless' if headless else 'SDL2'

        log_level("ERROR")
        self.pyboy = PyBoy(
                gb_path,
                debugging=False,
                disable_input=False,
                window_type=head,
                hide_window='--quiet' in sys.argv,
            )

        self.screen = self.pyboy.botsupport_manager().screen()

        if not headless:
            self.pyboy.set_emulation_speed(self.emulation_speed)
    
    def reset(self, state=None):
        
        with open(state, "rb") as f:
            self.pyboy.load_state(f)
        
        self.frame_count = 0
        R, C = self.screen.raw_screen_buffer_dims()
        self.frames = np.zeros((3, R // 2, C //2))
    
    def close(self):
        self.pyboy.stop(save=False)
        
    def run(self, action):
        # press button then release after some steps
        self.pyboy.send_input(EXTENDED_ACTIONS[action])
        # disable rendering when we don't need it
        if self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(RELEASE_ARROW[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(RELEASE_BUTTON[action - 4])
                if EXTENDED_ACTIONS[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if i == self.act_freq-1:
                self.pyboy._rendering(True)
            self.pyboy.tick()
        
        ## TEMP
        if self.frame_count % 50 == 0:
            try:
                plt.imsave(
                    self.s_path / Path(f'curframe_{self.instance_id}.jpeg'), self.screen.screen_ndarray())
            except:
                print("Windows hates you")
        
        self.frames[2] = self.frames[1]
        self.frames[1] = self.frames[0]
        self.frames[0] = self.screen.screen_ndarray()[::2, ::2, 0]
        
        self.frame_count += 1
    
    def save_state(self, filepath):
        with open(Path(f'{filepath}'), "wb") as f:
            self.pyboy.save_state(f)

    def save_screenshot(self, filepath):
        plt.imsave(Path(f'{filepath}.jpeg'), self.screen.screen_ndarray())

    def get_frames(self):
        return self.frames
    '''
        print('', flush=True)
        fs_path = self.s_path / Path('states')
        fs_path.mkdir(exist_ok=True)
        obs = self.render(False)
        plt.imsave(fs_path / Path(f'{filename}_{self.reset_count}_{self.instance_id}.jpeg'), obs)
        with open(fs_path / Path(f'{filename}_{self.reset_count}_{self.instance_id}.state'), "wb") as f:
            self.pyboy.save_state(f)
    '''