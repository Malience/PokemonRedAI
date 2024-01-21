import sys
import matplotlib.pyplot as plt
import glob
from os.path import exists
from pathlib import Path
import uuid


from pyboy import PyBoy
from pyboy.logger import log_level
from pyboy.utils import WindowEvent

from poke_mem_utility import *
from ram_map import *
    
if __name__ == '__main__':
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')
    sess_path.mkdir(exist_ok=True)
    
    emulation_speed = 2
    gb_path = './PokemonRed.gb'
    init_state = './has_pokedex_nballs.state'
    
    
    head = 'SDL2'
    log_level("ERROR")
    
    pyboy = PyBoy(
            gb_path,
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window='--quiet' in sys.argv,
        )

    screen = pyboy.botsupport_manager().screen()
    pyboy.set_emulation_speed(emulation_speed)
    
    print(screen.screen_ndarray().shape)
    
    with open(init_state, "rb") as f: pyboy.load_state(f)
    
    frame = 0
    dumpmem = False
    printind = 0#0xCC24 + 0x370
    printtarget = 0#0xC3A0
    while True:
        action = 7
        
        x, y, n = position(pyboy)
        #print(f'x: {x}, y: {y}, n: {n}')
        
        if dumpmem and frame%10 == 0:
            print('saved')
            dump_mem_range(env, 0xCC24, 0xED99, f'mem_dump_{frame}.txt')
        
        desc = ''''''
        
        #print_mem_range(env, 0xCC51, 0xCC52, desc)
        
        if printind > 0:
            dat = env.read_m(printind)
            print(f'{hex(printind)} - undefined: {dat}')
            
        if printtarget > 0:
            #print_mem_radius(env, printtarget, 5)
            print_mem_range(env, printtarget, printtarget + 40)
        
        if frame % 50 == 0:
            plt.imsave(sess_path / Path(f'curframe.jpeg'), screen.screen_ndarray()[::2,::2,0])
        
        pyboy.tick()
        frame+=1
