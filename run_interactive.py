
import glob
from os.path import exists
from pathlib import Path
import uuid
from emulator import Emulator

from poke_mem_utility import *
    
if __name__ == '__main__':

    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')
    sess_path.mkdir(exist_ok=True)

    states_path = Path('states')
    states_path.mkdir(exist_ok=True)
    
    gb_path = './PokemonRed.gb'
    init_state = './has_pokedex_nballs.state'
    
    
    emu = Emulator(sess_path, gb_path, instance_id='main', headless=False)
    emu.reset(init_state)
    while True:
        emu.run(7)
