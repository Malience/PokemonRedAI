
import glob
from os.path import exists
from pathlib import Path
import uuid
from pokered_env import PokeRedEnv

from poke_mem_utility import *
    
if __name__ == '__main__':

    session_id = str(uuid.uuid4())[:8]
    sess_path = f'session_{session_id}'
    ep_length = 2**16 #2048 * 8
    
    states = glob.glob('./states_explore/*.state')
    
    env_config = {
                'headless': False, 'action_freq': 24, 'init_state': './has_pokedex_nballs.state',
                'max_steps': ep_length, 'session_path': Path(sess_path), 'gb_path': './PokemonRed.gb',
                'print_freq': 50, 'emulation_speed': 2, 'save_final_state': False, 'extra_buttons': True
            }
    
    num_cpu = 1 #64 #46  # Also sets the number of episodes per training iteration
    env = PokeRedEnv(env_config) #SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    #env_checker.check_env(env)
    file_name = 'models_map_discovery/poke_900277fe___48000_steps'
    #file_name = ''#session_09283997/poke_09283997___12000_steps'#models_explore/poke_b70e5751___6400_steps'
    #checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path, name_prefix=f'poke_{session_id}__')
    
    #model = load_model(file_name, env, ep_length, num_cpu)
    
    agent_enabled = True;
    
    #keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    frame = 0
    dumpmem = False
    printind = 0#0xCC24 + 0x370
    printtarget = 0xC3A0
    while True:
        action = 7 # pass action
        #screen = obs['frame_0']
        #if agent_enabled: action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step({env.player0: action})
        
        #x, y, n = env.pos()
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
        
        env.render()
        frame+=1
