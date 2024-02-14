import json
from emulator import Emulator
from map_table import map_table
from ram_parsing import position, money, read_string, parse_party, parse_inventory, parse_storage
from ram_map import PLAYER_NAME_ADDR_RANGE, PLAYER_PARTY_ADDR, BOX_ADDR
import uuid

def save_state(emulator: Emulator, folder='./states/', filename: str=None):
    x, y, n = position(emulator.pyboy)

    map_name = map_table[n]

    if filename is None:
        hashid = str(uuid.uuid4())[:8]
        filename = f'State_{map_name}_{hashid}'

    dct = {}
    
    playerInfo = {
        'Name': read_string(emulator.pyboy, PLAYER_NAME_ADDR_RANGE),
        'Money': money(emulator.pyboy),
        'Location': {'x': x, 'y': y, 'n': n, 'name': map_name}
    }

    party = parse_party(emulator.pyboy, PLAYER_PARTY_ADDR, 6)
    box = parse_party(emulator.pyboy, BOX_ADDR, 20)

    inventory = parse_inventory(emulator.pyboy)
    storage = parse_storage(emulator.pyboy)

    dct['PlayerInfo'] = playerInfo
    dct['Party'] = party
    dct['Box'] = box
    dct['Inventory'] = inventory
    dct['Storage'] = storage

    with open(f'{folder}/{filename}.json', 'w') as f:
        json.dump(dct, f)
    emulator.save_state(f'{folder}/{filename}.state')
    emulator.save_screenshot(f'{folder}/{filename}.jpeg')