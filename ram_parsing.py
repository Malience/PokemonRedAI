from ram_map import *
from text_table import text_table
from item_table import item_table

HP_ADDR =  [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
MAX_HP_ADDR = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
PARTY_ADDR = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
PARTY_LEVEL_ADDR = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
POKE_XP_ADDR = [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]
CAUGHT_POKE_ADDR = range(0xD2F7, 0xD309)
SEEN_POKE_ADDR = range(0xD30A, 0xD31D)
OPPONENT_LEVEL_ADDR = [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
X_POS_ADDR = 0xD362
Y_POS_ADDR = 0xD361
MAP_N_ADDR = 0xD35E
BADGE_1_ADDR = 0xD356
OAK_PARCEL_ADDR = 0xD74E
OAK_POKEDEX_ADDR = 0xD74B
OPPONENT_LEVEL = 0xCFF3
ENEMY_POKE_COUNT = 0xD89C
EVENT_FLAGS_START_ADDR = 0xD747
EVENT_FLAGS_END_ADDR = 0xD761
MUSEUM_TICKET_ADDR = 0xD754
MONEY_ADDR_1 = 0xD347
MONEY_ADDR_100 = 0xD348
MONEY_ADDR_10000 = 0xD349
BATTLE_TYPE_ADDR = 0xD057




def bcd(num):
    return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)

def bit_count(bits):
    return bin(bits).count('1')

def read_bit(game, addr, bit) -> bool:
    # add padding so zero will read '0b100000000' instead of '0b0'
    return bin(256 + game.get_memory_value(addr))[-bit-1] == '1'

def read_uint16(game, start_addr):
    '''Read 2 bytes'''
    val_256 = game.get_memory_value(start_addr)
    val_1 = game.get_memory_value(start_addr + 1)
    return 256*val_256 + val_1

def read_uint(game, addr_range, offset=0):
    val = 0
    for addr in addr_range:
        val = (val << 8) + game.get_memory_value(addr + offset)
    return val


def read_bytes(game, addr_range, offset=0):
    bytearray = []
    for addr in addr_range:
        bytearray.append(game.get_memory_value(addr + offset))
    return bytearray

def bytes_to_string(bytearray):
    s = ''
    for b in bytearray:
        if b == 80 or b == 0: # strings are null terminated
            return s
        s += text_table[b]
    return s

def read_string(game, addr_range, offset=0):
    return bytes_to_string(read_bytes(game, addr_range, offset))

def map_n(game):
    return game.get_memory_value(MAP_N_ADDR)

def position(game):
    r_pos = game.get_memory_value(Y_POS_ADDR)
    c_pos = game.get_memory_value(X_POS_ADDR)
    map_n = game.get_memory_value(MAP_N_ADDR)
    return r_pos, c_pos, map_n

def party(game):
    party = [game.get_memory_value(addr) for addr in PARTY_ADDR]
    party_size = game.get_memory_value(PARTY_SIZE_ADDR)
    party_levels = [game.get_memory_value(addr) for addr in PARTY_LEVEL_ADDR]
    return party, party_size, party_levels

def opponent(game):
    return [game.get_memory_value(addr) for addr in OPPONENT_LEVEL_ADDR]

def oak_parcel(game):
    return read_bit(game, OAK_PARCEL_ADDR, 1) 

def pokedex_obtained(game):
    return read_bit(game, OAK_POKEDEX_ADDR, 5)
 
def pokemon_seen(game):
    seen_bytes = [game.get_memory_value(addr) for addr in SEEN_POKE_ADDR]
    return sum([bit_count(b) for b in seen_bytes])

def pokemon_caught(game):
    caught_bytes = [game.get_memory_value(addr) for addr in CAUGHT_POKE_ADDR]
    return sum([bit_count(b) for b in caught_bytes])

def hp(game):
    '''Percentage of total party HP'''
    party_hp = [read_uint16(game, addr) for addr in HP_ADDR]
    party_max_hp = [read_uint16(game, addr) for addr in MAX_HP_ADDR]

    # Avoid division by zero if no pokemon
    sum_max_hp = sum(party_max_hp)
    if sum_max_hp == 0:
        return 1

    return sum(party_hp) / sum_max_hp

def money(game):
    return (100 * 100 * bcd(game.get_memory_value(MONEY_ADDR_1))
        + 100 * bcd(game.get_memory_value(MONEY_ADDR_100))
        + bcd(game.get_memory_value(MONEY_ADDR_10000)))

def badges(game):
    badges = game.get_memory_value(BADGE_1_ADDR)
    return bit_count(badges)

def events(game):
    '''Adds up all event flags, exclude museum ticket'''
    num_events = sum(bit_count(game.get_memory_value(i))
        for i in range(EVENT_FLAGS_START_ADDR, EVENT_FLAGS_END_ADDR))
    museum_ticket = int(read_bit(game, MUSEUM_TICKET_ADDR, 0))

    # Omit 13 events by default
    return max(num_events - 13 - museum_ticket, 0)


def battle_type(game):
    '''Type of Battle: 0 - No Battle   1 - Wild?   2 - Trainer   Others?'''
    return game.get_memory_value(BATTLE_TYPE_ADDR)


def parse_pokemon(game, addr):
    dct = {
        'Pokemon ID': game.get_memory_value(addr + POKEMON_ID_OFFSET),
        'Current HP': read_uint(game, POKEMON_CURRENT_HP_OFFSET_RANGE, addr),
        'Status': game.get_memory_value(addr + POKEMON_STATUS_OFFSET),

        'Type 1': game.get_memory_value(addr + POKEMON_TYPE1_OFFSET),
        'Type 2': game.get_memory_value(addr + POKEMON_TYPE2_OFFSET),

        'Catch Rate': game.get_memory_value(addr + POKEMON_CATCH_RATE_OFFSET),

        'Move 1': game.get_memory_value(addr + POKEMON_MOVE1_OFFSET),
        'Move 2': game.get_memory_value(addr + POKEMON_MOVE2_OFFSET),
        'Move 3': game.get_memory_value(addr + POKEMON_MOVE3_OFFSET),
        'Move 4': game.get_memory_value(addr + POKEMON_MOVE4_OFFSET),

        'Trainer ID': read_uint(game, POKEMON_TRAINER_ID_OFFSET_RANGE, addr),
        'Experience': read_uint(game, POKEMON_EXP_OFFSET_RANGE, addr),

        'HP EV': read_uint(game, POKEMON_HP_EV_OFFSET_RANGE, addr),
        'Attack EV': read_uint(game, POKEMON_ATTACK_EV_OFFSET_RANGE, addr),
        'Defense EV': read_uint(game, POKEMON_DEFENSE_EV_OFFSET_RANGE, addr),
        'Speed EV': read_uint(game, POKEMON_SPEED_EV_OFFSET_RANGE, addr),
        'Special EV': read_uint(game, POKEMON_SPECIAL_EV_OFFSET_RANGE, addr),

        'Attack/Defense IV': game.get_memory_value(addr + POKEMON_ATTACK_DEFENSE_IV_OFFSET),
        'Speed/Special IV': game.get_memory_value(addr + POKEMON_SPEED_SPECIAL_IV_OFFSET),

        'PP Move 1': game.get_memory_value(addr + POKEMON_PP_MOVE1_OFFSET),
        'PP Move 2': game.get_memory_value(addr + POKEMON_PP_MOVE2_OFFSET),
        'PP Move 3': game.get_memory_value(addr + POKEMON_PP_MOVE3_OFFSET),
        'PP Move 4': game.get_memory_value(addr + POKEMON_PP_MOVE4_OFFSET),

        'Level': game.get_memory_value(addr + POKEMON_LEVEL_OFFSET),

        'Max HP': read_uint(game, POKEMON_MAX_HP_OFFSET_RANGE, addr),
        'Attack': read_uint(game, POKEMON_ATTACK_OFFSET_RANGE, addr),
        'Defense': read_uint(game, POKEMON_DEFENSE_OFFSET_RANGE, addr),
        'Speed': read_uint(game, POKEMON_SPEED_OFFSET_RANGE, addr),
        'Special': read_uint(game, POKEMON_SPECIAL_OFFSET_RANGE, addr),
    }
    return dct

def parse_party(game, addr, count):
    dct = {
        'Count': game.get_memory_value(addr + PARTY_COUNT_OFFSET),
        # Skip indices
    }

    # 1st pokemon should be after the count (1 byte), the list of each pokemons species index (count bytes),
    # and a termination byte (total = count + 2)
    pokemon1_offset = count + 2

    ot_offset = pokemon1_offset + POKEMON_STRUCTURE_SIZE * count
    name_offset = ot_offset + PARTY_OT_NAME_SIZE * count

    # Load in each pokemon in the party
    for i in range(count):
        # Skip empty slots
        if game.get_memory_value(addr + pokemon1_offset + POKEMON_STRUCTURE_SIZE * i) == 0:
            continue

        # Trainer name and Pokemon name strings must be loaded seperately
        trainer_name = read_string(game, range(0, 10), addr + ot_offset + PARTY_OT_NAME_SIZE * i)
        pokemon_name = read_string(game, range(0, 10), addr + name_offset + PARTY_NICKNAME_SIZE * i)
        pokemon = parse_pokemon(game, addr + pokemon1_offset + POKEMON_STRUCTURE_SIZE * i)
        pokemon['Pokemon Name'] = pokemon_name
        pokemon['Trainer Name'] = trainer_name
        dct[f'Pokemon {i+1}'] = pokemon
    
    return dct



def parse_items(game, addr, count):
    dct = {
        'Total Items': game.get_memory_value(addr)
    }
    items = []
    for i in range(count):
        itemid = game.get_memory_value(addr + 0x1 + 2 * i)
        if itemid == 0:
            continue

        quantity = game.get_memory_value(addr + 0x2 + 2 * i)
        
        # Not sure if this is a glitch or what, but a TM55 shows up in the 3rd slot with 0 quantity
        if quantity == 0:
            continue

        item_name = item_table[itemid]

        items.append((item_name, quantity))

    dct['Items'] = items
    return dct

def parse_inventory(game):
    return parse_items(game, INVENTORY_ADDR, 20)

def parse_storage(game):
    return parse_items(game, STORAGE_ADDR, 50)


def parse_money(game):
    s = ''
    for b in MONEY_ADDR_RANGE:
        s += hex(game.get_memory_value(b))[2:]

    return int(s)