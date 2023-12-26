
mem_map = {
    0xCFC4: 'Text/Menu Open (0 - 1)'
}

def print_mem_radius(env, target, radius):
    print_mem_range(env, target - radius, target + radius, target)

def strdex(dex):
    hexdex = hex(dex)
    hexdex = str(hexdex)
    hexdex = hexdex[:2] + hexdex[2:].upper()
    return hexdex

def print_mem_range(env, start, end, target=-1):
    print('~~~~~~~~~~')
    for i in range(start, end + 1):
        d = strdex(i)
        d = f'{d} - {mem_map[i]}' if i in mem_map else f'{d} - undefined'
        if i == target: d += ' - TGT'
        mem = env.read_m(i)
        print(f'{d}: {mem}')
    print('~~~~~~~~~~')
    #TODO: ranges

def dump_mem_range(env, start, end, file):
    data = ''
    with open(file, 'wb') as f:
        for i in range(start, end):
            mem = env.read_m(i).to_bytes(1, 'little')
            f.write(mem)