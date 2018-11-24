def print_stream(b):
    for i in b:
        print(bin(i)[2:].zfill(8))


def decode_b(bit_length, b):
    """
    decode bits in jpeg's compressed code of number
    :param bit_length: bits len
    :param b: bits
    :return: int
    """
    if bit_length > b.bit_length():
        bb = -2 ** bit_length + 1 + b
        return bb
    else:
        return b


def bytes_2_int(b, byteorder='big', signed=False):
    return int.from_bytes(b, byteorder=byteorder, signed=signed)


def save_bytes(b, idx_seg, fp='img.slt', format=False, int_len=0):
    with open(fp, 'w') as f:
        if isinstance(b, int):
            b = b.to_bytes(int_len, 'big')
            for i in b:
                if format:
                    f.writelines(
                        ''.join(['\t%-6s |' % idx_seg, hex(i)[2:].zfill(2), '\t', bin(i)[2:].zfill(8), '\t', '%d' % i]))
                else:
                    f.writelines(hex(i)[2:].zfill(2) + ' ')
                idx_seg += 1
        elif isinstance(b, bytes):
            for i in b:
                if format:
                    f.writelines(
                        ''.join(['\t%-6s |' % idx_seg, hex(i)[2:].zfill(2), '\t', bin(i)[2:].zfill(8), '\t', '%d' % i]))
                else:
                    f.writelines(hex(i)[2:].zfill(2) + ' ')
                idx_seg += 1
        else:
            raise Exception('Unsupported type:', type(b))
        return idx_seg


def print_bytes(b, idx_seg, int_len=0):
    if isinstance(b, int):
        b = b.to_bytes(int_len, 'big')
        for i in b:
            print('\t%-6s |' % idx_seg, hex(i)[2:].zfill(2), '\t', bin(i)[2:].zfill(8), '\t', i)
            idx_seg += 1
    elif isinstance(b, bytes):
        for i in b:
            print('\t%-6s |' % idx_seg, hex(i)[2:].zfill(2), '\t', bin(i)[2:].zfill(8), '\t', i)
            idx_seg += 1
    else:
        raise Exception('Unsupport type:', type(b))
    return idx_seg


def pop_bit(b, byteorder='big'):
    assert isinstance(b, bytes)
    for i in b:
        idxs = reversed(range(8)) if byteorder == 'big' else range(8)
        for bit_idx in idxs:
            yield (i >> bit_idx) & 0x01


def read_bits(cnt, popper):
    re = 0
    for i in range(cnt):
        re <<= 1
        re |= popper.__next__()
    return re


def read_byte(popper):
    return read_bits(8, popper)


def pop_byte(popper):
    while True:
        try:
            yield read_byte(popper)
        except:
            raise StopIteration('EOF')


def read_bytes(popper, n, return_int=True):
    """
    pop n bytes
    :param popper:
    :param return_int:
    :param n:
    :return:
    """
    if return_int:
        return read_bits(n * 8, popper)
    else:
        b = b''
        for i in range(n):
            b += read_byte(popper).to_bytes(1, 'big')
        return b
