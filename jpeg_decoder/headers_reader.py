import numpy as np

from jpeg_decoder.byte_N_bit_utils import print_bytes, read_bytes, bytes_2_int
from jpeg_decoder.zigzager import zigzag

flags_names = {
    0xd8: 'SOI 图像开始',
    0xe0: 'APP 0 JFIF应用数据块',
    0xe1: 'APP 1',
    0xe2: 'APP 2',
    0xe3: 'APP 3',
    0xe4: 'APP 4',
    0xe5: 'APP 5',
    0xe6: 'APP 6',
    0xe7: 'APP 7',
    0xe8: 'APP 8',
    0xe9: 'APP 9',
    0xea: 'APP10',
    0xeb: 'APP11',
    0xec: 'APP12',
    0xed: 'APP13',
    0xee: 'APP14',
    0xef: 'APP15',
    0xdb: 'DQT 量化表',
    0xc0: 'SOF0 帧开始',
    0xc4: 'DHT 霍夫曼(Huffman)表',
    0xda: 'SOS 扫描线开始',
    0xd9: 'EOI 图像结束',
    # TODO ????
    0xdd: 'DRI(Define Restart Interval)??'
}


def soi(flag, name, popper, print_details=False):
    """
    SOI
    :return:
    """
    return {'flag': flag, 'name': name}


def app_n(flag, name, popper, print_details=False):
    """
    APP N
    :return:
    """
    APP_N = {}
    length = read_bytes(popper, 2)
    content = read_bytes(popper, length - 2, False)
    if print_details:
        idx_seg = 1
        idx_seg = print_bytes(length, idx_seg, 2)
        print('\t\t\t\t\t\t\t\t', 'Length = {length}'.format(length=length))
        print_bytes(content, idx_seg)
    APP_N['flag'] = flag
    APP_N['name'] = name
    APP_N['length'] = length
    APP_N['content'] = content
    return APP_N


def app_0(flag, name, popper, print_details=False):
    """
    APP 0
    :return:
    """
    length = read_bytes(popper, 2)
    content = read_bytes(popper, length - 2, False)
    if print_details:
        idx_seg = 1
        idx_seg = print_bytes(length, idx_seg, 2)
        print('\t\t\t\t\t\t\t\t', 'Length = {length}'.format(length=length))
        idx_seg = print_bytes(content, idx_seg)

    identifier = bytes_2_int(content[:5])
    version = bytes_2_int(content[5:7])
    units = content[7]
    x_density = bytes_2_int(content[8:10])
    y_density = bytes_2_int(content[10:12])
    x_thumbnail = content[12]
    y_thumbnail = content[13]
    if print_details:
        print(
            '''    {length:#06X}\tLength\t{length}
        {identifier:#012X}\tIdentifier\t{identifier_b}
        {version:#06X}\tVersion
        {units:#04X}\tUnits\t零时表示未指定，为1表示英寸，为2表示厘米
        {x_density:#06X}\tx_density\t{x_density}
        {y_density:#06X}\ty_density\t{y_density}
        {x_thumbnail:#04X}\tx_thumbnail\t{x_thumbnail}
        {y_thumbnail:#04X}\ty_thumbnail\t{y_thumbnail}'''
                .format(
                length=length,
                identifier=identifier,
                identifier_b=identifier.to_bytes(5, 'big'),
                version=version,
                units=units,
                x_density=x_density,
                y_density=y_density,
                x_thumbnail=x_thumbnail,
                y_thumbnail=y_thumbnail,
            )
        )
    # remain = length - 16
    # if remain > 0:
    #     remain = read_bytes(remain, False)
    #     print(remain)
    # else:
    #     remain = b''
    remain = content[12:]

    return {
        'flag': flag,
        'name': name,
        'length': length,
        'identifier': identifier,
        'version': version,
        'units': units,
        'x_density': x_density,
        'y_density': y_density,
        'x_thumbnail': x_thumbnail,
        'y_thumbnail': y_thumbnail,
        'data': remain,
        'content': content
    }


def decode_dqt(dq_table, Pq, print_details=False):
    t = np.array(list(dq_table))
    if Pq == 1:
        raise Exception('Unsupported Pq')

    # t = t.reshape((8, 8))
    if print_details:
        print('Quantization Table:')
        print(zigzag(t))
    return t


def dqt(flag, name, popper, print_details=False):
    """
    DQT
    :return:
    """
    DQT = {}
    length = read_bytes(popper, 2)
    content = read_bytes(popper, length - 2, False)
    Pq = (content[0] & 0xf0) >> 4
    Tq = content[0] & 0x0f
    table = content[1:]
    if print_details:
        idx_seg = 1
        idx_seg = print_bytes(length, idx_seg, 2)
        print('\t\t\t\t\t\t\t\t', 'Length = {length}'.format(length=length))
        idx_seg = print_bytes(content[:1], idx_seg)
        print('\t\t\t\t\t\t\t\t', '(Pq,Tq)',
              '高四位Pq为量化表的数据精确度，Pq=0时，Q0~Qn的值为8位，Pq=1时，Qt的值为16位，Tq表示量化表的编号，为0~3。在基本系统中，Pq=0，Tq=0~1，也就是说最多有两个量化表。')
        print('\t\t\t\t\t\t\t\t', '(Pq,Tq)=(%d,%d)' % (Pq, Tq))
        print_bytes(table, idx_seg)
    table = decode_dqt(table, Pq, print_details)

    DQT['flag'] = flag
    DQT['name'] = name
    DQT['length'] = length
    DQT['content'] = content
    DQT['Pq'] = Pq
    DQT['Tq'] = Tq
    DQT['table'] = table
    return DQT

    # print(dq_table)


def sof_0(flag, name, popper, print_details=False):
    """
    SOF0
    :return:
    """
    SOF0 = {}
    length = read_bytes(popper, 2)
    content = read_bytes(popper, length - 2, False)
    if print_details:
        idx_seg = 1
        idx_seg = print_bytes(length, idx_seg, 2)
        print('\t\t\t\t\t\t\t\t', 'Length = {length}'.format(length=length))
        print_bytes(content, idx_seg)
    P = content[0]
    Y = bytes_2_int(content[1:3])
    X = bytes_2_int(content[3:5])
    Nf = content[5]
    Tqs = []
    for c in range(Nf):
        # 成分编号
        Cn = content[6 + c * 3]
        # 水平垂直采样因子
        HnVn = content[7 + c * 3]
        # 量化表编号
        Tqn = content[8 + c * 3]

        Tqs.append([Cn, HnVn, Tqn])

    SOF0['flag'] = flag
    SOF0['name'] = name
    SOF0['length'] = length
    SOF0['P'] = P
    SOF0['Y'] = Y
    SOF0['X'] = X
    SOF0['Nf'] = Nf
    SOF0['Tqs'] = Tqs
    if print_details:
        print('\tSOF0')
        for k in SOF0:
            print('\t\t', '%8s' % k, '\t', SOF0[k])
    SOF0['content'] = content
    return SOF0


def print_huf_codes(d, print_details=False):
    if print_details:
        print('\t\t', '%8s' % 'codes')
        print('\t\t\t\t\t',
              '{code_s:16s}    ({code:6s})    {value:4s}'.format(code_s='Code', code=' Code',
                                                                 value='Value'))
        for code in d:
            print('\t\t\t\t\t',
                  '{code_s:16s}    ({code:#06X})    {value:#04X}'.format(code_s=bin(code[0])[2:].zfill(code[1]),
                                                                         code=code[0], value=d[code]))


def dht(flag, name, popper, print_details=False):
    DHT = {}
    length = read_bytes(popper, 2)
    content = read_bytes(popper, length - 2, False)
    if print_details:
        idx_seg = 1
        idx_seg = print_bytes(length, idx_seg, 2)
        print('\t\t\t\t\t\t\t\t', 'Length = {length}'.format(length=length))
        print_bytes(content, idx_seg)
    TcTh = content[0]
    Tc = (TcTh & 0xf0) >> 4
    Th = TcTh & 0x0f

    cnts = dict(zip(range(1, 17), content[1:17]))

    vals = content[17:]
    vals_cur = 0
    codes = {}
    code_val = 0
    for k in cnts:
        # print(k, cnts[k])
        for cnt in range(cnts[k]):
            # print('\t', (bin(val), k))

            # # codes[code]={len,value}
            # codes[val] = {'len':k,'val':vals[vals_cur]}

            # # codes[(code,len)]=value
            codes[(code_val, k)] = vals[vals_cur]
            vals_cur += 1
            code_val += 1
        code_val <<= 1
        # print()

    DHT['flag'] = flag
    DHT['name'] = name
    DHT['length'] = length
    DHT['Tc'] = Tc
    DHT['Th'] = Th
    DHT['table'] = codes

    if print_details:
        for k in DHT:
            if k == 'table':
                print_huf_codes(DHT['table'], print_details)
            else:
                print('\t\t', '%8s' % k, '\t', DHT[k])




    DHT['content'] = content
    return DHT


def sos(flag, name, popper, print_details=False):
    SOS = {}
    length = read_bytes(popper, 2)
    content = read_bytes(popper, length - 2, False)
    if print_details:
        idx_seg = 1
        idx_seg = print_bytes(length, idx_seg, 2)
        print('\t\t\t\t\t\t\t\t', 'Length = {length}'.format(length=length))
        print_bytes(content, idx_seg)
    Ns = content[0]
    CsTdas = []
    for c in range(Ns):
        Csn = content[1 + c * 2]
        TdnTan = content[2 + c * 2]
        CsTdas.append([Csn, TdnTan])
    Ss = content[7]
    Se = content[8]
    AhAl = content[9]
    Ah = (AhAl & 0xf0) >> 4
    Al = AhAl & 0x0f

    SOS['flag'] = flag
    SOS['name'] = name
    SOS['length'] = length
    SOS['Ns'] = Ns
    SOS['CsTdas'] = CsTdas
    SOS['Ss'] = Ss
    SOS['Se'] = Se
    SOS['Ah'] = Ah
    SOS['Al'] = Al

    if print_details:
        for k in SOS:
            print('\t\t', '%8s' % k, '\t', SOS[k])

    SOS['content'] = content
    return SOS


flags_fun = {
    0xd8: soi,
    0xe0: app_0,
    0xe1: app_n,
    0xe2: app_n,
    0xe3: app_n,
    0xe4: app_n,
    0xe5: app_n,
    0xe6: app_n,
    0xe7: app_n,
    0xe8: app_n,
    0xe9: app_n,
    0xea: app_n,
    0xeb: app_n,
    0xec: app_n,
    0xed: app_n,
    0xee: app_n,
    0xef: app_n,
    0xdb: dqt,
    0xc0: sof_0,
    0xc4: dht,
    0xda: sos,
    0xd9: 'EOI 图像结束',
    # TODO ????
    0xdd: 'DRI(Define Restart Interval)??'
}
