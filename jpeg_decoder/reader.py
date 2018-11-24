import cv2
import numpy as np

from jpeg_decoder.byte_N_bit_utils import bytes_2_int, print_stream, decode_b, print_bytes
from jpeg_decoder.paster import merge_mcu, merge_channels, merge_mcus
from jpeg_decoder.plter import plt_mcu, plt_3_arrays
from jpeg_decoder.zigzager import zigzag
from jpeg_decoder.expander import expand

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


def soi(flag, name):
    """
    SOI
    :return:
    """
    return {'flag': flag, 'name': name}


def app_n(flag, name):
    """
    APP N
    :return:
    """
    APP_N = {}
    length = read_bytes(2)
    content = read_bytes(length - 2, False)
    idx_seg = 1
    idx_seg = print_bytes(length, idx_seg, 2)
    print('\t\t\t\t\t\t\t\t', 'Length = {length}'.format(length=length))
    print_bytes(content, idx_seg)
    APP_N['flag'] = flag
    APP_N['name'] = name
    APP_N['length'] = length
    APP_N['content'] = content
    return APP_N


def app_0(flag, name):
    """
    APP 0
    :return:
    """
    length = read_bytes(2)
    content = read_bytes(length - 2, False)
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


def decode_dqt(dq_table, Pq):
    t = np.array(list(dq_table))
    if Pq == 1:
        raise Exception('Unsupported Pq')

    # t = t.reshape((8, 8))
    print('Quantization Table:')
    print(zigzag(t))
    return t


def dqt(flag, name):
    """
    DQT
    :return:
    """
    DQT = {}
    length = read_bytes(2)
    content = read_bytes(length - 2, False)
    idx_seg = 1
    idx_seg = print_bytes(length, idx_seg, 2)
    print('\t\t\t\t\t\t\t\t', 'Length = {length}'.format(length=length))
    idx_seg = print_bytes(content[:1], idx_seg)
    print('\t\t\t\t\t\t\t\t', '(Pq,Tq)',
          '高四位Pq为量化表的数据精确度，Pq=0时，Q0~Qn的值为8位，Pq=1时，Qt的值为16位，Tq表示量化表的编号，为0~3。在基本系统中，Pq=0，Tq=0~1，也就是说最多有两个量化表。')
    Pq = (content[0] & 0xf0) >> 4
    Tq = content[0] & 0x0f
    print('\t\t\t\t\t\t\t\t', '(Pq,Tq)=(%d,%d)' % (Pq, Tq))

    table = content[1:]
    print_bytes(table, idx_seg)
    table = decode_dqt(table, Pq)

    DQT['flag'] = flag
    DQT['name'] = name
    DQT['length'] = length
    DQT['content'] = content
    DQT['Pq'] = Pq
    DQT['Tq'] = Tq
    DQT['table'] = table
    return DQT

    # print(dq_table)



def sof_0(flag, name):
    """
    SOF0
    :return:
    """
    SOF0 = {}
    length = read_bytes(2)
    content = read_bytes(length - 2, False)
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
    print('\tSOF0')
    for k in SOF0:
        print('\t\t', '%8s' % k, '\t', SOF0[k])
    SOF0['content'] = content
    return SOF0


def print_huf_codes(d):
    print('\t\t', '%8s' % 'codes')
    print('\t\t\t\t\t',
          '{code_s:16s}    ({code:6s})    {value:4s}'.format(code_s='Code', code=' Code',
                                                             value='Value'))
    for code in d:
        print('\t\t\t\t\t',
              '{code_s:16s}    ({code:#06X})    {value:#04X}'.format(code_s=bin(code[0])[2:].zfill(code[1]),
                                                                     code=code[0], value=d[code]))


def dht(flag, name):
    DHT = {}
    length = read_bytes(2)
    content = read_bytes(length - 2, False)
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

    for k in DHT:
        if k == 'codes':
            print_huf_codes(DHT['codes'])
        else:
            print('\t\t', '%8s' % k, '\t', DHT[k])

    DHT['content'] = content
    return DHT


def sos(flag, name):
    SOS = {}
    length = read_bytes(2)
    content = read_bytes(length - 2, False)
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


def read_bytes(n, return_int=True):
    """
    read n bytes and move cur globally
    :param n:
    :return:
    """
    global cur
    cur += n
    return ord(img[cur - n:cur]) if n <= 1 else int.from_bytes(img[cur - n:cur], byteorder='big',
                                                               signed=False) if return_int else img[cur - n:cur]


headers = {}


def print_dht(dht):
    print(dht['name'])
    print('TcTh', bin(dht['Tc'])[2:].zfill(4) + bin(dht['Th'])[2:].zfill(4))
    ks = list(dht['table'].keys())
    ks.sort()
    for k in ks:
        print('%-5d' % k[0], '%20s' % bin(k[0])[2:].zfill(k[1]), '{v:#04X}'.format(v=dht['table'][k]), dht['table'][k])


colors_YCrCb = ['Y', 'Cr', 'Cb']
colors_RGB = ['R', 'G', 'B']


def concatenate(a, b, c):
    a = a.reshape((a.shape[0], a.shape[1], -1))
    b = b.reshape((b.shape[0], b.shape[1], -1))
    c = c.reshape((c.shape[0], c.shape[1], -1))
    return np.concatenate((a, b, c), 2)


# TODO ycrcb2rgb
def YCrCb2RGB(mcu, colors_tqs):
    # print(mcu)

    if colors_tqs[1]['DUC'] > 1 or colors_tqs[2]['DUC'] > 1:
        raise Exception('Unsupported operation when sampling of Cr or Cb >1(=4)')

    _mcu = []
    YCrCbChannels = [[], [], []]
    RGBChannels = [[], [], []]

    data_unit_idx = 0
    for i in range(colors_tqs[0]['V']):
        for j in range(colors_tqs[0]['H']):
            data_unit_Y = mcu[0][data_unit_idx]
            data_unit_Cr = expand(mcu[1][0], (i, j))
            data_unit_Cb = expand(mcu[2][0], (i, j))

            YCrCbChannels[0].append(data_unit_Y)
            YCrCbChannels[1].append(data_unit_Cr)
            YCrCbChannels[2].append(data_unit_Cb)

            # # todo convert YCrCb to RGB
            # # wheel:
            #
            # data_unit = concatenate(data_unit_Y,data_unit_Cr,data_unit_Cb)
            # data_unit += 128
            # data_unit=data_unit.astype('uint8')
            # print(data_unit)
            # data_unit = cv2.cvtColor(data_unit, cv2.COLOR_YCrCb2BGR)

            # self
            # data_unit_Y+=128
            # data_unit_Cr+=128
            # data_unit_Cb+=128
            #
            # data_unit_R = data_unit_Y + data_unit_Cb
            # data_unit_G = data_unit_Y - 0.34414 * data_unit_Cb - 0.71414 * data_unit_Cr
            # data_unit_B = data_unit_Y + data_unit_Cr

            # data_unit_R = data_unit_Y + 1.402 * data_unit_Cb + 128
            # data_unit_G = data_unit_Y - 0.34414 * data_unit_Cr - 0.71414 * data_unit_Cb + 128
            # data_unit_B = data_unit_Y + 1.772 * data_unit_Cr + 128

            # todo ???? why it works
            data_unit_R = data_unit_Y + 1.402 * data_unit_Cb + 128
            data_unit_B = data_unit_Y - 0.34414 * data_unit_Cr - 0.71414 * data_unit_Cb + 128
            data_unit_G = data_unit_Y + 1.772 * data_unit_Cr + 128

            data_unit_R[data_unit_R > 255] = 255
            data_unit_G[data_unit_G > 255] = 255
            data_unit_B[data_unit_B > 255] = 255
            data_unit_R[data_unit_R < 0] = 0
            data_unit_G[data_unit_G < 0] = 0
            data_unit_B[data_unit_B < 0] = 0

            data_unit_R = data_unit_R.astype('uint8')
            data_unit_G = data_unit_G.astype('uint8')
            data_unit_B = data_unit_B.astype('uint8')

            RGBChannels[0].append(data_unit_R)
            RGBChannels[1].append(data_unit_G)
            RGBChannels[2].append(data_unit_B)

            data_unit = concatenate(data_unit_R, data_unit_B, data_unit_G)

            _mcu.append(data_unit)
            data_unit_idx += 1

    _mcu = merge_mcu(_mcu)
    RGBChannels = merge_channels(RGBChannels)
    YCrCbChannels = merge_channels(YCrCbChannels)

    # plt_mcu(_mcu,YCrCbChannels,cmap='binary_r')
    # plt_mcu(_mcu,RGBChannels,channels_name=colors_RGB,cmap='binary_r')
    # plt_mcu(RGBChannels,channels_name=colors_RGB,cmap='binary_r')
    # plt_mcu(_mcu)
    return _mcu


def convert_mcu(mcu, colors_tqs):
    QTs = {T['Tq']: T for T in headers[0xdb]}
    # print(mcu)
    for color in mcu:
        for idx, data_unit in enumerate(mcu[color]):
            data_unit = np.array(data_unit)
            data_unit = np.multiply(data_unit, QTs[colors_tqs[color]['QTable']]['table'])
            data_unit = zigzag(data_unit)
            data_unit = data_unit.astype('float')
            data_unit = cv2.idct(data_unit)
            data_unit = data_unit.astype('int')
            # print(colors_YCrCb[color], idx)
            # print(data_unit)
            mcu[color][idx] = data_unit
    # todo find wheels
    mcu = YCrCb2RGB(mcu, colors_tqs)
    return mcu


def crop_image(img_show, sof0):
    row_cnt = sof0['Y']
    col_cnt = sof0['X']
    return img_show[:row_cnt,:col_cnt]

def read_scan(SCAN=True, show_img=True, print_data = False, print_decode = False):
    global cur
    _img = img[cur:]
    cur = len(img)

    # print('data old:')
    # print_bytes(_img, 1)
    # save_bytes(_img, 1)

    # replace 0xff00 and 0xffd9
    _cur = 0
    scan = b''
    while _cur < len(_img):
        i = _img[_cur]
        if i == 0xff:
            i_next = _img[_cur + 1]
            if i_next == 0xd9:
                print(hex((i << 8) | i_next))
                break
            elif i_next == 0x00:
                # print('Warn! 0xFF00 @cur', _cur + 1)
                scan += i.to_bytes(1, 'big')
                _cur += 2
                continue
            elif i_next == 0xff:
                print('Warn! 0xFFFF @cur', _cur + 1)
                _cur += 1
                continue
            elif i_next in range(0xd0, 0xd8):
                # TODO 0XD0--0XD7，组成RSTn标记，需要忽视整个RSTn标记，即不对当前0XFF和紧接着的0XDn两个字节进行译码，并按RST标记的规则调整译码变量；???
                scan += i.to_bytes(1, 'big')
                raise Exception('Unimpled')

        else:
            scan += i.to_bytes(1, 'big')

        _cur += 1

    # save_bytes(scan, 1)

    Y_DC_Huffman_table_id = (headers[0xda][0]['CsTdas'][0][1] >> 4) | 0x00
    Y_AC_Huffman_table_id = (headers[0xda][0]['CsTdas'][0][1] & 0x0f) | 0x10
    U_DC_Huffman_table_id = (headers[0xda][0]['CsTdas'][1][1] >> 4) | 0x00
    U_AC_Huffman_table_id = (headers[0xda][0]['CsTdas'][1][1] & 0x0f) | 0x10
    V_DC_Huffman_table_id = (headers[0xda][0]['CsTdas'][2][1] >> 4) | 0x00
    V_AC_Huffman_table_id = (headers[0xda][0]['CsTdas'][2][1] & 0x0f) | 0x10
    table_ids = [
        (Y_DC_Huffman_table_id, Y_AC_Huffman_table_id),
        (U_DC_Huffman_table_id, U_AC_Huffman_table_id),
        (V_DC_Huffman_table_id, V_AC_Huffman_table_id)
    ]
    DHTs = {}
    for _dht in headers[0xc4]:
        DHTs[(_dht['Tc'] << 4) | _dht['Th']] = _dht  # ['table']
        if print_decode:
            print_dht(_dht)
    if print_data:
        print('data:')
        print_bytes(scan, 1)
    if not SCAN:
        return
    if print_data:
        print_stream(scan)

    # get H123,V123
    sof0_Tqs = headers[0xc0][0]['Tqs']
    colors_tqs = {}
    for tq in sof0_Tqs:
        # tq[0]-颜色分量号，-1映射到012
        colors_tqs[tq[0] - 1] = {}
        # 水平和垂直采样因子
        colors_tqs[tq[0] - 1]['H'] = tq[1] >> 4
        colors_tqs[tq[0] - 1]['V'] = tq[1] & 0x0f
        # MCU 中 data unit(s) 的数量
        colors_tqs[tq[0] - 1]['DUC'] = colors_tqs[tq[0] - 1]['H'] * colors_tqs[tq[0] - 1]['V']
        # 量化表编号
        colors_tqs[tq[0] - 1]['QTable'] = tq[2]

    prev = 0
    DC = True
    got = True
    RLE_val = False
    code_len = 0

    # todo convert to numpy
    MCUs = []
    mcu = {0: [], 1: [], 2: []}
    data_unit = []

    in_mcu = True
    mcu_idx = 0
    current_color = 0
    data_unit_idx = 0

    # 差分编码
    diff = [0, 0, 0]

    for i in scan:
        ii = i | (prev << 8)
        for a in reversed(range(8)):
            code_len += 1
            t = (ii >> a)
            # print(t, bin(t))
            if RLE_val:
                if code_len >= val_len:
                    ii &= (0xff >> (8 - a))
                    # hit the val
                    RLE_val = decode_b(code_len, t)
                    if print_decode:
                        print('RLE', bin(t)[2:].zfill(code_len), RLE_val)
                    data_unit.append(RLE_val)
                    code_len = 0
                    RLE_val = False
                    if len(data_unit) >= 64:
                        DC = True
                        data_unit_idx += 1
                        # # # todo 差分编码矫正
                        # # Dc_n = Diff_n-1 + Diff_n
                        # tmp = data_unit[0]
                        # data_unit[0] += diff[current_color]
                        # diff[current_color] = tmp

                        # Dc_n = Dc_n-1 + Diff_n
                        data_unit[0] += diff[current_color]
                        diff[current_color] = data_unit[0]

                        mcu[current_color].append(data_unit)
                        data_unit = []
                        if data_unit_idx >= colors_tqs[current_color]['DUC']:
                            if print_decode:
                                print('---Color', colors_YCrCb[current_color], 'end')
                            current_color += 1
                            current_color %= 3
                            data_unit_idx = 0
                            if current_color == 0:
                                # todo big image - do add to img sync
                                mcu = convert_mcu(mcu, colors_tqs)
                                MCUs.append(mcu)
                                if print_decode:
                                    print('===MCU', mcu_idx, ' end')
                                mcu = {0: [], 1: [], 2: []}
                                mcu_idx += 1
                                in_mcu = True
                continue
            if got:
                got = False
                if DC:
                    table = DHTs[table_ids[current_color][0]]
                else:
                    table = DHTs[table_ids[current_color][1] | 0x10]
            # if t in table['table'] and code_len == table['table'][t]['len']:
            if (t, code_len) in table['table']:
                # hit the huffman code
                got = True
                # val = table['table'][(t, code_len)]['val']
                val = table['table'][(t, code_len)]
                if DC:
                    if in_mcu and current_color == 0:
                        if print_decode:
                            print('===MCU', mcu_idx, '===')
                        in_mcu = False
                    if print_decode:
                        print('---Col', colors_YCrCb[current_color], '---data unit', data_unit_idx, '---')
                        print('DC:', end=' ')
                        print('hcode', bin(t)[2:].zfill(code_len), 'val', '{v:#04X}, {v}'.format(v=val))
                    if val == 0:
                        # DC = True
                        # print('Color', colors[current_color], 'end')
                        # current_color += 1
                        # current_color %= 3

                        DC = False
                        val_len = val
                        RLE_val = False
                        # todo dc=0 : append 0
                        data_unit.append(RLE_val)
                    else:
                        DC = False
                        val_len = val
                        RLE_val = True
                else:
                    # AC
                    if print_decode:
                        print('AC:', end=' ')
                        print('hcode', bin(t)[2:].zfill(code_len), 'val',  '{v:#04X}, {v}'.format(v=val))
                    if val == 0:
                        DC = True
                        data_unit_idx += 1
                        rest_z = 64 - len(data_unit)
                        if rest_z > 0:
                            data_unit += [0] * rest_z

                        # # todo 差分编码矫正
                        # tmp = data_unit[0]
                        # data_unit[0] += diff[current_color]
                        # diff[current_color] = tmp

                        data_unit[0] += diff[current_color]
                        diff[current_color] = data_unit[0]

                        mcu[current_color].append(data_unit)
                        data_unit = []
                        if data_unit_idx >= colors_tqs[current_color]['DUC']:
                            if print_decode:
                                print('---Color', colors_YCrCb[current_color], 'end')
                            current_color += 1
                            current_color %= 3
                            data_unit_idx = 0
                            if current_color == 0:
                                # todo big image - do add to img sync
                                mcu = convert_mcu(mcu, colors_tqs)
                                MCUs.append(mcu)
                                if print_decode:
                                    print('===MCU', mcu_idx, ' end')
                                mcu = {0: [], 1: [], 2: []}
                                mcu_idx += 1
                                in_mcu = True
                    else:
                        zero_cnt = val >> 4
                        if print_decode:
                            print('\tzero cnt:', zero_cnt)
                        data_unit += ([0] * zero_cnt)
                        val_len = val & 0x0f
                        # todo 应该是0xF0才会到这里，假设16个0后面一定会有AC编码
                        if val_len == 0:
                            RLE_val = False
                            # todo debug AC code == 0xF0 ? zero_cnt=15,+0?
                            data_unit.append(0)

                        else:
                            RLE_val = True

                # print('hcode',bin(t)[2:].zfill(table['table'][t]['len']))
                # print('hcode', bin(t)[2:].zfill(table['table'][(t, code_len)]),'val','{v:#04X}, {v}'.format(v=val))
                ii &= (0xff >> (8 - a))
                code_len = 0
        prev = ii

    # print('===================\nMCUs')
    # print(MCUs)
    img_show = merge_mcus(MCUs, colors_tqs, headers[0xc0][0])
    img_show = crop_image(img_show, headers[0xc0][0])
    if show_img:
        plt_mcu(img_show)
    return img_show

FILE='lena.jpg'
SCAN = True
# todo shrot cut 4 file
with open(FILE, 'rb') as img_file:
    img = img_file.read()
    assert isinstance(img, bytes)
    cur = 0
    while cur < len(img):
        b = read_bytes(1)
        if b == 0xff:
            if img[cur] in flags_names:
                _flag = read_bytes(1)
                flag = 0xff00 | _flag
                print(hex(flag).upper(), '\t', flags_names[_flag])

                if callable(flags_fun[_flag]):
                    if _flag not in headers:
                        headers[_flag] = []
                    headers[_flag].append(flags_fun[_flag](_flag, flags_names[_flag]))
                else:
                    print(flags_names[_flag])

                if _flag == 0xda:
                    img_array = read_scan(SCAN)
                    break

                continue

print(img_array)
img_cv = cv2.imread(FILE)
img_cv = cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB)
delta = np.abs(img_array.astype('int') - img_cv.astype('int')).astype('uint8')
plt_3_arrays(img_array, img_cv, delta)
print(img_cv)
print(delta)
print(np.sum(delta))
print('-' * 20)
# print(headers)
