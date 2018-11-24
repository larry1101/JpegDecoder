import numpy as np

import cv2

from jpeg_decoder.byte_N_bit_utils import decode_b, pop_byte
from jpeg_decoder.expander import expand
from jpeg_decoder.paster import merge_mcus, merge_mcu
from jpeg_decoder.zigzager import zigzag

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
            # wheel:

            # todo find Y's error in +=128merg
            data_unit_Y = data_unit_Y.astype('int')
            data_unit_Y += 128
            data_unit_Y[data_unit_Y >= 256] = 255
            data_unit_Cr = data_unit_Cr.astype('int')
            data_unit_Cr += 128
            data_unit_Cb = data_unit_Cb.astype('int')
            data_unit_Cb += 128
            data_unit = concatenate(data_unit_Y, data_unit_Cr, data_unit_Cb)
            # data_unit = data_unit.astype('int')
            # data_unit += 128
            data_unit = data_unit.astype('uint8')
            # print(data_unit)
            data_unit = cv2.cvtColor(data_unit, cv2.COLOR_YCrCb2BGR)

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

            # # todo ???? why it works
            # data_unit_R = data_unit_Y + 1.402 * data_unit_Cb + 128
            # data_unit_B = data_unit_Y - 0.34414 * data_unit_Cr - 0.71414 * data_unit_Cb + 128
            # data_unit_G = data_unit_Y + 1.772 * data_unit_Cr + 128
            #
            # data_unit_R[data_unit_R > 255] = 255
            # data_unit_G[data_unit_G > 255] = 255
            # data_unit_B[data_unit_B > 255] = 255
            # data_unit_R[data_unit_R < 0] = 0
            # data_unit_G[data_unit_G < 0] = 0
            # data_unit_B[data_unit_B < 0] = 0
            #
            # data_unit_R = data_unit_R.astype('uint8')
            # data_unit_G = data_unit_G.astype('uint8')
            # data_unit_B = data_unit_B.astype('uint8')
            #
            # RGBChannels[0].append(data_unit_R)
            # RGBChannels[1].append(data_unit_G)
            # RGBChannels[2].append(data_unit_B)
            #
            # data_unit = concatenate(data_unit_R, data_unit_B, data_unit_G)

            _mcu.append(data_unit)
            data_unit_idx += 1

    _mcu = merge_mcu(_mcu)
    # RGBChannels = merge_channels(RGBChannels)
    # YCrCbChannels = merge_channels(YCrCbChannels)

    # plt_mcu(_mcu,YCrCbChannels,cmap='binary_r')
    # plt_mcu(_mcu,RGBChannels,channels_name=colors_RGB,cmap='binary_r')
    # plt_mcu(RGBChannels,channels_name=colors_RGB,cmap='binary_r')
    # plt_mcu(_mcu)
    return _mcu


def convert_mcu(mcu, colors_tqs, headers):
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
    return img_show[:row_cnt, :col_cnt]


def convert_RLE_codes(RLE_codes):
    # DC
    data_unit = [RLE_codes[0][1]]
    for RLE in RLE_codes[1:]:
        if RLE == 0:
            rest_z = 64 - len(data_unit)
            if rest_z > 0:
                data_unit += [0] * rest_z
        else:
            data_unit += ([0] * RLE[0])
            data_unit.append(RLE[1])
    return data_unit


def print_dht(dht):
    print(dht['name'])
    print('TcTh', bin(dht['Tc'])[2:].zfill(4) + bin(dht['Th'])[2:].zfill(4))
    ks = list(dht['table'].keys())
    ks.sort()
    for k in ks:
        print('%-5d' % k[0], '%20s' % bin(k[0])[2:].zfill(k[1]), '{v:#04X}'.format(v=dht['table'][k]),
              dht['table'][k])


def read_scan(popper, headers, SCAN=True):
    # print('data old:')
    # print_bytes(_img, 1)
    # save_bytes(_img, 1)

    # replace 0xff00 and 0xffd9
    # _cur = 0
    byte_popper = pop_byte(popper)
    scan = b''
    for i in byte_popper:
        if i == 0xff:
            i_next = byte_popper.__next__()
            if i_next == 0xd9:
                print(hex((i << 8) | i_next).upper(), 'Scan end!')
                print()
                break
            elif i_next == 0x00:
                # print('Warn! 0xFF00 @cur', _cur + 1)
                scan += i.to_bytes(1, 'big')
                continue
            elif i_next == 0xff:
                print('Warn! 0xFFFF')
                continue
            elif i_next in range(0xd0, 0xd8):
                # TODO 0XD0--0XD7，组成RSTn标记，需要忽视整个RSTn标记，即不对当前0XFF和紧接着的0XDn两个字节进行译码，并按RST标记的规则调整译码变量；???
                scan += i.to_bytes(1, 'big')
                raise Exception('Unimpled')

        else:
            scan += i.to_bytes(1, 'big')

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
        print_dht(_dht)
    # print('data:')
    # print_bytes(scan, 1)
    if not SCAN:
        return
    # print_stream(scan)

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
    RLE_codes = []
    data_unit_len = 0
    RLE_val = False
    code_len = 0
    val_len = 0

    # todo convert to numpy
    MCUs = []
    mcu = {0: [], 1: [], 2: []}
    data_unit = []

    # in_mcu = True
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
                    RLE_value = decode_b(code_len, t)
                    RLE_codes[-1][1] = RLE_value
                    # # print('RLE', bin(t)[2:].zfill(code_len), RLE_value)
                    # data_unit.append(RLE_value)
                    code_len = 0
                    RLE_val = False
                    # if len(data_unit) >= 64:
                    if data_unit_len >= 64:
                        # todo decode RLE codes
                        # get data unit from RLE codes and reset RLE_codes, data_unit_len
                        data_unit = convert_RLE_codes(RLE_codes)
                        RLE_codes = []
                        data_unit_len = 0

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
                        # # # todo 差分编码矫正end

                        mcu[current_color].append(data_unit)
                        data_unit = []
                        if data_unit_idx >= colors_tqs[current_color]['DUC']:
                            # print('---Color', colors_YCrCb[current_color], 'end')
                            current_color += 1
                            current_color %= 3
                            data_unit_idx = 0
                            if current_color == 0:
                                # todo big image - do add to img sync
                                mcu = convert_mcu(mcu, colors_tqs, headers)
                                MCUs.append(mcu)
                                # print('===MCU', mcu_idx, ' end')
                                mcu = {0: [], 1: [], 2: []}
                                mcu_idx += 1
                                # in_mcu = True
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
                    # if in_mcu and current_color == 0:
                    #     # print('===MCU', mcu_idx, '===')
                    #     in_mcu = False
                    # print('---Col', colors_YCrCb[current_color], '---data unit', data_unit_idx, '---')
                    # print('DC:', end=' ')
                    # print('hcode', bin(t)[2:].zfill(code_len), 'val',
                    #       '{v:#04X}, {v}'.format(v=val))
                    if val == 0:
                        # DC = True
                        # print('Color', colors[current_color], 'end')
                        # current_color += 1
                        # current_color %= 3

                        DC = False
                        val_len = val
                        RLE_val = False
                        # todo dc=0 : append 0
                        # data_unit.append(0)
                        RLE_codes.append([0, 0])
                        data_unit_len += 1
                    else:
                        DC = False
                        val_len = val
                        RLE_val = True
                        RLE_codes.append([0, 0])
                        data_unit_len += 1
                else:
                    # AC
                    # print('AC:', end=' ')
                    # print('hcode', bin(t)[2:].zfill(code_len), 'val',
                    #       '{v:#04X}, {v}'.format(v=val))
                    if val == 0:
                        DC = True
                        data_unit_idx += 1
                        # rest_z = 64 - len(data_unit)
                        # if rest_z > 0:
                        #     data_unit += [0] * rest_z

                        RLE_codes.append(0)
                        # todo decode RLE codes or merge them
                        # get data unit from RLE codes and reset RLE_codes, data_unit_len
                        data_unit = convert_RLE_codes(RLE_codes)
                        RLE_codes = []
                        data_unit_len = 0

                        # # todo 差分编码矫正
                        # tmp = data_unit[0]
                        # data_unit[0] += diff[current_color]
                        # diff[current_color] = tmp

                        data_unit[0] += diff[current_color]
                        diff[current_color] = data_unit[0]

                        mcu[current_color].append(data_unit)
                        data_unit = []
                        if data_unit_idx >= colors_tqs[current_color]['DUC']:
                            # print('---Color', colors_YCrCb[current_color], 'end')
                            current_color += 1
                            current_color %= 3
                            data_unit_idx = 0
                            if current_color == 0:
                                # todo big image - do add to img sync
                                mcu = convert_mcu(mcu, colors_tqs, headers)
                                MCUs.append(mcu)
                                # print('===MCU', mcu_idx, ' end')
                                mcu = {0: [], 1: [], 2: []}
                                mcu_idx += 1
                                # in_mcu = True
                    else:
                        zero_cnt = val >> 4
                        # print('\tzero cnt:', zero_cnt)
                        # data_unit += ([0] * zero_cnt)
                        RLE_codes.append([zero_cnt, 0])
                        data_unit_len += zero_cnt
                        data_unit_len += 1
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
    return img_show
