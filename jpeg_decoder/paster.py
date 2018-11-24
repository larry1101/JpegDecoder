from math import ceil

import numpy as np


def gen_past_matrices():
    m = np.array(range(0, 16 * 16))
    m = m.reshape((16, 16))
    return [m[0:8, 0:8], m[0:8, 8:16], m[8:16, 0:8], m[8:16, 8:16]]


past_matrices = gen_past_matrices()


def merge_mcu(mcu):
    assert len(mcu) == 4
    channels_cnt = 1 if len(mcu[0].shape) == 2 else 3
    if channels_cnt == 3:
        _mcu = np.zeros((16, 16, 3)).reshape((-1, 3))
        for data_unit_idx, data_unit in enumerate(mcu):
            _mcu[past_matrices[data_unit_idx].reshape((-1,))] = data_unit.reshape((-1, 3))
        _mcu = _mcu.reshape((16, 16, 3))
        _mcu = _mcu.astype('uint8')
        return _mcu
    else:
        # channels_cnt = 1
        _mcu = np.zeros((16, 16)).reshape((-1,))
        for data_unit_idx, data_unit in enumerate(mcu):
            _mcu[past_matrices[data_unit_idx].reshape((-1,))] = data_unit.reshape((-1,))
        _mcu = _mcu.reshape((16, 16))
        _mcu = _mcu.astype('uint8')
        return _mcu


def merge_channels(channels):
    for idx, channel in enumerate(channels):
        channels[idx] = merge_mcu(channel)
    return channels


def merge_mcus(MCUs, colors_tqs, sof0):
    if colors_tqs[0]['DUC'] == 4:
        # each mcu is 16x16x3

        row_cnt = sof0['Y']
        col_cnt = sof0['X']

        row_cnt_mcu = ceil(row_cnt / 16)
        col_cnt_mcu = ceil(col_cnt / 16)

        img_show = np.zeros((row_cnt_mcu * 16 * col_cnt_mcu * 16, 3), dtype='uint8')
        loc_map = np.arange(0, row_cnt_mcu * 16 * col_cnt_mcu * 16, dtype='int').reshape(
            (row_cnt_mcu * 16, col_cnt_mcu * 16))

        for idx, MCU in enumerate(MCUs):
            r = int(idx / col_cnt_mcu)
            c = int(idx - col_cnt_mcu * r)
            # get location matrix
            r_real = r * 16
            c_real = c * 16
            loc_matrix = loc_map[r_real:r_real + 16, c_real:c_real + 16].reshape((-1,))
            # MCU=MCU.reshape((-1, 3))
            img_show[loc_matrix] = MCU.reshape((-1, 3))
            # print('MCU', idx, 'merged')
        img_show = img_show.reshape((row_cnt_mcu * 16, col_cnt_mcu * 16, 3))
        return img_show

    else:
        raise Exception('Unsupported')
