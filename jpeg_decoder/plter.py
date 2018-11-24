import matplotlib.pyplot as plt


def plt_mcu(mcu, channels=None, channels_name=None, cmap='jet'):
    # mcu & channels
    if channels_name is None:
        channels_name = ['Y', 'Cr', 'Cb']
    if channels is not None:
        # both mcu and separated channels
        assert len(mcu.shape) == 3 and mcu.shape[2] == 3 and len(channels) == 3
        plt.subplot(2, 2, 1)
        plt.imshow(mcu)
        # plt.colorbar()
        for idx, channel in enumerate(channels):
            # channel = merge_mcu(channel)
            plt.subplot(2, 2, idx + 2)
            plt.title(channels_name[idx])
            plt.imshow(channel, cmap=cmap)
            plt.colorbar()
            plt.axis('off')
        plt.show()
        return

    # unmerged channel
    if len(mcu) == 3 and len(mcu[0]) == 4:
        raise Exception('MCU unmerged')
        # only 3 separated channels
        # for idx, channel in enumerate(mcu):
        #     channel = merge_mcu(channel)
        #     plt.subplot(1, 3, idx + 1)
        #     plt.title(channels_name[idx])
        #     plt.imshow(channel, cmap=cmap)
        # plt.colorbar()
        # plt.show()
        # return

    # only an unmerged mcu
    if len(mcu) == 4:
        raise Exception('MCU unmerged')
        # _mcu = merge_mcu(mcu)
        # # assert len(mcu) == 4
        # # _mcu = np.zeros((16, 16, 3)).reshape((-1,3))
        # # for data_unit_idx, data_unit in enumerate(mcu):
        # #     # plt.subplot(2, 2, data_unit_idx + 1)
        # #     # plt.imshow(data_unit)
        # #     # plt.axis('off')
        # #     _mcu[past_matrices[data_unit_idx].reshape((-1,))] = data_unit.reshape((-1,3))
        #
        # # print('++++++++++++++++++++++++++++++++++mcu')
        # # for data_unit in mcu:
        # #     print(data_unit)
        # # plt.show()
        # # plt.subplot(1, 1, 1)
        # # print('----------------------------------mcu')
        # # print(_mcu)
        # plt.imshow(_mcu)
        # plt.show()
        # return

    # channel
    if len(mcu) == 3:
        for idx, channel in enumerate(mcu):
            plt.subplot(1, 3, idx + 1)
            plt.title(channels_name[idx])
            plt.imshow(channel, cmap=cmap)
        plt.colorbar()
        plt.show()
        return

    # only an mcu
    if len(mcu.shape) == 3:
        if mcu.shape[2] == 3:
            plt.subplot(1, 1, 1)
            plt.imshow(mcu)
            plt.show()
            return


def plt_3_arrays(a, b, c):
    plt.subplot(1, 3, 1)
    plt.title('Our decoder')
    plt.axis('off')
    plt.imshow(a)
    plt.subplot(1, 3, 2)
    plt.title('OpenCV')
    plt.axis('off')
    plt.imshow(b)
    plt.subplot(1, 3, 3)
    plt.title('Delta')
    plt.axis('off')
    plt.imshow(c)
    plt.show()