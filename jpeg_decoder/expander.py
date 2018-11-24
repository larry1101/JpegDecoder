import numpy as np


LEN = 8


def gen_expand_matrices(LEN):
    array = []
    for row in range(1, LEN * LEN * 2, LEN * 2):
        r = list(range(row, row + LEN * 2))
        r = np.array(r)
        r = r / 2
        r = np.ceil(r).astype('int')
        array.append(r)
        array.append(r)

    array = np.array(array)
    array -= 1
    # print(array)

    a = array[0:LEN, 0:LEN]
    # print(a)
    b = array[0:LEN, LEN:LEN * 2]
    # print(b)
    c = array[LEN:LEN * 2, 0:LEN]
    # print(c)
    d = array[LEN:LEN * 2, LEN:LEN * 2]
    # print(d)

    return np.array([
        [a, b],
        [c, d]
    ])


expand_matrices = gen_expand_matrices(LEN)


def expand(source, location):
    return source.reshape((-1,))[expand_matrices[location[0], location[1]].reshape((-1,))].reshape(
        expand_matrices[0, 0].shape)
