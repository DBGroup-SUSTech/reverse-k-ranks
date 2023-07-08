import numpy as np
import struct


# to get the .vecs
# np.set_printoptions(threshold=np.inf)  # display all the content when print the numpy array


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy(), d


def fvecs_read(fname):
    data, d = ivecs_read(fname)
    return data.view('float32').astype(np.float32), d


def dvecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    data = a.reshape(-1, 2 * d + 1)[:, 1:].copy()
    return data.view('float64').astype(np.float64), d


def bvecs_read(fname):
    a = np.fromfile(fname, dtype='uint8')
    d = a[:4].view('uint8')[0]
    return a.reshape(-1, d + 4)[:, 4:].copy(), d


# put the part of file into cache, prevent the slow load that file is too big
def fvecs_read_mmap(fname):
    x = np.memmap(fname, dtype='int32', mode='r', order='C')
    # x = np.memmap(fname, dtype='int32')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:], d


def bvecs_read_mmap(fname):
    x = np.memmap(fname, dtype='uint8', mode='r', order='C')
    # x = np.memmap(fname, dtype='uint8')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:], d


def ivecs_read_mmap(fname):
    x = np.memmap(fname, dtype='int32', mode='r', order='C')
    # x = np.memmap(fname, dtype='int32')
    d = x[0]
    return x.reshape(-1, d + 1)[:, 1:], d


# store in format of vecs
def fvecs_write(filename, vecs):
    f = open(filename, "wb")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))  # *dimension就是int, dimension就是list
        f.write(struct.pack('f' * len(x), *x))

    f.close()


def dvecs_write(filename, vecs):
    f = open(filename, "wb")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))  # *dimension就是int, dimension就是list
        f.write(struct.pack('d' * len(x), *x))

    f.close()


def ivecs_write(filename, vecs):
    f = open(filename, "wb")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))
        f.write(struct.pack('i' * len(x), *x))

    f.close()


def bvecs_write(filename, vecs):
    f = open(filename, "wb")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))
        f.write(struct.pack('B' * len(x), *x))

    f.close()


if __name__ == '__main__':
    item_l, d = fvecs_read(
        '/home/bianzheng/Dataset/MIPS/user_item/dataset-dimension/movielens-27m-1000d/movielens-27m_item.fvecs')
    user_l, d = fvecs_read(
        '/home/bianzheng/Dataset/MIPS/user_item/dataset-dimension/movielens-27m-1000d/movielens-27m_user.fvecs')
    item_l = item_l[:, :]
    user_l = user_l[:, :]
    print(item_l)
    print(user_l)
    # fvecs_write('/home/bianzheng/Dataset/Reverse-MIPS/netflixsmall/netflixsmall_item.fvecs', item_l)
    # fvecs_write('/home/bianzheng/Dataset/Reverse-MIPS/netflixsmall/netflixsmall_user.fvecs', user_l)
    pass

# data, d = ivecs_read("data/dataset/deep/gnd-50.ivecs")
# print(data.shape)
# print(d)
