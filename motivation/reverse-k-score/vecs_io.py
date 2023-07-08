import os
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
    username = 'bianzheng'
    basic_dir = f'/home/{username}/Dataset/ReverseMIPS/'
    dataset_l = ['amazon-electronics', 'amazon-home-kitchen', 'amazon-office-products', 'fake-normal', 'fake-uniform',
                 'fakebig', 'movielens-1b',
                 'yahoomusic_big', 'yelp']
    n_dim = 150
    for dataset in dataset_l:
        data_item_l, d = fvecs_read(os.path.join(basic_dir, dataset, f'{dataset}_data_item.fvecs'))
        user_l, d = fvecs_read(os.path.join(basic_dir, dataset, f'{dataset}_user.fvecs'))
        query_item_l, d = fvecs_read(os.path.join(basic_dir, dataset, f'{dataset}_query_item.fvecs'))

        n_data_item = len(data_item_l)
        n_user = len(user_l)
        n_query = len(query_item_l)
        index_size_tb = n_data_item * n_user * 4 / 1024 / 1024 / 1024 / 1024
        print(
            f"{dataset}, data_item {data_item_l.shape}, user {user_l.shape}, query_item {query_item_l.shape}, index size {index_size_tb:.2f}TB")
