import numpy as np
import vecs_io
import os


def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


if __name__ == '__main__':
    dataset_name = 'movielens-27m'
    new_dataset_name = 'movielens-27m-extreme'
    extrace_query_ID = [821]
    repeat_time = 1000

    basic_dir = os.path.join('/home', 'bianzheng', 'Dataset', 'ReverseMIPS')

    query_l, d = vecs_io.dvecs_read(
        os.path.join(basic_dir, dataset_name, '{}_query_item.dvecs'.format(dataset_name)))
    user_l, d = vecs_io.dvecs_read(
        os.path.join(basic_dir, dataset_name, '{}_user.dvecs'.format(dataset_name)))
    item_l, d = vecs_io.dvecs_read(
        os.path.join(basic_dir, dataset_name, '{}_data_item.dvecs'.format(dataset_name)))

    smp_query_l = query_l[extrace_query_ID]
    smp_query_l = np.tile(smp_query_l, (repeat_time, 1))

    delete_file_if_exist(os.path.join(basic_dir, new_dataset_name))
    os.mkdir(os.path.join(basic_dir, new_dataset_name))

    new_query_path = os.path.join(basic_dir, new_dataset_name, '{}_query_item.dvecs'.format(new_dataset_name))
    new_user_path = os.path.join(basic_dir, new_dataset_name, '{}_user.dvecs'.format(new_dataset_name))
    new_item_path = os.path.join(basic_dir, new_dataset_name, '{}_data_item.dvecs'.format(new_dataset_name))

    vecs_io.dvecs_write(new_query_path, smp_query_l)
    vecs_io.dvecs_write(new_user_path, user_l)
    vecs_io.dvecs_write(new_item_path, item_l)
    print("finish convert")

    print(smp_query_l)
    print(smp_query_l.shape)
