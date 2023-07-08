import numpy as np
import vecs_io
import os


def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


def gen_data_normal_update(n_user: int, n_data_item: int, n_query_item: int,
                           n_update_item: int,
                           vec_dim: int, dataset: str,
                           basic_dir: str):
    query_item_l = np.random.normal(loc=0, scale=5, size=(n_query_item, vec_dim))
    data_item_l = np.random.normal(loc=0, scale=5, size=(n_data_item, vec_dim))
    user_l = np.random.normal(loc=0, scale=5, size=(n_user, vec_dim))
    update_item_l = np.random.normal(loc=0, scale=5, size=(n_update_item, vec_dim))

    output_dir = os.path.join(basic_dir, dataset)
    delete_file_if_exist(output_dir)
    os.mkdir(output_dir)

    vecs_io.fvecs_write("%s/%s_query_item.fvecs" % (output_dir, dataset), query_item_l)
    vecs_io.fvecs_write("%s/%s_data_item.fvecs" % (output_dir, dataset), data_item_l)
    vecs_io.fvecs_write("%s/%s_user.fvecs" % (output_dir, dataset), user_l)
    vecs_io.fvecs_write("%s/%s_data_item_update.fvecs" % (output_dir, dataset), update_item_l)


def gen_data_uniform_update(n_user: int, n_data_item: int, n_query_item: int,
                            n_update_item: int,
                            vec_dim: int, dataset: str,
                            basic_dir: str):
    query_item_l = np.random.uniform(0, 100, size=(n_query_item, vec_dim))
    data_item_l = np.random.uniform(0, 100, size=(n_data_item, vec_dim))
    user_l = np.random.uniform(0, 100, size=(n_user, vec_dim))
    update_item_l = np.random.uniform(0, 100, size=(n_update_item, vec_dim))

    output_dir = os.path.join(basic_dir, dataset)
    delete_file_if_exist(output_dir)
    os.mkdir(output_dir)

    vecs_io.fvecs_write("%s/%s_query_item.fvecs" % (output_dir, dataset), query_item_l)
    vecs_io.fvecs_write("%s/%s_data_item.fvecs" % (output_dir, dataset), data_item_l)
    vecs_io.fvecs_write("%s/%s_user.fvecs" % (output_dir, dataset), user_l)
    vecs_io.fvecs_write("%s/%s_data_item_update.fvecs" % (output_dir, dataset), update_item_l)


def gen_data_independent_update(n_user: int, n_data_item: int, n_query_item: int,
                                n_update_item: int,
                                vec_dim: int, dataset: str,
                                basic_dir: str):
    query_item_l = np.random.uniform(0, 100, size=(n_query_item, vec_dim))
    data_item_l = np.random.uniform(0, 100, size=(n_data_item, vec_dim))
    user_l = np.random.uniform(0, 100, size=(n_user, vec_dim))
    update_item_l = np.random.uniform(0, 100, size=(n_update_item, vec_dim))

    output_dir = os.path.join(basic_dir, dataset)
    delete_file_if_exist(output_dir)
    os.mkdir(output_dir)

    vecs_io.fvecs_write("%s/%s_query_item.fvecs" % (output_dir, dataset), query_item_l)
    vecs_io.fvecs_write("%s/%s_data_item.fvecs" % (output_dir, dataset), data_item_l)
    vecs_io.fvecs_write("%s/%s_user.fvecs" % (output_dir, dataset), user_l)
    vecs_io.fvecs_write("%s/%s_data_item_update.fvecs" % (output_dir, dataset), update_item_l)


if __name__ == '__main__':
    # reverse k ranks是给定item, 需要输出user
    basic_dir = '/home/bianzheng/Dataset/ReverseMIPS'

    n_query_item = 100
    n_dim = 30
    n_data_item = 5000
    n_user = 1000
    n_update_item = 400

    gen_data_uniform_update(n_user, n_data_item, n_query_item,
                            n_update_item,
                            n_dim, f"fake-uniform-{n_dim}d-update-item", basic_dir)
    gen_data_normal_update(n_user, n_data_item, n_query_item,
                           n_update_item,
                           n_dim, f"fake-normal-{n_dim}d-update-item", basic_dir)

    n_query_item = 100
    n_dim = 30
    n_data_item = 5000
    n_user = 5000
    n_update_item = 4000

    gen_data_independent_update(n_user, n_data_item, n_query_item,
                                n_update_item,
                                n_dim, f'fakebig-{n_dim}d-update-item', basic_dir)
