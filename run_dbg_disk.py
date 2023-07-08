import os
import filecmp
import numpy as np

import run_rkr
from script.data_convert import vecs_io
import run_rkr as rkr_info
import run_rkr_update as rkr_update_info


def compute_n_sample_by_memory_index_sample_only(dataset_prefix: str, memory_capacity: int):
    element_size = 4
    n_user = rkr_info.dataset_m[dataset_prefix][2]
    n_sample = memory_capacity * 1024 * 1024 * 1024 / element_size / n_user
    return int(n_sample)


def compute_n_sample_by_memory_index_sample_only_update(dataset_name: str, memory_capacity: int):
    element_size = 4
    n_user = rkr_update_info.dataset_m[dataset_name][2]
    n_sample = memory_capacity * 1024 * 1024 * 1024 / element_size / n_user
    return int(n_sample)


def compute_k_max_RMIPS(dataset_prefix: str, n_dim: int, memory_capacity: int):
    n_user = rkr_info.dataset_m[dataset_prefix][2]
    n_data_item = rkr_info.dataset_m[dataset_prefix][0]
    sizeof_float = 4
    sizeof_int = 4
    sizeof_pointer = 8
    block_l_remain_size = n_user * n_dim * sizeof_float + n_user * sizeof_pointer
    user_item_remain_size = (n_user + n_data_item) * (n_dim * sizeof_float + sizeof_int + sizeof_int + sizeof_float)
    k_max = 1.0 * (memory_capacity * 1024 * 1024 * 1024 - block_l_remain_size - user_item_remain_size) / (
            (n_user * 2 + n_data_item) * sizeof_float) / 6
    k_max = min(n_data_item, k_max)
    return int(k_max)


def test_build_score_table(dataset_name: str, eval_size_gb: int = 100000):
    os.system(
        f'cd build && ./bstb --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} --eval_size_gb {eval_size_gb}')


def run():
    dataset_l = ['amazon-home-kitchen', 'movielens-1b', 'yahoomusic_big', 'yelp']

    index_dir = f'/home/{username}/reverse-k-ranks/index'
    for dataset_prefix, n_dim in [
        ('amazon-home-kitchen', 150),
        # ('movielens-1b', 150),
        # ('yahoomusic_big', 150),
        ('yelp', 150),
    ]:
        dataset_name = f'{dataset_prefix}-{n_dim}d-update-item'

        n_data_item = rkr_info.dataset_m[dataset_prefix][0]
        n_user = rkr_info.dataset_m[dataset_prefix][2]
        sample_topk = 200

        memory_capacity = 64
        n_sample = compute_n_sample_by_memory_index_sample_only(dataset_prefix=dataset_prefix,
                                                                memory_capacity=memory_capacity)
        n_sample_item = 5000
        import multiprocessing
        n_thread = -1
        topk_l = [10, 50, 100, 150, 200]
        method_name = 'QSRPNormalLP'

        os.system(
            "cd build && ./qdibc --index_dir {} --dataset_dir {} --dataset_name {} --n_sample_item {} --sample_topk {}".format(
                index_dir, dataset_dir, dataset_name, n_sample_item, sample_topk
            ))
        os.system(
            "cd build && ./bsvdi --index_dir {} --dataset_dir {} --dataset_name {} --SIGMA {}".format(
                index_dir, dataset_dir, dataset_name, 0.7
            ))

        rkr_info.run_sample_method(index_dir, dataset_dir,
                                   method_name, dataset_name, topk_l, n_sample, n_data_item, n_user,
                                   n_sample_item, sample_topk, n_thread)

    for dataset_prefix, n_dim in [
        ('amazon-home-kitchen', 150),
        # ('movielens-1b', 150),
        # ('yahoomusic_big', 150),
        ('yelp', 150),
    ]:
        dataset_name = f'{dataset_prefix}-{n_dim}d-update-user'
        n_data_item = rkr_info.dataset_m[dataset_prefix][0]
        n_user = rkr_info.dataset_m[dataset_prefix][2]
        sample_topk = 200

        memory_capacity = 64
        n_sample = compute_n_sample_by_memory_index_sample_only(dataset_prefix=dataset_prefix,
                                                                memory_capacity=memory_capacity)
        n_sample_item = 5000
        import multiprocessing
        n_thread = -1
        topk_l = [10, 50, 100, 150, 200]
        method_name = 'QSRPNormalLP'

        os.system(
            "cd build && ./qdibc --index_dir {} --dataset_dir {} --dataset_name {} --n_sample_item {} --sample_topk {}".format(
                index_dir, dataset_dir, dataset_name, n_sample_item, sample_topk
            ))
        os.system(
            "cd build && ./bsvdi --index_dir {} --dataset_dir {} --dataset_name {} --SIGMA {}".format(
                index_dir, dataset_dir, dataset_name, 0.7
            ))

        rkr_info.run_sample_method(index_dir, dataset_dir,
                                   method_name, dataset_name, topk_l, n_sample, n_data_item, n_user,
                                   n_sample_item, sample_topk, n_thread)


"""
run template 

rkr_info.run_sample_method(index_dir, dataset_dir,
                          'QSRPNormal', dataset_name, topk_l, n_sample, n_data_item, n_user,
                          n_sample_item, sample_topk, n_thread)

method_name = 'QSRPNormalLP'
for topk in topk_l:
    os.system(
        f"cd build && ./rri --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} "
        f"--topk {topk} --method_name {method_name} --n_sample {n_sample} "
        f"--n_sample_query {n_sample_item} --sample_topk {sample_topk} --n_thread {n_thread}"
    )

"""

if __name__ == '__main__':
    username = 'zhengbian'
    dataset_dir = f'/home/{username}/Dataset/ReverseMIPS'
    index_dir = f'/home/{username}/reverse-k-ranks/index'
    dataset_l = ['amazon-home-kitchen', 'movielens-1b', 'yahoomusic_big', 'yelp']

    run()
