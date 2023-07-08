import os
import filecmp
import numpy as np

import run_rkr
from script.data_convert import vecs_io
import run_rkr_update as rkr_update_info


def compute_n_sample_by_memory_index_sample_only_update(dataset_name: str, memory_capacity: int):
    element_size = 4
    n_user = rkr_update_info.dataset_m[dataset_name][2]
    n_sample = memory_capacity * 1024 * 1024 * 1024 / element_size / n_user
    return int(n_sample)


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
        update_type = 'data_item'
        n_data_item = rkr_update_info.dataset_m[dataset_name][0]
        n_user = rkr_update_info.dataset_m[dataset_name][2]
        sample_topk = 200

        memory_capacity = 64
        n_sample = compute_n_sample_by_memory_index_sample_only_update(dataset_name=dataset_name,
                                                                       memory_capacity=memory_capacity)
        n_sample_item = 5000
        n_thread = -1
        topk_l = [100]
        updateID_l = [0, 1, 2, 3, 4]
        update_operator = 'insert'
        method_name = 'QSRPNormalLPUpdate'

        os.system(
            "cd build && ./qdibc --index_dir {} --dataset_dir {} --dataset_name {} --n_sample_item {} --sample_topk {}".format(
                index_dir, dataset_dir, dataset_name, n_sample_item, sample_topk
            ))
        os.system(
            "cd build && ./bsvdi --index_dir {} --dataset_dir {} --dataset_name {} --SIGMA {}".format(
                index_dir, dataset_dir, dataset_name, 0.7
            ))

        rkr_update_info.run_sample_method_update(index_dir, dataset_dir, method_name, dataset_name,
                                                 n_sample, n_sample_item, sample_topk,
                                                 n_data_item, n_user,
                                                 update_type, update_operator, updateID_l,
                                                 topk_l, n_thread)

        update_operator = 'delete'
        for topk in topk_l:
            for updateID in updateID_l:
                os.system(
                    f"cd build && ./rriu --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} "
                    f"--method_name {method_name} --n_sample {n_sample} --n_sample_query {n_sample_item} --sample_topk {sample_topk} "
                    f"--update_type {update_type} --update_operator {update_operator} --updateID {updateID} "
                    f"--topk {topk} --n_thread {n_thread} "
                )

    for dataset_prefix, n_dim in [
        ('amazon-home-kitchen', 150),
        # ('movielens-1b', 150),
        # ('yahoomusic_big', 150),
        ('yelp', 150),
    ]:
        dataset_name = f'{dataset_prefix}-{n_dim}d-update-user'
        update_type = 'user'
        n_data_item = rkr_update_info.dataset_m[dataset_name][0]
        n_user = rkr_update_info.dataset_m[dataset_name][2]
        sample_topk = 200

        memory_capacity = 64
        n_sample = compute_n_sample_by_memory_index_sample_only_update(dataset_name=dataset_name,
                                                                       memory_capacity=memory_capacity)
        n_sample_item = 5000
        n_thread = -1
        topk_l = [100]
        updateID_l = [0, 1, 2, 3, 4]
        update_operator = 'insert'
        method_name = 'QSRPNormalLPUpdate'

        os.system(
            "cd build && ./qdibc --index_dir {} --dataset_dir {} --dataset_name {} --n_sample_item {} --sample_topk {}".format(
                index_dir, dataset_dir, dataset_name, n_sample_item, sample_topk
            ))
        os.system(
            "cd build && ./bsvdi --index_dir {} --dataset_dir {} --dataset_name {} --SIGMA {}".format(
                index_dir, dataset_dir, dataset_name, 0.7
            ))

        rkr_update_info.run_sample_method_update(index_dir, dataset_dir, method_name, dataset_name,
                                                 n_sample, n_sample_item, sample_topk,
                                                 n_data_item, n_user,
                                                 update_type, update_operator, updateID_l,
                                                 topk_l, n_thread)

        update_operator = 'delete'
        for topk in topk_l:
            for updateID in updateID_l:
                os.system(
                    f"cd build && ./rriu --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} "
                    f"--method_name {method_name} --n_sample {n_sample} --n_sample_query {n_sample_item} --sample_topk {sample_topk} "
                    f"--update_type {update_type} --update_operator {update_operator} --updateID {updateID} "
                    f"--topk {topk} --n_thread {n_thread} "
                )


"""
run template 

rkr_update_info.run_sample_method_update(index_dir, dataset_dir, method_name, dataset_name,
                                                 n_sample, n_sample_item, sample_topk,
                                                 n_data_item, n_user,
                                                 update_type, update_operator, updateID_l,
                                                 topk_l, n_thread)

method_name = 'QSUpdate'
for topk in topk_l:
    for updateID in updateID_l:
        os.system(
            f"cd build && ./rriu --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} "
            f"--method_name {method_name} --n_sample {n_sample} --n_sample_query {n_sample_item} --sample_topk {sample_topk} "
            f"--update_type {update_type} --update_operator {update_operator} --updateID {updateID} "
            f"--topk {topk} --n_thread {n_thread} "
        )

"""

if __name__ == '__main__':
    username = 'zhengbian'
    dataset_dir = f'/home/{username}/Dataset/ReverseMIPS'
    index_dir = f'/home/{username}/reverse-k-ranks/index'
    dataset_l = ['amazon-home-kitchen', 'movielens-1b', 'yahoomusic_big', 'yelp']

    run()
