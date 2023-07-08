import os
import numpy as np
import run_rkr as rkr_info
import run_dbg_disk as dbg_disk


def run():
    # dataset_l = ['movielens-27m', 'netflix', 'yahoomusic_big', 'yelp', 'amazon-home-kitchen']
    dataset_l = ['amazon-home-kitchen', 'movielens-1b', 'yahoomusic_big', 'yelp']
    # dataset_l = ['amazon-home-kitchen']
    # dataset_l = ['netflix', 'movielens-27m']

    for dataset_prefix, n_dim in [
        ('movielens-1b', 150),
    ]:
        dataset_name = f'{dataset_prefix}-{n_dim}d'
        memory_capacity = 16
        n_data_item = rkr_info.dataset_m[dataset_prefix][0]
        n_user = rkr_info.dataset_m[dataset_prefix][2]
        n_sample = dbg_disk.compute_n_sample_by_memory_index_sample_only(dataset_prefix=dataset_prefix,
                                                                         memory_capacity=memory_capacity)
        simpfer_k_max = dbg_disk.compute_k_max_RMIPS(dataset_prefix=dataset_prefix, n_dim=n_dim,
                                                     memory_capacity=memory_capacity)
        n_sample_item = 5000
        sample_topk = 200
        topk_l = [10, 50, 100, 150, 200]

        stop_time = 14400

        for topk in topk_l:
            os.system(
                f'cd build && ./rri --index_dir {index_dir} --dataset_dir {dataset_dir} --dataset_name {dataset_name} '
                f'--topk {topk} --method_name {"GridIndex"} --stop_time {stop_time}')

    for dataset_prefix, n_dim in [
        ('yelp', 64),
        ('yelp', 512),
    ]:
        dataset_name = f'{dataset_prefix}-{n_dim}d'
        memory_capacity = 16
        n_data_item = rkr_info.dataset_m[dataset_prefix][0]
        n_user = rkr_info.dataset_m[dataset_prefix][2]
        n_sample = dbg_disk.compute_n_sample_by_memory_index_sample_only(dataset_prefix=dataset_prefix,
                                                                         memory_capacity=memory_capacity)
        simpfer_k_max = dbg_disk.compute_k_max_RMIPS(dataset_prefix=dataset_prefix, n_dim=n_dim,
                                                     memory_capacity=memory_capacity)
        n_sample_item = 5000
        sample_topk = 200
        topk_l = [100]

        stop_time = 21600

        for topk in topk_l:
            os.system(
                f'cd build && ./rri --index_dir {index_dir} --dataset_dir {dataset_dir} --dataset_name {dataset_name} '
                f'--topk {topk} --method_name {"GridIndex"} --stop_time {stop_time}')


if __name__ == '__main__':
    index_dir = os.path.join('/home', 'zhengbian', 'reverse-k-ranks', 'index')
    dataset_dir = os.path.join('/home', 'zhengbian', 'Dataset', 'ReverseMIPS')
    # index_dir = os.path.join('/data', 'ReverseMIPS')
    # dataset_l = ['movielens-27m', 'netflix', 'yahoomusic', 'yelp']
    # dataset_l = ['netflix-small', 'movielens-27m-small']
    run()
