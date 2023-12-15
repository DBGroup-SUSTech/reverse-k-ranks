import os
import numpy as np
import pandas as pd
import multiprocessing


class CMDcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


'''first is n_item, second is n_query, third is n_user'''
dataset_m = {'fake-normal': [5000, 100, 1000], }


def run_sample_method(index_dir: str, dataset_dir: str, method_name: str, dataset_name: str,
                      topk_l: list, n_sample: int,
                      n_data_item: int, n_user: int, n_sample_item: int, sample_topk: int,
                      n_thread: int,
                      other_config=""):
    os.system(
        f"cd build && ./fsr --index_dir {index_dir} --dataset_name {dataset_name} --method_name {method_name} --n_sample {n_sample} --n_data_item {n_data_item} --n_user {n_user} --n_sample_query {n_sample_item} --sample_topk {sample_topk}"
    )
    os.system(
        f"cd build && ./bsibc --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} --method_name {method_name} --n_sample {n_sample} --n_sample_query {n_sample_item} --sample_topk {sample_topk}")

    if method_name == "QSRPNormalLP" or \
            method_name == "QSRPUniformLP":
        os.system(
            f"cd build && ./bri --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} --method_name {method_name} --n_sample {n_sample} --n_sample_query {n_sample_item} --sample_topk {sample_topk}")

    for topk in topk_l:
        os.system(
            f"cd build && ./rri --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} "
            f"--topk {topk} --method_name {method_name} --n_sample {n_sample} "
            f"--n_sample_query {n_sample_item} --sample_topk {sample_topk} --n_thread {n_thread} {other_config}"
        )


def run(dataset_l, topk_l, index_dir, dataset_dir):
    os.system('cd result/rank && rm *')
    os.system('cd result/single_query_performance && rm *')
    os.system('cd result/vis_performance && rm *')
    os.system('cd index/memory_index && rm *')
    os.system('cd index/qrs_to_sample_index && rm *')
    os.system('cd index/query_distribution && rm -r *')
    os.system('cd index/svd_index && rm -r *')
    os.system('cd index/rmips_index && rm -r *')

    for dataset_prefix, n_dim in dataset_l:
        dataset_name = f'{dataset_prefix}-{n_dim}d'
        for topk in topk_l:
            os.system(
                f'cd build && ./rri --index_dir {index_dir} --dataset_dir {dataset_dir} --dataset_name {dataset_name} '
                f'--topk {topk} --method_name {"GridIndex"} --stop_time 3600')
            os.system(
                f'cd build && ./rri --index_dir {index_dir} --dataset_dir {dataset_dir} --dataset_name {dataset_name} --topk {topk} --method_name {"Rtree"} --stop_time 3600')

        n_sample_item = 150
        sample_topk = 40
        n_data_item = dataset_m[dataset_prefix][0]
        n_user = dataset_m[dataset_prefix][2]
        n_sample = 20
        n_thread = -1
        os.system(
            "cd build && ./qdibc --index_dir {} --dataset_dir {} --dataset_name {} --n_sample_item {} --sample_topk {}".format(
                index_dir, dataset_dir, dataset_name, n_sample_item, sample_topk
            ))
        os.system(
            "cd build && ./bsvdi --index_dir {} --dataset_dir {} --dataset_name {} --SIGMA {}".format(
                index_dir, dataset_dir, dataset_name, 0.7
            ))

        run_sample_method(index_dir, dataset_dir,
                          'QS', dataset_name, topk_l, n_sample, n_data_item, n_user,
                          n_sample_item, sample_topk, n_thread)
        run_sample_method(index_dir, dataset_dir,
                          'QSRPNormalLP', dataset_name, topk_l, n_sample, n_data_item, n_user,
                          n_sample_item, sample_topk, n_thread)

        run_sample_method(index_dir, dataset_dir,
                          'QSRPUniformLP', dataset_name, topk_l, n_sample, n_data_item, n_user,
                          n_sample_item, sample_topk, n_thread)
        run_sample_method(index_dir, dataset_dir, 'US', dataset_name, topk_l, n_sample, n_data_item, n_user,
                          n_sample_item, sample_topk, n_thread)


if __name__ == '__main__':
    dataset_l = [('fake-normal', 30)]
    topk_l = [10, 20, 30]
    # topk_l = [10]
    index_dir = "/home/bianzheng/reverse-k-ranks-github/reverse-k-ranks/index"
    dataset_dir = "/home/bianzheng/reverse-k-ranks-github/reverse-k-ranks/dataset"

    run(dataset_l=dataset_l, topk_l=topk_l, index_dir=index_dir, dataset_dir=dataset_dir)
