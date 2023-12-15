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


'''first is n_item, second is n_query, third is n_user, then is n_update_user, then is n_update_item'''
dataset_m = {'fake-normal-30d-update-item': [5000, 100, 1000, -1, 400],
             'fake-normal-30d-update-user': [5000, 100, 1000, 100, -1], }


def run_sample_method_update(index_dir: str, dataset_dir: str, method_name: str, dataset_name: str,
                             n_sample: int, n_sample_item: int, sample_topk: int,
                             n_data_item: int, n_user: int,
                             update_type: str, update_operator: str, updateID_l: list,
                             topk_l: list, n_thread: int):
    os.system(
        f"cd build && ./fsr --index_dir {index_dir} --dataset_name {dataset_name} --method_name {method_name} --n_sample {n_sample} --n_data_item {n_data_item} --n_user {n_user} --n_sample_query {n_sample_item} --sample_topk {sample_topk}"
    )
    os.system(
        f"cd build && ./bsibc --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} --method_name {method_name} --n_sample {n_sample} --n_sample_query {n_sample_item} --sample_topk {sample_topk}")

    if method_name == "QSRPNormalLPUpdate":
        os.system(
            f"cd build && ./bri --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} --method_name {method_name} --n_sample {n_sample} --n_sample_query {n_sample_item} --sample_topk {sample_topk}")

    for topk in topk_l:
        for updateID in updateID_l:
            os.system(
                f"cd build && ./rriu --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} "
                f"--method_name {method_name} --n_sample {n_sample} --n_sample_query {n_sample_item} --sample_topk {sample_topk} "
                f"--update_type {update_type} --update_operator {update_operator} --updateID {updateID} "
                f"--topk {topk} --n_thread {n_thread} "
            )


def run_update(dataset_l, topk_l, index_dir, dataset_dir):
    os.system('cd result/rank && rm *')
    os.system('cd result/single_query_performance && rm *')
    os.system('cd result/vis_performance && rm *')
    os.system('cd index/memory_index && rm *')
    os.system('cd index/qrs_to_sample_index && rm *')
    os.system('cd index/query_distribution && rm -r *')
    os.system('cd index/svd_index && rm -r *')
    os.system('cd index/rmips_index && rm -r *')

    for update_type in update_type_l:
        for update_operator in update_operator_l:
            for dataset_prefix, n_dim in dataset_l:
                dataset_name = f'{dataset_prefix}-{n_dim}d-update-item' if update_type == 'data_item' else f'{dataset_prefix}-{n_dim}d-update-user'

                n_sample_item = 150
                sample_topk = 40
                n_data_item = dataset_m[dataset_name][0]
                n_user = dataset_m[dataset_name][2]
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

                run_sample_method_update(index_dir, dataset_dir,
                                         'QSRPNormalLPUpdate', dataset_name,
                                         n_sample, n_sample_item, sample_topk,
                                         n_data_item, n_user,
                                         update_type, update_operator, updateID_l,
                                         topk_l, n_thread)

                run_sample_method_update(index_dir, dataset_dir,
                                         'QSUpdate', dataset_name,
                                         n_sample, n_sample_item, sample_topk,
                                         n_data_item, n_user,
                                         update_type, update_operator, updateID_l,
                                         topk_l, n_thread)


if __name__ == '__main__':
    dataset_l = [('fake-normal', 30)]
    topk_l = [10, 20, 30]
    update_type_l = ['data_item', 'user']
    update_operator_l = ['insert', 'delete']
    updateID_l = [0, 1, 2, 3, 4]
    index_dir = "/home/bianzheng/reverse-k-ranks-github/reverse-k-ranks/index"
    dataset_dir = "/home/bianzheng/reverse-k-ranks-github/reverse-k-ranks/dataset"

    run_update(dataset_l=dataset_l, topk_l=topk_l, index_dir=index_dir, dataset_dir=dataset_dir)
