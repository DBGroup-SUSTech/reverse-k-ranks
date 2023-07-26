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
             'fake-uniform-30d-update-item': [5000, 100, 1000, -1, 400],
             'fakebig-30d-update-item': [5000, 100, 5000, -1, 4000],
             'fake-normal-30d-update-user': [5000, 100, 1000, 100, -1],
             'fake-uniform-30d-update-user': [5000, 100, 1000, 100, -1],
             'fakebig-30d-update-user': [5000, 100, 5000, 1000, -1],

             'amazon-home-kitchen-150d-update-item': [401218, 1000, 2511610, -1, 8025],
             'amazon-home-kitchen-150d-update-user': [409243, 1000, 2462362, 49248, -1],
             'yelp-150d-update-item': [156455, 1000, 2189457, -1, 3130],
             'yelp-150d-update-user': [159585, 1000, 2146526, 42931, -1],
             }

"amazon-home-kitchen-150d-update-item, n_user 2511610, n_data_item 401218, n_query 1000, n_update_item 8025"
"yelp-150d-update-item, n_user 2189457, n_data_item 156455, n_query 1000, n_update_item 3130"
"amazon-home-kitchen-150d-update-user, n_user 2462362, n_data_item 409243, n_query 1000, n_update_user 49248"
"yelp-150d-update-user, n_user 2146526, n_data_item 159585, n_query 1000, n_update_user 42931"


# def cmp_file(file1, file2):
#     method1_result_l = np.loadtxt(file1, delimiter=',', dtype=np.int32)
#     method2_result_l = np.loadtxt(file2, delimiter=',', dtype=np.int32)
#
#     assert len(method1_result_l) == len(method2_result_l)
#
#     for i in range(len(method1_result_l)):
#         intersect = set(method1_result_l[i]).intersection(set(method2_result_l[i]))
#         if len(intersect) != len(method1_result_l[i]):
#             print(f"queryID {i}, file line {i + 1}")
#             diff = set(method1_result_l[i]).difference(set(method2_result_l[i]))
#             print("diff: {}".format(diff))
#             print(f"groundtruth {method1_result_l[i]}")
#             print(f"test file {method2_result_l[i]}")
#             return False
#     return True

def cmp_file(file1, file2):
    with open(file1, 'r') as f:
        method1_result_l = []
        for line in f:
            res = list(map(int, [_ for _ in line.split(",") if _ != '\n']))
            method1_result_l.append(res)

    with open(file2, 'r') as f:
        method2_result_l = []
        for line in f:
            res = list(map(int, [_ for _ in line.split(",") if _ != '\n']))
            method2_result_l.append(res)

    assert len(method1_result_l) == len(method2_result_l)

    for i in range(len(method1_result_l)):
        intersect = set(method1_result_l[i]).intersection(set(method2_result_l[i]))
        if len(intersect) != len(method2_result_l[i]):
            print(f"queryID {i}, file line {i + 1}")
            diff = set(method1_result_l[i]).difference(set(method2_result_l[i]))
            print("diff: {}".format(diff))
            print(f"groundtruth {method1_result_l[i]}")
            print(f"test file {method2_result_l[i]}")
            return False
    return True


suffix_m = {
    'QSRPNormalLPUpdate': f'n_sample_20-n_sample_query_150-sample_topk_40-n_thread_{multiprocessing.cpu_count()}',
    'QSUpdate': f'n_sample_20-n_sample_query_150-sample_topk_40-n_thread_{multiprocessing.cpu_count()}',
}


def cmp_file_all(baseline_method, compare_method_l, dataset_l, topk_l, update_l, update_type, update_operator):
    flag = True
    for dataset_prefix, dimension in dataset_l:
        dataset_name = f'{dataset_prefix}-{dimension}d-update-item' if update_type == 'data_item' else f'{dataset_prefix}-{dimension}d-update-user'

        for updateID in update_l:
            for topk in topk_l:
                for method_idx in range(len(compare_method_l)):
                    cmp_method = compare_method_l[method_idx]
                    if baseline_method in suffix_m:
                        baseline_dir = os.path.join('result', 'rank',
                                                    '{}-{}-top{}-{}-updateID_{}-{}-{}-userID.csv'.format(
                                                        dataset_name, baseline_method, topk, suffix_m[baseline_method],
                                                        updateID, update_type, update_operator))
                    else:
                        baseline_dir = os.path.join('result', 'rank',
                                                    '{}-{}-top{}-updateID_{}-{}-{}-userID.csv'.format(
                                                        dataset_name, baseline_method, topk,
                                                        updateID, update_type, update_operator))

                    if cmp_method in suffix_m:
                        cmp_dir = os.path.join('result', 'rank',
                                               '{}-{}-top{}-{}-updateID_{}-{}-{}-userID.csv'.format(
                                                   dataset_name, cmp_method, topk, suffix_m[cmp_method],
                                                   updateID, update_type, update_operator))
                    else:
                        cmp_dir = os.path.join('result', 'rank',
                                               '{}-{}-top{}-updateID_{}-{}-{}-userID.csv'.format(
                                                   dataset_name, cmp_method, topk,
                                                   updateID, update_type, update_operator))

                    flag = cmp_file(baseline_dir, cmp_dir)
                    if not flag:
                        print("file have diff {} {}".format(baseline_dir, cmp_dir))
                        exit(-1)
    if flag:
        print("no error, no bug")
    return True


def run_sample_method_update(index_dir: str, dataset_dir: str, method_name: str, dataset_name: str,
                             n_sample: int, n_sample_item: int, sample_topk: int,
                             n_data_item: int, n_user: int,
                             update_type: str, update_operator: str, updateID_l: list,
                             topk_l: list, n_thread: int):
    os.system(
        f"cd build && ./fsr --index_dir {index_dir} --dataset_name {dataset_name} --method_name {method_name} --n_sample {n_sample} --n_data_item {n_data_item} --n_user {n_user} --n_sample_query {n_sample_item} --sample_topk {sample_topk}"
    )
    # os.system(
    #     f"cd build && ./bsibs --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} --method_name {method_name} --n_sample {n_sample} --micro_n_sample_query {n_sample_item} --sample_topk {sample_topk}")
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
    method_name_l = [
        'MemoryBruteForceUpdate',

        'QSRPNormalLPUpdate',
        'QSUpdate',
    ]

    os.system('cd result/rank && rm *')
    os.system('cd result/single_query_performance && rm *')
    os.system('cd result/vis_performance && rm *')
    os.system('cd index/memory_index && rm *')
    os.system('cd index/qrs_to_sample_index && rm *')
    os.system('cd index/query_distribution && rm -r *')
    os.system('cd index/svd_index && rm -r *')
    os.system('cd index/rmips_index && rm -r *')

    is_no_bug_l = []
    for update_type in update_type_l:
        for update_operator in update_operator_l:
            for dataset_prefix, n_dim in dataset_l:
                dataset_name = f'{dataset_prefix}-{n_dim}d-update-item' if update_type == 'data_item' else f'{dataset_prefix}-{n_dim}d-update-user'
                for topk in topk_l:
                    for updateID in updateID_l:
                        os.system(
                            f'cd build && ./rriu --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} '
                            f'--method_name {"MemoryBruteForceUpdate"} --topk {topk} '
                            f'--update_type {update_type} --update_operator {update_operator} --updateID {updateID}')

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

                # send_email.send('test complete')

                # topk_l = [10, 20, 30, 40, 50]
            no_bug_flag = cmp_file_all('MemoryBruteForceUpdate', method_name_l, dataset_l, topk_l,
                                       update_l=updateID_l,
                                       update_type=update_type, update_operator=update_operator)
            is_no_bug_l.append(no_bug_flag)
    assert len(is_no_bug_l) == len(update_type_l) * len(update_operator_l)
    for i in range(len(is_no_bug_l)):
        operator_i = i % len(update_operator_l)
        type_i = i // len(update_operator_l) % len(update_type_l)
        if is_no_bug_l[i]:
            print(f'{update_type_l[type_i]} {update_operator_l[operator_i]} {updateID_l} is no bug')


if __name__ == '__main__':
    dataset_l = [('fake-normal', 30)]
    topk_l = [10, 20, 30]
    update_type_l = ['data_item', 'user']
    update_operator_l = ['insert', 'delete']
    updateID_l = [0, 1, 2, 3, 4]
    index_dir = "/home/bianzheng/github/reverse-k-ranks/index"
    dataset_dir = "/home/bianzheng/github/reverse-k-ranks/dataset"

    run_update(dataset_l=dataset_l, topk_l=topk_l, index_dir=index_dir, dataset_dir=dataset_dir)
