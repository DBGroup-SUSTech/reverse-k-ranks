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
dataset_m = {'fake-normal': [5000, 100, 1000],
             'fake-uniform': [5000, 100, 1000],
             'fakebig': [5000, 100, 5000],
             'fakehuge': [100000, 100, 100000],
             'fakehuge2': [100000, 100, 200000],
             'fakehuge3': [200000, 100, 100000],

             'ml-1m': [3706, 100, 6040],
             'lastfm': [17632, 100, 1892],

             'movielens-1b': [272038, 1000, 2197225],
             'yahoomusic_big': [135736, 1000, 1823179],
             'yelp': [159585, 1000, 2189457],
             'amazon-electronics': [475002, 1000, 4201696],
             'amazon-home-kitchen': [409243, 1000, 2511610],
             'amazon-office-products': [305800, 1000, 3404914],

             'amazon-home-kitchen_weighted_query': [409243, 1000, 2511610],
             'yelp_weighted_query': [159585, 1000, 2189457],

             'amazon-home-kitchen-150d-update-item': [401218, 1000, 2511610, -1, 8025],
             'amazon-home-kitchen-150d-update-user': [409243, 1000, 2462362, 49248, -1],
             'yelp-150d-update-item': [156455, 1000, 2189457, -1, 3130],
             'yelp-150d-update-user': [159585, 1000, 2146526, 42931, -1],

             }


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
    'QS': f'n_sample_20-n_sample_query_150-sample_topk_40-n_thread_{multiprocessing.cpu_count()}',
    'QSRPNormalLP': f'n_sample_20-n_sample_query_150-sample_topk_40-n_thread_{multiprocessing.cpu_count()}',

    'QSRPRefineComputeIPBound': f'n_sample_20-n_sample_query_150-sample_topk_40-n_thread_{multiprocessing.cpu_count()}',
    'QSRPRefineComputeAll': f'n_sample_20-n_sample_query_150-sample_topk_40-n_thread_{multiprocessing.cpu_count()}',
    'QSRPRefineLEMP': f'n_sample_20-n_sample_query_150-sample_topk_40-n_thread_{multiprocessing.cpu_count()}',

    'QSRPUniformCandidateNormalLP': 'n_sample_20-n_sample_query_150-sample_topk_40',
    'QSRPUniformLP': 'n_sample_20-n_sample_query_150-sample_topk_40',
    'US': f'n_sample_20-n_thread_{multiprocessing.cpu_count()}',
}


def cmp_file_all(baseline_method, compare_method_l, dataset_l, topk_l):
    flag = True
    for dataset_prefix, dimension in dataset_l:
        dataset_name = f'{dataset_prefix}-{dimension}d'

        for topk in topk_l:
            for method_idx in range(len(compare_method_l)):
                cmp_method = compare_method_l[method_idx]
                if baseline_method in suffix_m:
                    baseline_dir = os.path.join('result', 'rank',
                                                '{}-{}-top{}-{}-userID.csv'.format(
                                                    dataset_name, baseline_method, topk, suffix_m[baseline_method]))
                else:
                    baseline_dir = os.path.join('result', 'rank',
                                                '{}-{}-top{}-userID.csv'.format(
                                                    dataset_name, baseline_method, topk))

                if cmp_method in suffix_m:
                    cmp_dir = os.path.join('result', 'rank',
                                           '{}-{}-top{}-{}-userID.csv'.format(
                                               dataset_name, cmp_method, topk, suffix_m[cmp_method]))
                else:
                    cmp_dir = os.path.join('result', 'rank',
                                           '{}-{}-top{}-userID.csv'.format(
                                               dataset_name, cmp_method, topk))

                flag = cmp_file(baseline_dir, cmp_dir)
                if not flag:
                    print("file have diff {} {}".format(baseline_dir, cmp_dir))
                    exit(-1)
    if flag:
        print("no error, no bug")


def cmp_file_single_query_performance(baseline_method, compare_method, dataset_l, topk_l):
    flag = True
    for dataset_prefix, dimension in dataset_l:
        dataset_name = f'{dataset_prefix}-{dimension}d'

        for topk in topk_l:
            assert baseline_method in suffix_m
            baseline_dir = os.path.join('result', 'single_query_performance',
                                        '{}-{}-top{}-{}-single-query-performance.csv'.format(
                                            dataset_name, baseline_method, topk, suffix_m[baseline_method]))
            baseline_df = pd.read_csv(baseline_dir)
            compare_dir = os.path.join('result', 'single_query_performance',
                                       '{}-{}-top{}-{}-single-query-performance.csv'.format(
                                           dataset_name, compare_method, topk, suffix_m[compare_method]))
            compare_df = pd.read_csv(compare_dir)
            assert len(baseline_df) == len(compare_df)
            equal_l = baseline_df['n_refine_user'] == compare_df['n_refine_user']
            for i, is_equal in enumerate(equal_l, 0):
                if not is_equal:
                    flag = False
                    print("file have diff {} {}".format(baseline_dir, compare_dir))
                    print(f"queryID {i}, line {i + 1}")
                    print(
                        f"{baseline_method} {baseline_df['n_refine_user'].iloc[i]}, {compare_method} {compare_df['n_refine_user'].iloc[i]}")
                    exit(-1)
    if flag:
        print("no difference in single query performance")


def run_sample_method(index_dir: str, dataset_dir: str, method_name: str, dataset_name: str,
                      topk_l: list, n_sample: int,
                      n_data_item: int, n_user: int, n_sample_item: int, sample_topk: int,
                      n_thread: int,
                      other_config=""):
    os.system(
        f"cd {basic_dir}/build && ./fsr --index_dir {index_dir} --dataset_name {dataset_name} --method_name {method_name} --n_sample {n_sample} --n_data_item {n_data_item} --n_user {n_user} --n_sample_query {n_sample_item} --sample_topk {sample_topk}"
    )
    # os.system(
    #     f"cd build && ./bsibs --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} --method_name {method_name} --n_sample {n_sample} --micro_n_sample_query {n_sample_item} --sample_topk {sample_topk}")
    os.system(
        f"cd {basic_dir}/build && ./bsibc --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} --method_name {method_name} --n_sample {n_sample} --n_sample_query {n_sample_item} --sample_topk {sample_topk}")

    if method_name == "QSRPNormalLP" or \
            method_name == "QSRPUniformCandidateNormalLP" or \
            method_name == "QSRPUniformLP" or \
 \
            method_name == 'QSRPRefineComputeIPBound' or \
            method_name == 'QSRPRefineComputeAll' or \
            method_name == 'QSRPRefineLEMP':
        os.system(
            f"cd {basic_dir}/build && ./bri --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} --method_name {method_name} --n_sample {n_sample} --n_sample_query {n_sample_item} --sample_topk {sample_topk}")

    for topk in topk_l:
        os.system(
            f"cd {basic_dir}/build && ./rri --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir} "
            f"--topk {topk} --method_name {method_name} --n_sample {n_sample} "
            f"--n_sample_query {n_sample_item} --sample_topk {sample_topk} --n_thread {n_thread} {other_config}"
        )


def run(dataset_l, topk_l, index_dir, dataset_dir):
    method_name_l = [
        # 'BatchDiskBruteForce',
        'MemoryBruteForce',
        # 'DiskBruteForce',

        # 'GridIndex',
        # 'QS',
        'QSRPNormalLP',

        # 'QSRPRefineComputeIPBound',
        # 'QSRPRefineComputeAll',
        # 'QSRPRefineLEMP',

        # 'QSRPUniformLP',
        # 'QSRPUniformCandidateNormalLP',
        # 'Rtree',
        # 'RtreeItemOnly',
        # 'LinearScan',
        'US',
    ]

    os.system(f'cd {basic_dir}/result/rank && rm *')
    os.system(f'cd {basic_dir}/result/single_query_performance && rm *')
    os.system(f'cd {basic_dir}/result/vis_performance && rm *')
    os.system(f'cd {basic_dir}/index/memory_index && rm *')
    os.system(f'cd {basic_dir}/index/qrs_to_sample_index && rm *')
    os.system(f'cd {basic_dir}/index/query_distribution && rm -r *')
    os.system(f'cd {basic_dir}/index/svd_index && rm -r *')
    os.system(f'cd {basic_dir}/index/rmips_index && rm -r *')

    for dataset_prefix, n_dim in dataset_l:
        dataset_name = f'{dataset_prefix}-{n_dim}d'
        # os.system('cd build && ./bst --dataset_name {}'.format(dataset_name))
        for topk in topk_l:
            # os.system(
            #     f'cd build && ./progress --index_dir {index_dir} --basic_dir {dataset_dir} --dataset_name {dataset_name} --method_name {"BatchDiskBruteForce"} --topk {topk}')
            os.system(
                f'cd {basic_dir}/build && ./progress --index_dir {index_dir} --basic_dir {dataset_dir} --dataset_name {dataset_name} --method_name {"MemoryBruteForce"} --topk {topk}')
            # os.system(
            #     f'cd build && ./progress --index_dir {index_dir} --basic_dir {dataset_dir} --dataset_name {dataset_name} --method_name {"DiskBruteForce"} --topk {topk}')
            # os.system(
            #     f'cd build && ./rri --index_dir {index_dir} --dataset_dir {dataset_dir} --dataset_name {dataset_name} '
            #     f'--topk {topk} --method_name {"GridIndex"} --stop_time 3600')
            # os.system(
            #     f'cd build && ./rri --index_dir {index_dir} --dataset_dir {dataset_dir} --dataset_name {dataset_name} --topk {topk} --method_name {"Rtree"} --stop_time 3600')
            # os.system(
            #     f'cd build && ./rri --index_dir {index_dir} --dataset_dir {dataset_dir} --dataset_name {dataset_name} '
            #     f'--topk {topk} --method_name {"RtreeItemOnly"} --stop_time 3600')
            # os.system(
            #     f'cd build && ./rri --index_dir {index_dir} --dataset_dir {dataset_dir} --dataset_name {dataset_name} '
            #     f'--topk {topk} --method_name {"LinearScan"} --stop_time 3600')

        # n_sample_item = 150
        # sample_topk = 40
        # n_data_item = dataset_m[dataset_prefix][0]
        # n_user = dataset_m[dataset_prefix][2]
        # n_sample = 20
        # n_thread = -1
        # os.system(
        #     "cd build && ./qdibc --index_dir {} --dataset_dir {} --dataset_name {} --n_sample_item {} --sample_topk {}".format(
        #         index_dir, dataset_dir, dataset_name, n_sample_item, sample_topk
        #     ))
        # os.system(
        #     "cd build && ./bsvdi --index_dir {} --dataset_dir {} --dataset_name {} --SIGMA {}".format(
        #         index_dir, dataset_dir, dataset_name, 0.7
        #     ))
        #
        # # run_sample_method(index_dir, dataset_dir,
        # #                   'QS', dataset_name, topk_l, n_sample, n_data_item, n_user,
        # #                   n_sample_item, sample_topk, n_thread)
        # run_sample_method(index_dir, dataset_dir,
        #                   'QSRPNormalLP', dataset_name, topk_l, n_sample, n_data_item, n_user,
        #                   n_sample_item, sample_topk, n_thread)
        #
        # # run_sample_method(index_dir, dataset_dir, "QSRPRefineComputeAll", dataset_name, topk_l, n_sample, n_data_item,
        # #                   n_user, n_sample_item,
        # #                   sample_topk, n_thread)
        # # run_sample_method(index_dir, dataset_dir, 'QSRPRefineComputeIPBound', dataset_name, topk_l, n_sample,
        # #                   n_data_item, n_user,
        # #                   n_sample_item,
        # #                   sample_topk, n_thread)
        # # run_sample_method(index_dir, dataset_dir,
        # #                   'QSRPRefineLEMP', dataset_name, topk_l, n_sample, n_data_item, n_user,
        # #                   n_sample_item, sample_topk, n_thread)
        #
        # # run_sample_method(index_dir, dataset_dir,
        # #                   'QSRPUniformLP', dataset_name, topk_l, n_sample, n_data_item, n_user,
        # #                   n_sample_item, sample_topk, n_thread)
        # # run_sample_method(index_dir, dataset_dir,
        # #                   'QSRPUniformCandidateNormalLP', dataset_name, topk_l, n_sample, n_data_item, n_user,
        # #                   n_sample_item, sample_topk, n_thread)
        # run_sample_method(index_dir, dataset_dir, 'US', dataset_name, topk_l, n_sample, n_data_item, n_user,
        #                   n_sample_item, sample_topk, n_thread)

    # send_email.send('test complete')

    # topk_l = [10, 20, 30, 40, 50]
    # cmp_file_all('MemoryBruteForce', method_name_l, dataset_l, topk_l)
    # cmp_file_single_query_performance('QSRPRefineComputeAll', 'QSRPRefineComputeIPBound', dataset_l, topk_l)
    # cmp_file_single_query_performance('QSRPRefineComputeAll', 'QSRPRefineLEMP', dataset_l, topk_l)


if __name__ == '__main__':
    # dataset_l = [('fake-normal', 30), ('fake-uniform', 30), ('fakebig', 30)]
    dataset_l = [('lastfm', 150), ('ml-1m', 150)]
    # dataset_l = [('ml-1m', 150)]
    # dataset_l = [('fake-normal', 30)]
    topk_l = [10, 50, 100, 150, 200]
    # topk_l = [10, 20, 30]
    # topk_l = [10]
    index_dir = "/home/bianzheng/reverse-k-ranks/index"
    dataset_dir = "/home/bianzheng/Dataset/ReverseMIPS"
    basic_dir = '/home/bianzheng/reverse-k-ranks'

    run(dataset_l=dataset_l, topk_l=topk_l, index_dir=index_dir, dataset_dir=dataset_dir)

    dataset_l = ['lastfm', 'ml-1m']
    for dataset in dataset_l:
        os.system(f'rm -rf /home/bianzheng/reverse-k-ranks/result/rank/{dataset}')
        os.mkdir(f'/home/bianzheng/reverse-k-ranks/result/rank/{dataset}')
        os.system(
            f'mv /home/bianzheng/reverse-k-ranks/result/rank/{dataset}-* /home/bianzheng/reverse-k-ranks/result/rank/{dataset}')
