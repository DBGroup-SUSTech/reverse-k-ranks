import os
import filecmp
import numpy as np
import run_dbg_disk


def cmp_rtk(baseline_method, rtk_method, dataset_l, topk_l):
    suffix_m = {
        'Simpfer': 'simpfer_k_max_1',
        'SimpferPP': 'simpfer_k_max_1',
        'SimpferPPEstIPCost': 'simpfer_k_max_1',
        'SimpferPPCache': 'simpfer_k_max_25',
    }
    have_bug = False
    for dataset_prefix, n_dim in dataset_l:
        dataset_name = f'{dataset_prefix}-{n_dim}d'
        for topk in topk_l:
            if baseline_method in suffix_m:
                baseline_dir = os.path.join('result', 'rank',
                                            '{}-{}-top{}-{}-{}.csv'.format(
                                                dataset_name, baseline_method, topk, suffix_m[baseline_method],
                                                'userID'))
            else:
                baseline_dir = os.path.join('result', 'rank',
                                            '{}-{}-top{}-{}.csv'.format(dataset_name, baseline_method, topk, 'userID'))

            if rtk_method in suffix_m:
                rtk_dir = os.path.join('result', 'rank',
                                       '{}-{}-top{}-{}-{}.csv'.format(
                                           dataset_name, rtk_method, topk, suffix_m[rtk_method],
                                           'userID'))
            else:
                rtk_dir = os.path.join('result', 'rank',
                                       '{}-{}-top{}-{}.csv'.format(
                                           dataset_name, rtk_method, topk, 'userID'))

            with open(rtk_dir, 'r') as f:
                rtk_result_l = []
                for line in f:
                    res = set(map(int, [_ for _ in line.split(",") if _ != '\n']))
                    rtk_result_l.append(res)

            baseline_result_l = np.loadtxt(baseline_dir, delimiter=',', dtype=np.int32)

            assert len(baseline_result_l) == len(rtk_result_l)

            for i in range(len(baseline_result_l)):
                intersect = set(baseline_result_l[i]).intersection(rtk_result_l[i])
                if len(intersect) != len(baseline_result_l[i]):
                    print("have bug dataset {}, topk {}, queryID {}".format(dataset_name, topk, i))
                    print(f"queryID {i} line {i + 1}")
                    print(f"baseline result {baseline_result_l[i]}")
                    print(f"method result {rtk_result_l[i]}")
                    diff = set(baseline_result_l[i]).difference(set(rtk_result_l[i]))
                    print("diff: {}".format(diff))
                    have_bug = True
                    exit(-1)
    if not have_bug:
        print("no error, no bug")


def run():
    os.system('cd result/rank && rm *')
    os.system('cd result/single_query_performance && rm *')
    os.system('cd result/vis_performance && rm *')
    os.system('cd index/memory_index && rm *')
    os.system('cd index/qrs_to_sample_index && rm *')
    os.system('cd index/query_distribution && rm -r *')

    for dataset_prefix, n_dim in dataset_l:
        dataset_name = f'{dataset_prefix}-{n_dim}d'
        for topk in topk_l:
            os.system(
                f'cd build && ./progress --dataset_name {dataset_name} --method_name {"BatchDiskBruteForce"} --topk {topk}')

            # os.system(f'cd build && ./progress --dataset_name {ds} --method_name {"DiskBruteForce"} --topk {topk}')
            # os.system(f'cd build && ./progress --dataset_name {ds} --method_name {"MemoryBruteForce"} --topk {topk}')

            # os.system(
            #     f'cd build && ./rri --dataset_name {dataset_name} --topk {topk} --method_name {"Simpfer"} --stop_time {10000000}')

            # import run_dbg_disk as dbg
            # k_max = dbg.compute_k_max_RMIPS(dataset_prefix, n_dim, 1)
            # os.system(
            #     f'cd build && ./rri --dataset_name {dataset_name} --topk {topk} --method_name {"SimpferPP"} --simpfer_k_max {1} --stop_time {10000000}')

            # simpfer_k_max = run_dbg_disk.compute_k_max_RMIPS(dataset_prefix=dataset_prefix, memory_capacity=16,
            #                                                  n_dim=n_dim)
            simpfer_k_max = 25
            os.system(
                f'cd build && ./bsppci --dataset_dir {dataset_dir} --dataset_name {dataset_name} --index_dir {index_dir}')
            os.system(
                f'cd build && ./rri --dataset_name {dataset_name} --topk {topk} --method_name {"SimpferPPCache"} '
                f'--simpfer_k_max {simpfer_k_max}')

            # os.system(
            #     f'cd build && ./rtk --dataset_name {dataset_name} --topk {topk} --method_name {"SimpferPPAboveKMax"} --stop_time {10000000}')
            # os.system(
            #     f'cd build && ./rtk --dataset_name {dataset_name} --topk {topk} --method_name {"SimpferPPBelowKMax"} --stop_time {10000000}')
            # os.system(
            #     f'cd build && ./rtk --dataset_name {dataset_name} --topk {topk} --method_name {"SimpferPPEstIPCost"} --simpfer_k_max {1} '
            #     f'--stop_time {10000000} '
            #     f'--min_cache_topk {16} --max_cache_topk {4096}')

    # cmp_rtk('BatchDiskBruteForce', 'SimpferPP', dataset_l, topk_l)
    cmp_rtk('BatchDiskBruteForce', 'SimpferPPCache', dataset_l, topk_l)
    # cmp_rtk('BatchDiskBruteForce', 'Simpfer', dataset_l, topk_l)


if __name__ == '__main__':
    dataset_l = [('fake-normal', 30), ('fake-uniform', 30), ('fakebig', 30)]
    # dataset_l = [('yahoomusic_big', 150)]
    # dataset_l = [('fake-uniform', 30)]
    topk_l = [10, 20, 30]
    # topk_l = [30]

    dataset_dir = "/home/bianzheng/Dataset/ReverseMIPS"
    index_dir = "/home/bianzheng/reverse-k-ranks/index"

    run()
