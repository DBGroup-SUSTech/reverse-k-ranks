import math
import re
import numpy as np
import pandas as pd

'''first is n_item, second is n_query, third is n_user'''
dataset_m = {
    'yelp': [159585, 1000, 2189457],
    'amazon': [409243, 1000, 2511610]}


def compute_k_max_RMIPS(dataset_prefix: str, n_dim: int, memory_capacity: int):
    n_user = dataset_m[dataset_prefix][2]
    n_data_item = dataset_m[dataset_prefix][0]
    sizeof_float = 4
    sizeof_int = 4
    sizeof_pointer = 8
    block_l_remain_size = n_user * n_dim * sizeof_float + n_user * sizeof_pointer
    user_item_remain_size = (n_user + n_data_item) * (n_dim * sizeof_float + sizeof_int + sizeof_int + sizeof_float)
    k_max = 1.0 * (memory_capacity * 1024 * 1024 * 1024 - block_l_remain_size - user_item_remain_size) / (
            (n_user * 2 + n_data_item) * sizeof_float) / 6
    k_max = min(n_data_item, k_max)
    return int(k_max)


def get_rtk_ip_cost(*, dataset_name: str):
    file = open(f'./estimate_IP_count-{dataset_name}.log')
    lines = file.read().split("\n")
    rtk_ip_cost_m = {}
    for line in lines:
        match_obj = re.match(
            r'\[.*\] \[info\] EstimateTopk: rtk_topk: (.*), ip_count: (.*)',
            line)
        if match_obj:
            rtk_topk = int(match_obj.group(1))
            ip_count = int(match_obj.group(2))
            rtk_ip_cost_m[rtk_topk] = ip_count
    return rtk_ip_cost_m


def get_query_ip_cost(dataset_name: str, rtk_ip_cost_m: dict, simpfer_k_max: int, n_user: int):
    file = open(f'./belowKMax-{dataset_name}.log')
    lines = file.read().split("\n")
    exam_queryID = 0
    prev_ip_cost = 0
    query_ip_cost_l = []
    for line in lines:
        match_obj = re.match(
            r'\[.*\] \[info\] queryID (.*), result_size (.*), rtk_topk (.*), accu_ip_cost (.*), accu_query_time (.*)s,',
            line)
        if match_obj:
            queryID = int(match_obj.group(1))
            result_size = int(match_obj.group(2))
            rtk_topk = int(match_obj.group(3))
            accu_ip_cost = int(match_obj.group(4))
            accu_query_time = float(match_obj.group(5))
            if queryID != exam_queryID:
                continue
            if rtk_topk <= simpfer_k_max:
                prev_ip_cost = accu_ip_cost
                if result_size >= 100:
                    query_ip_cost_l.append(prev_ip_cost)
                    exam_queryID += 1
                    prev_ip_cost = 0
            else:
                est_topk = 512 if dataset_name == 'amazon' else 1024
                prev_ip_cost += rtk_ip_cost_m[est_topk] + n_user
                query_ip_cost_l.append(prev_ip_cost)
                exam_queryID += 1
                prev_ip_cost = 0
    return query_ip_cost_l


ip_cost_time_m = {
    'amazon': [1387927502121.33, 1705211.011],
    'yelp': [1501602561416.67, 545225.8228]
}

if __name__ == '__main__':
    dataset_name_l = ['yelp', 'amazon']
    # dataset_name_l = ['amazon']
    for dataset in dataset_name_l:
        rtk_ip_cost_m = get_rtk_ip_cost(dataset_name=dataset)
        pred_ip_cost_time_pair = ip_cost_time_m[dataset]
        n_user = dataset_m[dataset][2]
        for memory_capacity in [16, 32, 64, 128, 256]:
            # for memory_capacity in [16]:
            simpfer_k_max = compute_k_max_RMIPS(dataset, 150, memory_capacity)
            # print(f"memory capacity {memory_capacity}GB, dataset {dataset}, simpfer_k_max {simpfer_k_max}")
            query_ip_cost_l = get_query_ip_cost(dataset, rtk_ip_cost_m, simpfer_k_max, n_user)
            if memory_capacity == 16 and dataset == 'yelp':
                np.savetxt('./yelp_query_ip_cost_l.txt', query_ip_cost_l)
            assert len(query_ip_cost_l) == 1000
            total_time = np.sum(query_ip_cost_l) / pred_ip_cost_time_pair[0] * pred_ip_cost_time_pair[1]
            print(
                f"total time {total_time:.2f}s, memory capacity {memory_capacity}GB, dataset {dataset}, simpfer_k_max {simpfer_k_max}")
