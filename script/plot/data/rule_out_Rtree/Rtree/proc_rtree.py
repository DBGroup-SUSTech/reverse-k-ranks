import re
import numpy as np


def compile_rmips(*, fname: str):
    file = open(fname)
    lines = file.read().split("\n")
    skip_queryID = -1
    result_query_l = []
    for line in lines:
        match_obj = re.match(
            r'\[.*\] finish queryID (.*), single_query_retrieval_time (.*)s, ip_cost (.*), n_proc_query (.*)',
            line)
        if match_obj:
            queryID = int(match_obj.group(1))
            retrieval_time = float(match_obj.group(2))
            ip_cost = float(match_obj.group(3))
            n_proc_query = float(match_obj.group(4))
            result_query_l.append((queryID, retrieval_time, ip_cost))
    total_ip = np.sum([_[2] for _ in result_query_l])
    total_time = np.sum([_[1] for _ in result_query_l])
    # print(np.setdiff1d(np.arange(670), np.array([_[0] for _ in result_query_l])))
    total_query = len([_[2] for _ in result_query_l])
    result_query_l = [total_query, total_ip, total_time]
    print("No.total query {}, total IP {}, total time {}s".format(len(result_query_l), total_ip, total_time))
    return result_query_l


def scale_time_by_ip_cost(result_query_l: list, ip_cost=3933048998, total_time=5554.200):
    pred_running_time_l = [float(_[2]) / ip_cost * total_time for _ in result_query_l]
    print(pred_running_time_l)
    return pred_running_time_l


if __name__ == '__main__':
    fname = f'./yelp-2d-sample.log'
    query_info_l = compile_rmips(fname=fname)
    print(query_info_l)
