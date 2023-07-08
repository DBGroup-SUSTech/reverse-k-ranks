import re
import numpy as np


def compile_rmips(*, fname: str, require_topk: int):
    file = open(fname)
    lines = file.read().split("\n")
    skip_queryID = -1
    result_query_l = []
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
            # if queryID == 18:
            #     print(f"queryID {queryID} skip_queryID {skip_queryID}")

            if skip_queryID == queryID:
                continue
            elif result_size >= require_topk:
                result_query_l.append((queryID, result_size, accu_ip_cost, accu_query_time))
                skip_queryID = queryID
        # else:
        #     print("No match!!")
    total_ip = np.sum([_[2] for _ in result_query_l])
    total_time = np.sum([_[3] for _ in result_query_l])
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
    dataset_l = ['amazon-home-kitchen', 'movielens-1b', 'yahoomusic_big', 'yelp']
    # dataset_l = ['amazon-home-kitchen']
    for ds in dataset_l:
        topk_info_l = []
        # for topk in [10]:
        for topk in [10, 50, 100, 150, 200]:
            fname = f'./dbg_other/{ds}.log'
            query_info_l = compile_rmips(fname=fname, require_topk=topk)
            topk_info_l.append(query_info_l)
        print(ds)
        print(topk_info_l)
