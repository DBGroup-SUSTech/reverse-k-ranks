import re
import numpy as np


def match_example():
    string = "[2022-12-02 10:55:34.548] [info] queryID 0, result_size 0, rtk_topk 1, accu_ip_cost 251266, accu_query_time 158.52s,"
    matchObj = re.match(
        r'\[.*\] \[info\] queryID (.*), result_size (.*), rtk_topk (.*), accu_ip_cost (.*), accu_query_time (.*)s,',
        string)

    if matchObj:
        queryID = int(matchObj.group(1))
        result_size = int(matchObj.group(2))
        rtk_topk = int(matchObj.group(3))
        accu_ip_cost = int(matchObj.group(4))
        accu_query_time = float(matchObj.group(5))
        print(type(queryID))
        print(queryID, result_size, rtk_topk, accu_ip_cost, accu_query_time)
    else:
        print("No match!!")


if __name__ == '__main__':
    file = open('log.log')
    lines = file.read().split("\n")
    require_topk = 50
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

            if skip_queryID == queryID:
                continue
            else:
                if result_size > require_topk:
                    result_query_l.append((queryID, result_size, accu_ip_cost, accu_query_time))
                    skip_queryID = queryID
        else:
            print("No match!!")
    print(result_query_l)
    total_ip = np.sum([_[2] for _ in result_query_l])
    total_time = np.sum([_[3] for _ in result_query_l])
    print("No.total query {}, total IP {}, total time {}s".format(len(result_query_l), total_ip, total_time))
