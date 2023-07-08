import copy

import numpy as np
import itertools
import bisect

if __name__ == '__main__':
    # np.random.seed(0)
    n_item = 8  # 规定最后一个ID是query
    n_user = 6
    n_query = 4
    k = 2
    n_sample = 3
    query_rank_l = np.random.randint(low=0, high=n_item, size=n_query * n_user).reshape(n_query, n_user)
    # query_rank_l = np.array([[6, 4, 0, 4, 6, 5],
    #                          [7, 5, 2, 4, 5, 0],
    #                          [0, 7, 1, 2, 1, 0],
    #                          [2, 3, 7, 4, 0, 5]], dtype=np.int32)
    print(query_rank_l)

    # 对同一个user有多少种采样方式
    item_combination_l = list(itertools.combinations(np.arange(n_item), n_sample))
    # 对不同的user可以搞出多少种采样方式
    # print(len(item_combination_l))
    user_permutation_l = list(itertools.combinations_with_replacement(np.arange(len(item_combination_l)), n_user))
    print(len(user_permutation_l))
    min_strategyID = -1
    min_fetch_disk = n_item * n_user * n_query
    min_strategy_bound_l = []
    min_n_fetch_disk_l = []
    min_query_bound_l = []
    for strategyID, sample_way in enumerate(user_permutation_l, 0):
        # 得到不同的采样方法
        n_fetch_disk = 0
        strategy_bound_l = [item_combination_l[idx] for idx in sample_way]
        n_fetch_disk_l = []
        query_bound_l = []
        for rank_l in query_rank_l:
            # 对不同的query进行测量
            rank_lb_l = []
            rank_ub_l = []
            assert len(sample_way) == n_user
            for userID in range(n_user):
                # 得到该user下面的采样方式
                sample_rank_l = item_combination_l[sample_way[userID]]
                query_this_rank = rank_l[userID]
                rank_lb = -1
                rank_ub = -1
                if query_this_rank <= sample_rank_l[0]:
                    rank_lb = sample_rank_l[0]
                    rank_ub = 0
                elif query_this_rank > sample_rank_l[n_sample - 1]:
                    rank_lb = n_item - 1
                    rank_ub = sample_rank_l[n_sample - 1] + 1
                else:
                    ranklb_idx = bisect.bisect_left(sample_rank_l, query_this_rank)
                    rank_lb = sample_rank_l[ranklb_idx]
                    rank_ub = sample_rank_l[ranklb_idx - 1] + 1
                assert rank_lb != -1 and rank_ub != -1
                rank_lb_l.append(rank_lb)
                rank_ub_l.append(rank_ub)
            lbtop = np.sort(rank_lb_l)[k - 1]
            candidate_idx_l = [i for i, ubrank in enumerate(rank_ub_l, 0) if ubrank <= lbtop]
            tmp_n_fetch_disk = 0
            for userID in candidate_idx_l:
                n_fetch_disk += (rank_lb_l[userID] - rank_ub_l[userID])
                tmp_n_fetch_disk += (rank_lb_l[userID] - rank_ub_l[userID])
            n_fetch_disk_l.append(tmp_n_fetch_disk)
            query_bound_l.append([(rank_ub_l[i], rank_lb_l[i]) for i in range(len(rank_lb_l))])

        if min_fetch_disk > n_fetch_disk:
            min_fetch_disk = n_fetch_disk
            min_strategyID = strategyID
            min_strategy_bound_l = copy.copy(strategy_bound_l)
            min_n_fetch_disk_l = copy.copy(n_fetch_disk_l)
            min_query_bound_l = copy.copy(query_bound_l)
    print(min_fetch_disk)
    print(min_strategy_bound_l)
    print(query_rank_l)
    print(min_n_fetch_disk_l)
    print(min_query_bound_l)
