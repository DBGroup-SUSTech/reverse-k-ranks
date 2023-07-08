//
// Created by BianZheng on 2022/3/10.
//

#ifndef REVERSE_K_RANKS_PRUNECANDIDATEBYBOUND_HPP
#define REVERSE_K_RANKS_PRUNECANDIDATEBYBOUND_HPP

#include <cassert>
#include <vector>
#include <algorithm>
#include "struct/RankBoundElement.hpp"

namespace ReverseMIPS {
    void
    PruneCandidateByBound(const std::vector<int> &lb_l, const std::vector<int> &ub_l,
                          const int &n_user, const int &topk,
                          std::vector<bool> &prune_l, std::vector<int> &topk_lb_heap) {
        assert(lb_l.size() == n_user);
        assert(ub_l.size() == n_user);
        assert(prune_l.size() == n_user);

        int n_cand = 0;
        int userID = 0;
        while (n_cand < topk) {
            if (prune_l[userID]) {
                userID++;
                continue;
            }
            topk_lb_heap[n_cand] = lb_l[userID];
            n_cand++;
            userID++;
        }
        std::make_heap(topk_lb_heap.begin(), topk_lb_heap.end(), std::less());
        int global_lb = topk_lb_heap.front();

        int topk_1 = topk - 1;
        for (; userID < n_user; userID++) {
            if (prune_l[userID]) {
                continue;
            }
            int tmp_lb = lb_l[userID];
            if (global_lb > tmp_lb) {
                std::pop_heap(topk_lb_heap.begin(), topk_lb_heap.end(), std::less());
                topk_lb_heap[topk_1] = tmp_lb;
                std::push_heap(topk_lb_heap.begin(), topk_lb_heap.end(), std::less());
                global_lb = topk_lb_heap.front();
            }
        }

        for (userID = 0; userID < n_user; userID++) {
            if (prune_l[userID]) {
                continue;
            }
            int tmp_ub = ub_l[userID];
            if (global_lb < tmp_ub) {
                prune_l[userID] = true;
            }
        }

    }

    void
    PruneCandidateByBound(const std::vector<int> &rank_l,
                          const int &n_user, const int &topk,
                          std::vector<bool> &prune_l) {
        assert(rank_l.size() == n_user);
        assert(prune_l.size() == n_user);
        std::vector<int> topk_heap_l(topk);

        int n_candidate = 0;
        int userID = 0;
        while (n_candidate < topk) {
            if (prune_l[userID]) {
                userID++;
                continue;
            }
            topk_heap_l[n_candidate] = rank_l[userID];
            n_candidate++;
            userID++;
        }

        std::make_heap(topk_heap_l.begin(), topk_heap_l.end(), std::less());
        int global_lb = topk_heap_l.front();

        int topk_1 = topk - 1;
        for (; userID < n_user; userID++) {
            if(prune_l[userID]){
                continue;
            }
            int tmp_lb = rank_l[userID];
            if (global_lb > tmp_lb) {
                std::pop_heap(topk_heap_l.begin(), topk_heap_l.end(), std::less());
                topk_heap_l[topk_1] = tmp_lb;
                std::push_heap(topk_heap_l.begin(), topk_heap_l.end(), std::less());
                global_lb = topk_heap_l.front();
            }
        }

        for (userID = 0; userID < n_user; userID++) {
            if (prune_l[userID]) {
                continue;
            }
            int tmp_ub = rank_l[userID];
            if (global_lb < tmp_ub) {
                prune_l[userID] = true;
            }
        }

    }

}
#endif //REVERSE_K_RANKS_PRUNECANDIDATEBYBOUND_HPP
