//
// Created by BianZheng on 2022/3/10.
//

#ifndef REVERSE_K_RANKS_PRUNECANDIDATEBYBOUND_HPP
#define REVERSE_K_RANKS_PRUNECANDIDATEBYBOUND_HPP

#include "alg/TopkMaxHeap.hpp"
#include "struct/RankBoundElement.hpp"

#include <cassert>
#include <omp.h>
#include <set>
#include <vector>
#include <algorithm>

namespace ReverseMIPS {

    void
    PruneCandidateByBound(const std::vector<int> &rank_lb_l, const std::vector<int> &rank_ub_l,
                          const int &n_user, const int &topk,
                          int &refine_user_size, int &n_result_user, int &n_prune_user,
                          std::vector<char> &prune_l, std::vector<char> &result_l) {

        assert(rank_lb_l.size() == n_user);
        assert(rank_ub_l.size() == n_user);
        assert(prune_l.size() == n_user);
        assert(result_l.size() == n_user);
        assert(topk - n_result_user <= refine_user_size);

        if (n_result_user >= topk) {
            return;
        }
        assert(n_result_user < topk);

        const int heap_size = topk - n_result_user;
        std::vector<int> lbr_heap(heap_size);
        std::vector<int> ubr_heap(heap_size);
        int tmp_heap_size = 0;
        int heap_userID = 0;
        for (heap_userID = 0; heap_userID < n_user; heap_userID++) {
            if (prune_l[heap_userID] || result_l[heap_userID]) {
                continue;
            }
            lbr_heap[tmp_heap_size] = rank_lb_l[heap_userID];
            ubr_heap[tmp_heap_size] = rank_ub_l[heap_userID];
            tmp_heap_size++;
            if (tmp_heap_size == heap_size) {
                break;
            }
        }
        std::make_heap(lbr_heap.begin(), lbr_heap.end(), std::less());
        std::make_heap(ubr_heap.begin(), ubr_heap.end(), std::less());
        int tmp_min_lbr_heap = lbr_heap.front();
        int tmp_min_ubr_heap = ubr_heap.front();
        for (heap_userID = heap_userID + 1; heap_userID < n_user; heap_userID++) {
            if (prune_l[heap_userID] || result_l[heap_userID]) {
                continue;
            }
            const int &tmp_lb = rank_lb_l[heap_userID];
            const int &tmp_ub = rank_ub_l[heap_userID];
            if (tmp_min_lbr_heap > tmp_lb) {
                std::pop_heap(lbr_heap.begin(), lbr_heap.end(), std::less());
                lbr_heap[heap_size - 1] = tmp_lb;
                std::push_heap(lbr_heap.begin(), lbr_heap.end(), std::less());
                tmp_min_lbr_heap = lbr_heap.front();
            }
            if (tmp_min_ubr_heap > tmp_ub) {
                std::pop_heap(ubr_heap.begin(), ubr_heap.end(), std::less());
                ubr_heap[heap_size - 1] = tmp_ub;
                std::push_heap(ubr_heap.begin(), ubr_heap.end(), std::less());
                tmp_min_ubr_heap = ubr_heap.front();
            }
        }

        const int min_topk_lb_rank = lbr_heap.front();
        const int min_topk_ub_rank = ubr_heap.front();
        assert(min_topk_lb_rank != -1 && min_topk_ub_rank != -1);
//        printf("min_topk_lb_rank %d, min_topk_ub_rank %d\n", min_topk_lb_rank, min_topk_ub_rank);

//#define DEBUG

        if (min_topk_lb_rank == min_topk_ub_rank) {
#ifdef DEBUG
            std::set<int> ub_rank_set;
            for (int tmp_heap_userID = 0; tmp_heap_userID < heap_size; tmp_heap_userID++) {
                ub_rank_set.insert(ubr_heap[tmp_heap_userID]);
            }
            for (int tmp_heap_userID = 0; tmp_heap_userID < heap_size; tmp_heap_userID++) {
                assert(ub_rank_set.find(lbr_heap[tmp_heap_userID]) != ub_rank_set.end());
            }
#endif
            n_result_user = topk;
            refine_user_size = 0;
#pragma omp parallel for default(none) shared(n_user, rank_ub_l, rank_lb_l, min_topk_lb_rank, result_l)
            for (int userID = 0; userID < n_user; userID++) {
                if (rank_ub_l[userID] <= min_topk_lb_rank && rank_lb_l[userID] <= min_topk_lb_rank) {
                    result_l[userID] = true;
                }
            }
#ifdef DEBUG
            int count_result = 0;
            for (int userID = 0; userID < n_user; userID++) {
                if (result_l[userID]) {
                    count_result++;
                }
            }
            assert(count_result >= topk);
#endif
            return;
        }
        assert(min_topk_lb_rank != min_topk_ub_rank);

        refine_user_size = 0;
#pragma omp parallel for default(none) shared(n_user, prune_l, result_l, rank_lb_l, rank_ub_l, min_topk_lb_rank, min_topk_ub_rank) reduction(+:n_result_user, refine_user_size, n_prune_user) num_threads(omp_get_num_procs())
        for (int userID = 0; userID < n_user; userID++) {
            if (prune_l[userID] || result_l[userID]) {
                assert(prune_l[userID] ^ result_l[userID]);
                continue;
            }

            assert(!prune_l[userID] && !result_l[userID]);
            const int &tmp_lb = rank_lb_l[userID];
            const int &tmp_ub = rank_ub_l[userID];
            if (tmp_lb <= min_topk_ub_rank) { // && min_topk_lb_rank != min_topk_ub_rank
                result_l[userID] = true;
                n_result_user++;
            } else if (min_topk_lb_rank <= tmp_ub) { // && min_topk_lb_rank != min_topk_ub_rank
                prune_l[userID] = true;
                n_prune_user++;
            } else {
                assert(!prune_l[userID] && !result_l[userID]);
                refine_user_size++;
            }

        }

        assert(refine_user_size >= 0);
        if (n_result_user + refine_user_size == topk) {
#pragma omp parallel for default(none) shared(n_user, prune_l, result_l) schedule(static, 10) num_threads(omp_get_num_procs())
            for (int userID = 0; userID < n_user; userID++) {
                if (prune_l[userID] || result_l[userID]) {
                    continue;
                } else {
                    result_l[userID] = true;
                }
            }
            refine_user_size = 0;
            n_result_user = topk;
        }

        assert(refine_user_size <= n_user);

#ifdef DEBUG
        int count_result = 0;
        for (int i = 0; i < n_user; i++) {
            if (result_l[i]) {
                count_result++;
            }
        }
        assert(count_result == n_result_user);

        int count_prune = 0;
        for (int userID = 0; userID < n_user; userID++) {
            if (prune_l[userID]) {
                count_prune++;
            }
        }
        assert(count_prune == n_prune_user);

        int count_refine_user_size = 0;
        for (int userID = 0; userID < n_user; userID++) {
            if (!prune_l[userID] && !result_l[userID]) {
                count_refine_user_size++;
            }
        }
        assert(count_refine_user_size == refine_user_size);
#endif
    }

//    void
//    PruneCandidateByBound(const std::vector<int> &rank_l,
//                          const int &n_user, const int &topk,
//                          std::vector<bool> &prune_l) {
//        assert(rank_l.size() == n_user);
//        assert(prune_l.size() == n_user);
//        std::vector<int> topk_heap_l(topk);
//
//        int n_candidate = 0;
//        int userID = 0;
//        while (n_candidate < topk) {
//            if (prune_l[userID]) {
//                userID++;
//                continue;
//            }
//            topk_heap_l[n_candidate] = rank_l[userID];
//            n_candidate++;
//            userID++;
//        }
//
//        std::make_heap(topk_heap_l.begin(), topk_heap_l.end(), std::less());
//        int global_lb = topk_heap_l.front();
//
//        int topk_1 = topk - 1;
//        for (; userID < n_user; userID++) {
//            if (prune_l[userID]) {
//                continue;
//            }
//            int tmp_lb = rank_l[userID];
//            if (global_lb > tmp_lb) {
//                std::pop_heap(topk_heap_l.begin(), topk_heap_l.end(), std::less());
//                topk_heap_l[topk_1] = tmp_lb;
//                std::push_heap(topk_heap_l.begin(), topk_heap_l.end(), std::less());
//                global_lb = topk_heap_l.front();
//            }
//        }
//
//        for (userID = 0; userID < n_user; userID++) {
//            if (prune_l[userID]) {
//                continue;
//            }
//            int tmp_ub = rank_l[userID];
//            if (global_lb < tmp_ub) {
//                prune_l[userID] = true;
//            }
//        }
//
//    }

}
#endif //REVERSE_K_RANKS_PRUNECANDIDATEBYBOUND_HPP
