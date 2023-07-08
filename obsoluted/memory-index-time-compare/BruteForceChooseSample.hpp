//
// Created by BianZheng on 2022/8/29.
//

#ifndef REVERSE_KRANKS_BRUTEFORCECHOOSESAMPLE_HPP
#define REVERSE_KRANKS_BRUTEFORCECHOOSESAMPLE_HPP

#include "util/TimeMemory.hpp"

#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>

namespace ReverseMIPS {

    int64_t ComputeCost(const std::vector<int> &sorted_user_rank_l,
                        const int64_t &n_query, const int64_t &n_user, const int64_t &n_data_item,
                        const std::vector<int> &sample_rank_l,
                        const int &n_sample_rank, const int &topk) {
        assert(sorted_user_rank_l.size() == n_query * n_user);
        assert(sample_rank_l.size() == n_sample_rank);

        std::vector<int64_t> cost_l(n_query);
#pragma omp parallel for default(none) shared(n_query, sorted_user_rank_l, n_user, topk, sample_rank_l, n_data_item, cost_l)
        for (int queryID = 0; queryID < n_query; queryID++) {
            const int topk_rank = sorted_user_rank_l[queryID * n_user + topk];
            auto lower_rank_ptr = std::lower_bound(sample_rank_l.begin(), sample_rank_l.end(), topk_rank,
                                                   [](const int &info, const int &value) {
                                                       return info < value;
                                                   });
            int topk_lower_rank = -1;
            if (lower_rank_ptr == sample_rank_l.end()) {
                topk_lower_rank = (int) n_data_item;
            } else {
                topk_lower_rank = *lower_rank_ptr;
            }
            assert(topk_lower_rank != -1);

            auto refine_ptr = std::lower_bound(sorted_user_rank_l.begin() + queryID * n_user,
                                               sorted_user_rank_l.begin() + (queryID + 1) * n_user, topk_lower_rank,
                                               [](const int &info, const int &value) {
                                                   return info <= value;
                                               });
            const int64_t n_refine = refine_ptr - (sorted_user_rank_l.begin() + queryID * n_user);
            cost_l[queryID] = n_refine;
            assert(n_refine >= topk);
        }

        int64_t cost = 0;
        for (int queryID = 0; queryID < n_query; queryID++) {
            cost += cost_l[queryID];
        }

        return cost;
    }

    void
    CalcAllCombination(const std::vector<int> &sorted_user_rank_l,
                       const int64_t &n_query, const int64_t &n_user, const int64_t n_data_item,
                       const int &n_sample_rank, const int &topk,
                       std::vector<int> &best_sample_rank_l, int64_t &min_cost) {

        min_cost = INT64_MAX;
        assert(best_sample_rank_l.size() == n_sample_rank);

        //try all the combinations
        std::vector<int> sample_rank_l(n_sample_rank);
        std::vector<int> selector(n_data_item);
        std::fill(selector.begin(), selector.begin() + n_sample_rank, 1);

        uint64_t counter = 0;
        const int report_every = 100000;
        TimeRecord record;
        record.reset();
        do {
            int size = 0;
            for (int i = 0; i < n_data_item; i++) {
                if (selector[i]) {
                    sample_rank_l[size] = i;
                    size++;
                }
            }

            int64_t cost = ComputeCost(sorted_user_rank_l,
                                       n_query, n_user, n_data_item,
                                       sample_rank_l,
                                       n_sample_rank, topk);
            assert(cost >= 0);
            if (min_cost > cost) {
                min_cost = cost;
                best_sample_rank_l.assign(sample_rank_l.begin(), sample_rank_l.end());
            }

//            combinations.push_back(selected);
//            do_sth(selected);
//            std::copy(sample_rank_l.begin(), sample_rank_l.end(), std::ostream_iterator<int>(std::cout, " "));
//            std::cout << std::endl;
            if (counter % report_every == 0 && counter != 0) {
                spdlog::info(
                        "Compute all cases table {}, {:.2f} s/iter, Mem: {} Mb",
                        counter, record.get_elapsed_time_second(),
                        get_current_RSS() / (1024 * 1024));
                record.reset();
            }
            counter++;

        } while (prev_permutation(selector.begin(), selector.end()));
    }

}
#endif //REVERSE_KRANKS_BRUTEFORCECHOOSESAMPLE_HPP
