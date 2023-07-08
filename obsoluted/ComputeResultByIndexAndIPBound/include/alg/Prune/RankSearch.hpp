//
// Created by BianZheng on 2022/3/18.
//

#ifndef REVERSE_KRANKS_RANKSEARCH_HPP
#define REVERSE_KRANKS_RANKSEARCH_HPP

#include "struct/DistancePair.hpp"
#include <memory>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class RankSearch {

        int n_cache_rank_, cache_bound_every_, n_data_item_, n_user_;
        std::unique_ptr<int[]> known_rank_idx_l_; // n_cache_rank
        std::unique_ptr<double[]> bound_distance_table_; // n_user * n_cache_rank
    public:
        int n_max_disk_read_;

        inline RankSearch() {}

        inline RankSearch(const int &cache_bound_every, const int &n_data_item,
                          const int &n_user) {
            const int n_cache_rank = n_data_item / cache_bound_every;
            this->n_cache_rank_ = n_cache_rank;
            this->cache_bound_every_ = cache_bound_every;
            this->n_data_item_ = n_data_item;
            this->n_user_ = n_user;
            known_rank_idx_l_ = std::make_unique<int[]>(n_cache_rank_);
            bound_distance_table_ = std::make_unique<double[]>(n_user_ * n_cache_rank_);
            if (cache_bound_every >= n_data_item) {
                spdlog::error("cache bound every larger than n_data_item, program exit");
                exit(-1);
            }
            if (n_cache_rank_ <= 0) {
                spdlog::error("cache rank size is too small, program exit");
                exit(-1);
            }
            assert(n_cache_rank_ > 0);

            Preprocess();

        }

        void Preprocess() {
            for (int known_rank_idx = cache_bound_every_ - 1, idx = 0;
                 known_rank_idx < n_data_item_; known_rank_idx += cache_bound_every_, idx++) {
                known_rank_idx_l_[idx] = known_rank_idx;
            }

            if (n_cache_rank_ >= 2) {
                assert(known_rank_idx_l_[0] == known_rank_idx_l_[1] - (known_rank_idx_l_[0] + 1));
            }
            n_max_disk_read_ = std::max(known_rank_idx_l_[0] + 1,
                                        n_data_item_ - known_rank_idx_l_[n_cache_rank_ - 1]);

            spdlog::info("rank bound: cache_bound_every {}, n_cache_rank {}", cache_bound_every_, n_cache_rank_);
        }

        void LoopPreprocess(const DistancePair *distance_ptr, const int &userID) {
            std::vector<double> IP_l(n_data_item_);
            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                IP_l[itemID] = distance_ptr[itemID].dist_;
            }
            LoopPreprocess(IP_l.data(), userID);
        }

        void LoopPreprocess(const double *distance_ptr, const int &userID) {
            for (int crankID = 0; crankID < n_cache_rank_; crankID++) {
                int itemID = known_rank_idx_l_[crankID];
                bound_distance_table_[userID * n_cache_rank_ + crankID] = distance_ptr[itemID];
            }
        }

        inline bool
        CoarseBinarySearch(const double &queryIP, const int &userID, const int &global_lower_bucket,
                           int &rank_lb, int &rank_ub, double &IP_lb, double &IP_ub) const {
            double *search_iter = bound_distance_table_.get() + userID * n_cache_rank_;

            assert(0 <= global_lower_bucket && global_lower_bucket <= n_cache_rank_);
            if (global_lower_bucket < n_cache_rank_ &&
                queryIP < search_iter[global_lower_bucket]) {
                return true;
            }

            int bucket_ub = std::ceil(1.0 * (rank_ub - cache_bound_every_ + 1) / cache_bound_every_);
            int bucket_lb = std::floor(1.0 * (rank_lb - cache_bound_every_ + 1) / cache_bound_every_);

            bucket_lb = bucket_lb == n_cache_rank_ ? n_cache_rank_ - 1 : bucket_lb;

            double *iter_begin = search_iter + bucket_ub;
            double *iter_end = search_iter + bucket_lb + 1;

            double *lb_ptr = std::lower_bound(iter_begin, iter_end, queryIP,
                                              [](const double &arrIP, double queryIP) {
                                                  return arrIP > queryIP;
                                              });
            int bucket_idx = bucket_ub + (int) (lb_ptr - iter_begin);
            int tmp_rank_lb = known_rank_idx_l_[bucket_idx];
            int tmp_rank_ub = bucket_idx == 0 ? 0 : known_rank_idx_l_[bucket_idx - 1];

            if (tmp_rank_ub <= rank_ub && rank_lb <= tmp_rank_lb) {
                return false;
            }

            if (lb_ptr == iter_end) {
                rank_ub = tmp_rank_ub;
                IP_ub = bound_distance_table_[userID * n_cache_rank_ + bucket_idx - 1];
            } else if (lb_ptr == iter_begin) {
                rank_lb = tmp_rank_lb;
                IP_lb = bound_distance_table_[userID * n_cache_rank_ + bucket_idx];
            } else {
                rank_lb = tmp_rank_lb;
                rank_ub = tmp_rank_ub;
                IP_lb = bound_distance_table_[userID * n_cache_rank_ + bucket_idx];
                IP_ub = bound_distance_table_[userID * n_cache_rank_ + bucket_idx - 1];
            }

            return false;
        }

        void RankBound(const std::vector<double> &queryIP_l, const int &topk,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l,
                       std::vector<std::pair<double, double>> &IPbound_l,
                       std::vector<bool> &prune_l,
                       std::vector<int> &rank_topk_max_heap) const {
            assert(rank_topk_max_heap.size() == topk);
            int count = 0;
            int userID = 0;
            while (count < topk) {
                if (prune_l[userID]) {
                    userID++;
                    continue;
                }
                int lower_rank = rank_lb_l[userID];
                int upper_rank = rank_ub_l[userID];
                assert(upper_rank <= lower_rank);
                double queryIP = queryIP_l[userID];

                double IP_lb = IPbound_l[userID].first;
                double IP_ub = IPbound_l[userID].second;

                CoarseBinarySearch(queryIP, userID, n_cache_rank_,
                                   lower_rank, upper_rank, IP_lb, IP_ub);
                IPbound_l[userID] = std::make_pair(IP_lb, IP_ub);

                rank_topk_max_heap[count] = lower_rank;
                rank_lb_l[userID] = lower_rank;
                rank_ub_l[userID] = upper_rank;
                count++;
                userID++;
            }
            assert(count == topk);
            std::make_heap(rank_topk_max_heap.begin(),
                           rank_topk_max_heap.end(),
                           std::less());
            int global_lower_rank = rank_topk_max_heap.front();
            int global_lower_bucket =
                    std::floor(1.0 * (global_lower_rank - cache_bound_every_ + 1) / cache_bound_every_) + 1;
            global_lower_bucket = global_lower_bucket > n_cache_rank_ ? n_cache_rank_ : global_lower_bucket;
            assert(global_lower_bucket >= 0);
            const int topk_1 = topk - 1;

            while (userID < n_user_) {
                if (prune_l[userID]) {
                    userID++;
                    continue;
                }

                int lower_rank = rank_lb_l[userID];
                int upper_rank = rank_ub_l[userID];
                assert(upper_rank <= lower_rank);
                double queryIP = queryIP_l[userID];

                if (global_lower_rank < upper_rank) {
                    prune_l[userID] = true;
                    userID++;
                    continue;
                }

                double IP_lb = IPbound_l[userID].first;
                double IP_ub = IPbound_l[userID].second;
                bool prune = CoarseBinarySearch(queryIP, userID, global_lower_bucket,
                                                lower_rank, upper_rank, IP_lb, IP_ub);
                IPbound_l[userID] = std::make_pair(IP_lb, IP_ub);

                if (prune) {
                    prune_l[userID] = true;
                    userID++;
                    continue;
                }

                if (lower_rank < global_lower_rank) {
                    std::pop_heap(rank_topk_max_heap.begin(), rank_topk_max_heap.end(),
                                  std::less());
                    rank_topk_max_heap[topk_1] = global_lower_rank;
                    std::push_heap(rank_topk_max_heap.begin(), rank_topk_max_heap.end(),
                                   std::less());
                    global_lower_rank = rank_topk_max_heap.front();
                    global_lower_bucket =
                            std::floor(1.0 * (global_lower_rank - cache_bound_every_ + 1) / cache_bound_every_) + 1;
                    global_lower_bucket = global_lower_bucket > n_cache_rank_ ? n_cache_rank_ : global_lower_bucket;
                }

                rank_lb_l[userID] = lower_rank;
                rank_ub_l[userID] = upper_rank;
                userID++;
            }

        }

    };
}
#endif //REVERSE_KRANKS_RANKSEARCH_HPP
