//
// Created by BianZheng on 2022/4/15.
//

#ifndef REVERSE_KRANKS_RANKSEARCHNOEARLYTERMINATION_HPP
#define REVERSE_KRANKS_RANKSEARCHNOEARLYTERMINATION_HPP

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

        void LoopPreprocess(const double *distance_ptr, const int &userID) {
            for (int crankID = 0; crankID < n_cache_rank_; crankID++) {
                int itemID = known_rank_idx_l_[crankID];
                bound_distance_table_[userID * n_cache_rank_ + crankID] = distance_ptr[itemID];
            }
        }

        inline bool
        CoarseBinarySearch(const double &queryIP, const int &userID,
                           int &rank_lb, int &rank_ub) const {
            double *search_iter = bound_distance_table_.get() + userID * n_cache_rank_;

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

            if (lb_ptr == iter_end) {
                rank_ub = tmp_rank_ub;
            } else if (lb_ptr == iter_begin) {
                rank_lb = tmp_rank_lb;
            } else {
                rank_lb = tmp_rank_lb;
                rank_ub = tmp_rank_ub;
            }

            return false;
        }

        void RankBound(const std::vector<double> &queryIP_l, const int &topk,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l,
                       std::vector<bool> &prune_l,
                       std::vector<int> &rank_topk_max_heap) const {
            assert(rank_topk_max_heap.size() == topk);
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID]) {
                    continue;
                }
                int lower_rank = rank_lb_l[userID];
                int upper_rank = rank_ub_l[userID];
                assert(upper_rank <= lower_rank);
                double queryIP = queryIP_l[userID];

                CoarseBinarySearch(queryIP, userID, lower_rank, upper_rank);
                rank_lb_l[userID] = lower_rank;
                rank_ub_l[userID] = upper_rank;
            }

        }

    };
}
#endif //REVERSE_KRANKS_RANKSEARCHNOEARLYTERMINATION_HPP
