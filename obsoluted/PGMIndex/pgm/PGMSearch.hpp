//
// Created by BianZheng on 2022/10/12.
//

#ifndef REVERSE_KRANKS_PGMSEARCH_HPP
#define REVERSE_KRANKS_PGMSEARCH_HPP

#include "pgm_index.hpp"
#include "struct/DistancePair.hpp"
#include <iostream>
#include <memory>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class PGMSearch {

        size_t n_data_item_, n_user_;
        std::vector<pgm::PGMIndex<double, 2, 2, double>> pgm_ins_l_;
        double *preprocess_cache_;
    public:

        inline PGMSearch() {}

        inline PGMSearch(const int &n_data_item, const int &n_user) {
            this->n_data_item_ = n_data_item;
            this->n_user_ = n_user;
            this->pgm_ins_l_.resize(n_user);

        }

        void StartPreprocess() {
            preprocess_cache_ = new double[n_data_item_];
        }

        void LoopPreprocess(const double *distance_ptr, const int &userID) {
            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                preprocess_cache_[itemID] = -distance_ptr[itemID];
            }
            pgm_ins_l_[userID] = pgm::PGMIndex<double, 2, 2, double>(preprocess_cache_, n_data_item_);
            if (userID == n_user_ - 1) {
                printf("segment count %ld, height %ld, size in bytes %ld\n",
                       pgm_ins_l_[userID].segments_count(), pgm_ins_l_[userID].height(),
                       pgm_ins_l_[userID].size_in_bytes());
            }
        }

        void FinishPreprocess() {
            delete[]preprocess_cache_;
            preprocess_cache_ = nullptr;
        }

        void SearchRankBound(const double &queryIP, const int &userID,
                             int &rank_lb, int &rank_ub) const {
            pgm::ApproxPos pos = pgm_ins_l_[userID].search(queryIP);
            rank_lb = pos.hi;
            rank_ub = pos.lo;
            assert(rank_lb - rank_ub <= 2 * pgm_ins_l_[0].epsilon_value + 2);
        }

        void RankBound(const std::vector<double> &queryIP_l,
                       const std::vector<bool> &prune_l, const std::vector<bool> &result_l,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l) const {
            assert(queryIP_l.size() == n_user_);
            assert(prune_l.size() == n_user_);
            assert(result_l.size() == n_user_);
            assert(rank_lb_l.size() == n_user_);
            assert(rank_ub_l.size() == n_user_);
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID] || result_l[userID]) {
                    continue;
                }
                const double queryIP = -queryIP_l[userID];
                int lower_rank = rank_lb_l[userID];
                int upper_rank = rank_ub_l[userID];
                assert(upper_rank <= lower_rank);

                int tmp_rank_lb, tmp_rank_ub;
                SearchRankBound(queryIP, userID, tmp_rank_lb, tmp_rank_ub);

                rank_lb_l[userID] = tmp_rank_lb;
                rank_ub_l[userID] = tmp_rank_ub;
            }
        }

        void RankBound(const std::vector<std::pair<double, double>> &queryIP_l,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l) const {
            for (int userID = 0; userID < n_user_; userID++) {
                const double queryIP_lb = -queryIP_l[userID].first;
                int qIP_lb_lower_rank, qIP_lb_upper_rank;

                SearchRankBound(queryIP_lb, userID,
                                qIP_lb_lower_rank, qIP_lb_upper_rank);

                const double queryIP_ub = -queryIP_l[userID].second;
                int qIP_ub_lower_rank, qIP_ub_upper_rank;
                SearchRankBound(queryIP_ub, userID,
                                qIP_ub_lower_rank, qIP_ub_upper_rank);

                rank_lb_l[userID] = qIP_lb_lower_rank;
                rank_ub_l[userID] = qIP_ub_upper_rank;
                assert(qIP_lb_upper_rank <= qIP_lb_lower_rank);
                assert(qIP_ub_upper_rank <= qIP_ub_lower_rank);
                assert(qIP_ub_upper_rank <= qIP_lb_lower_rank);
            }
        }

        uint64_t IndexSizeByte() {
            return pgm_ins_l_[0].size_in_bytes() * n_user_;
        }

    };
}
#endif //REVERSE_KRANKS_PGMSEARCH_HPP
