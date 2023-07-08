//
// Created by BianZheng on 2022/11/3.
//

#ifndef REVERSE_KRANKS_BITINDEX_HPP
#define REVERSE_KRANKS_BITINDEX_HPP

#include "../../../src/include/struct/DistancePair.hpp"

#include "../../../../../../usr/include/c++/9/iostream"
#include "../../../../../../usr/include/c++/9/memory"
#include "../../../../../../usr/include/c++/9/fstream"
#include "../../../../../../usr/local/include/spdlog/spdlog.h"
#include "../../../../../../usr/include/c++/9/set"
#include "../../../../../../usr/include/c++/9/numeric"

namespace ReverseMIPS {

    class BitIndex {

        size_t n_sample_, n_data_item_, n_user_, n_bit_;
        std::vector<bool> score_distribution_l_; // n_user * (n_sample_ - 1) * n_bit_
    public:

        inline BitIndex() {}

        inline BitIndex(const size_t &n_data_item, const size_t &n_user, const size_t &n_sample, const int& n_bit) {
            this->n_data_item_ = n_data_item;
            this->n_user_ = n_user;
            this->n_sample_ = n_sample;

            n_bit_ = n_bit;
            score_distribution_l_.resize(n_user_ * (n_sample_ - 1) * n_bit_);
        }

        void LoopPreprocess(const float *distance_ptr, const int *sample_rank_l, const int &userID) {

            for (int crankID = 0; crankID < n_sample_ - 1; crankID++) {
                const unsigned int rank_lb = sample_rank_l[crankID + 1];
                const unsigned int rank_ub = sample_rank_l[crankID];
                const float IP_ub = distance_ptr[rank_ub];
                const float IP_lb = distance_ptr[rank_lb];
                assert(IP_ub >= IP_lb);

                assert(0 <= rank_ub && rank_ub <= rank_lb && rank_lb < n_data_item_);

                const float distribution_distance = (IP_ub - IP_lb) / (float) (n_bit_ + 1);
                const int64_t offset = userID * (n_sample_ - 1) * n_bit_ + crankID * n_bit_;

                for (int scoreID = 0; scoreID < n_bit_; scoreID++) {
                    float sample_IP = IP_ub - (scoreID + 1) * distribution_distance;
                    const uint64_t pred_sample_rank =
                            (rank_lb - rank_ub) * (scoreID + 1) / (n_bit_ + 1) + rank_ub;
                    const float *iter_begin = distance_ptr;
                    const float *iter_end = distance_ptr + n_data_item_;

                    const float *lb_ptr = std::lower_bound(iter_begin, iter_end, sample_IP,
                                                            [](const float &arrIP, float queryIP) {
                                                                return arrIP > queryIP;
                                                            });
                    const int sample_rank = (int) (lb_ptr - iter_begin);
                    assert(0 <= sample_rank && sample_rank <= rank_lb);
                    assert(rank_ub <= pred_sample_rank && pred_sample_rank <= rank_lb);
                    const bool is_larger = sample_rank >= pred_sample_rank;
                    score_distribution_l_[offset + scoreID] = is_larger;
                }
            }

        }

        inline void ScoreDistribution(const float &queryIP, const int &userID, const int &bucketID,
                                      const float &IP_lb, const float &IP_ub,
                                      int &rank_lb, int &rank_ub) const {

            if (bucketID == 0 || bucketID == n_sample_) {
                return;
            } else { // lies between the middle
                assert(IP_ub >= queryIP && queryIP >= IP_lb);
                const float distribution_distance = (IP_ub - IP_lb) / (float) (n_bit_ + 1);
                const int itvID_ub = std::floor((IP_ub - queryIP) / distribution_distance);
                assert(0 <= itvID_ub && itvID_ub <= n_bit_);
                const int itvID_lb = itvID_ub + 1;

                unsigned int sample_rank_lb = rank_lb;
                unsigned int sample_rank_ub = rank_ub;

                const uint64_t sample_score_offset =
                        userID * (n_sample_ - 1) * n_bit_ + (bucketID - 1) * n_bit_;
                if (itvID_ub != 0) {
                    int first_true_idx = -1;
                    for (int sample_scoreID = itvID_ub - 1; sample_scoreID >= 0; sample_scoreID--) {
                        if (score_distribution_l_[sample_score_offset + sample_scoreID]) {
                            first_true_idx = sample_scoreID;
                            break;
                        }
                    }
                    if (first_true_idx != -1) {
                        const uint64_t pred_sample_rank =
                                (sample_rank_lb - sample_rank_ub) * (first_true_idx + 1) / (n_bit_ + 1) +
                                sample_rank_ub;
                        rank_ub = (int) pred_sample_rank;

                        assert(IP_ub - (first_true_idx + 1) * distribution_distance >= queryIP);
                    }
                    assert(first_true_idx == -1 ||
                           (0 <= first_true_idx && first_true_idx <= itvID_ub - 1));
                }

                if (itvID_lb != n_bit_ + 1) {
                    int first_false_idx = -1;
                    for (int sample_scoreID = itvID_lb - 1; sample_scoreID < n_bit_; sample_scoreID++) {
                        if (!score_distribution_l_[sample_score_offset + sample_scoreID]) {
                            first_false_idx = sample_scoreID;
                            break;
                        }
                    }
                    if (first_false_idx != -1) {
                        const uint64_t pred_sample_rank =
                                (sample_rank_lb - sample_rank_ub) * (first_false_idx + 1) / (n_bit_ + 1) +
                                sample_rank_ub;
                        rank_lb = (int) pred_sample_rank;

                        assert(IP_ub - (first_false_idx + 1) * distribution_distance <= queryIP);
                    }
                    assert(first_false_idx == -1 ||
                           (itvID_lb - 1 <= first_false_idx && first_false_idx <= n_bit_ - 1));
                }

            }
            assert(rank_ub <= rank_lb);
        }


        void ScoreDistributionRankBound(const std::vector<float> &queryIP_l,
                                        const std::vector<bool> &prune_l, const std::vector<bool> &result_l,
                                        const std::vector<int> &bucketID_l,
                                        const std::vector<std::pair<float, float>> &queryIP_bound_l,
                                        std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l) {
            assert(queryIP_l.size() == n_user_);
            assert(prune_l.size() == n_user_);
            assert(result_l.size() == n_user_);
            assert(queryIP_bound_l.size() == n_user_);
            assert(rank_lb_l.size() == n_user_);
            assert(rank_ub_l.size() == n_user_);
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID] || result_l[userID]) {
                    continue;
                }
                int lower_rank = rank_lb_l[userID];
                int upper_rank = rank_ub_l[userID];
                assert(upper_rank <= lower_rank);
                float queryIP = queryIP_l[userID];
                const float IP_lb = queryIP_bound_l[userID].first;
                const float IP_ub = queryIP_bound_l[userID].second;

                int bucketID = bucketID_l[userID];
                ScoreDistribution(queryIP, userID, bucketID,
                                  IP_lb, IP_ub,
                                  lower_rank, upper_rank);

                rank_lb_l[userID] = lower_rank;
                rank_ub_l[userID] = upper_rank;
            }


        }

        uint64_t IndexSizeByte() const {
            const uint64_t known_rank_idx_size = sizeof(int) * n_sample_;
            const uint64_t bound_distance_table_size = sizeof(float) * n_user_ * n_sample_;
            const uint64_t score_distribution_size = n_user_ * (n_sample_ - 1) * n_bit_ / 8;
            return known_rank_idx_size + bound_distance_table_size + score_distribution_size;
        }

    };
}
#endif //REVERSE_KRANKS_BITINDEX_HPP
