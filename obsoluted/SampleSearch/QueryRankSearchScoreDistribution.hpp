//
// Created by BianZheng on 2022/10/17.
//

#ifndef REVERSE_K_RANKS_QUERYRANKSEARCHSCOREDISTRIBUTION_HPP
#define REVERSE_K_RANKS_QUERYRANKSEARCHSCOREDISTRIBUTION_HPP

#include "struct/DistancePair.hpp"

#include <iostream>
#include <memory>
#include <fstream>
#include <spdlog/spdlog.h>
#include <set>
#include <numeric>

namespace ReverseMIPS {

    class QueryRankSearchScoreDistribution {

        size_t n_sample_, n_data_item_, n_user_, n_sample_score_;
        size_t n_sample_query_, sample_topk_;
        std::unique_ptr<double[]> bound_distance_table_; // n_user * n_sample_
        std::vector<bool> score_distribution_l_; // n_user * (n_sample_ - 1) * n_sample_score_
    public:
        std::unique_ptr<int[]> known_rank_idx_l_; // n_sample_

        inline QueryRankSearchScoreDistribution() {}

        inline QueryRankSearchScoreDistribution(const char *index_basic_dir, const char *dataset_name,
                                                const size_t &n_sample, const size_t &n_sample_query,
                                                const size_t &sample_topk) {
            LoadIndex(index_basic_dir, dataset_name, n_sample, n_sample_query, sample_topk);
            n_sample_score_ = 8;
            score_distribution_l_.resize(n_user_ * (n_sample_ - 1) * n_sample_score_);
        }

        const double *SampleData(const int &userID) const {
            return bound_distance_table_.get() + userID * n_sample_;
        }

        void LoopPreprocess(const double *distance_ptr, const int &userID) {

            for (int crankID = 0; crankID < n_sample_ - 1; crankID++) {
                const double IP_ub = bound_distance_table_[n_sample_ * userID + crankID];
                const double IP_lb = bound_distance_table_[n_sample_ * userID + crankID + 1];
                assert(IP_ub >= IP_lb);

                const unsigned int rank_ub = known_rank_idx_l_[crankID];
                const unsigned int rank_lb = known_rank_idx_l_[crankID + 1];
                assert(0 <= rank_ub && rank_ub <= rank_lb && rank_lb < n_data_item_);

                const double distribution_distance = (IP_ub - IP_lb) / (double) (n_sample_score_ + 1);
                const int64_t offset = userID * (n_sample_ - 1) * n_sample_score_ + crankID * n_sample_score_;

                for (int scoreID = 0; scoreID < n_sample_score_; scoreID++) {
                    double sample_IP = IP_ub - (scoreID + 1) * distribution_distance;
                    const uint64_t pred_sample_rank =
                            (rank_lb - rank_ub) * (scoreID + 1) / (n_sample_score_ + 1) + rank_ub;
                    const double *iter_begin = distance_ptr;
                    const double *iter_end = distance_ptr + n_data_item_;

                    const double *lb_ptr = std::lower_bound(iter_begin, iter_end, sample_IP,
                                                            [](const double &arrIP, double queryIP) {
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

        inline void
        CoarseBinarySearch(const double &queryIP, const int &userID,
                           int &bucketID, int &rank_lb, int &rank_ub) const {
            double *search_iter = bound_distance_table_.get() + userID * n_sample_;

            double *iter_begin = search_iter;
            double *iter_end = search_iter + n_sample_;

            double *lb_ptr = std::lower_bound(iter_begin, iter_end, queryIP,
                                              [](const double &arrIP, double queryIP) {
                                                  return arrIP > queryIP;
                                              });
            bucketID = (int) (lb_ptr - iter_begin);


            if (bucketID == 0) {
                rank_ub = 0;
                rank_lb = known_rank_idx_l_[0];
            } else if (bucketID == n_sample_) {
                rank_ub = known_rank_idx_l_[bucketID - 1];
                rank_lb = (int) n_data_item_;
            } else { // lies between the middle
                unsigned int sample_rank_lb = known_rank_idx_l_[bucketID];
                unsigned int sample_rank_ub = known_rank_idx_l_[bucketID - 1];
                rank_lb = (int) sample_rank_lb;
                rank_ub = (int) sample_rank_ub;
            }
        }

        inline void ScoreDistribution(const double &queryIP, const int &userID, const int &bucketID,
                                      int &rank_lb, int &rank_ub) const {

            if (bucketID == 0) {
                rank_ub = 0;
                rank_lb = known_rank_idx_l_[0];
            } else if (bucketID == n_sample_) {
                rank_ub = known_rank_idx_l_[bucketID - 1];
                rank_lb = (int) n_data_item_;
            } else { // lies between the middle
                const double IP_lb = bound_distance_table_[userID * n_sample_ + bucketID];
                const double IP_ub = bound_distance_table_[userID * n_sample_ + bucketID - 1];
                assert(IP_ub >= queryIP && queryIP >= IP_lb);
                const double distribution_distance = (IP_ub - IP_lb) / (double) (n_sample_score_ + 1);
                const int itvID_ub = std::floor((IP_ub - queryIP) / distribution_distance);
                assert(0 <= itvID_ub && itvID_ub <= n_sample_score_);
                const int itvID_lb = itvID_ub + 1;

                unsigned int sample_rank_lb = known_rank_idx_l_[bucketID];
                unsigned int sample_rank_ub = known_rank_idx_l_[bucketID - 1];

                const uint64_t sample_score_offset =
                        userID * (n_sample_ - 1) * n_sample_score_ + (bucketID - 1) * n_sample_score_;
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
                                (sample_rank_lb - sample_rank_ub) * (first_true_idx + 1) / (n_sample_score_ + 1) +
                                sample_rank_ub;
                        rank_ub = (int) pred_sample_rank;

                        assert(IP_ub - (first_true_idx + 1) * distribution_distance >= queryIP);
                    }
                    assert(first_true_idx == -1 ||
                           (0 <= first_true_idx && first_true_idx <= itvID_ub - 1));
                }

                if (itvID_lb != n_sample_score_ + 1) {
                    int first_false_idx = -1;
                    for (int sample_scoreID = itvID_lb - 1; sample_scoreID < n_sample_score_; sample_scoreID++) {
                        if (!score_distribution_l_[sample_score_offset + sample_scoreID]) {
                            first_false_idx = sample_scoreID;
                            break;
                        }
                    }
                    if (first_false_idx != -1) {
                        const uint64_t pred_sample_rank =
                                (sample_rank_lb - sample_rank_ub) * (first_false_idx + 1) / (n_sample_score_ + 1) +
                                sample_rank_ub;
                        rank_lb = (int) pred_sample_rank;

                        assert(IP_ub - (first_false_idx + 1) * distribution_distance <= queryIP);
                    }
                    assert(first_false_idx == -1 ||
                           (itvID_lb - 1 <= first_false_idx && first_false_idx <= n_sample_score_ - 1));
                }

            }
            assert(rank_ub <= rank_lb);

            assert(rank_lb - rank_ub <=
                   std::max(known_rank_idx_l_[n_sample_ - 1], (int) n_data_item_ - known_rank_idx_l_[n_sample_ - 1]));
        }

        void RankBound(const std::vector<double> &queryIP_l,
                       const std::vector<bool> &prune_l, const std::vector<bool> &result_l,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l, std::vector<int> &bucketID_l) const {
            assert(queryIP_l.size() == n_user_);
            assert(prune_l.size() == n_user_);
            assert(result_l.size() == n_user_);
            assert(rank_lb_l.size() == n_user_);
            assert(rank_ub_l.size() == n_user_);
            assert(bucketID_l.size() == n_user_);
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID] || result_l[userID]) {
                    continue;
                }
                int lower_rank = rank_lb_l[userID];
                int upper_rank = rank_ub_l[userID];
                assert(upper_rank <= lower_rank);
                double queryIP = queryIP_l[userID];

                int bucketID = 0;
                CoarseBinarySearch(queryIP, userID,
                                   bucketID, lower_rank, upper_rank);
                bucketID_l[userID] = bucketID;

                rank_lb_l[userID] = lower_rank;
                rank_ub_l[userID] = upper_rank;
            }
        }

        void ScoreDistributionRankBound(const std::vector<double> &queryIP_l,
                                        const std::vector<bool> &prune_l, const std::vector<bool> &result_l,
                                        const std::vector<int> &bucketID_l,
                                        std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l) {
            assert(queryIP_l.size() == n_user_);
            assert(prune_l.size() == n_user_);
            assert(result_l.size() == n_user_);
            assert(rank_lb_l.size() == n_user_);
            assert(rank_ub_l.size() == n_user_);
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID] || result_l[userID]) {
                    continue;
                }
                int lower_rank = rank_lb_l[userID];
                int upper_rank = rank_ub_l[userID];
                assert(upper_rank <= lower_rank);
                double queryIP = queryIP_l[userID];

                int bucketID = bucketID_l[userID];
                ScoreDistribution(queryIP, userID, bucketID, lower_rank, upper_rank);

                rank_lb_l[userID] = lower_rank;
                rank_ub_l[userID] = upper_rank;
            }


        }

        void SaveIndex(const char *index_basic_dir, const char *dataset_name) {
            char index_path[256];
            sprintf(index_path,
                    "%s/memory_index/QueryRankSearchKthRank-%s-n_sample_%ld-n_sample_query_%ld-sample_topk_%ld.index",
                    index_basic_dir, dataset_name, n_sample_, n_sample_query_, sample_topk_);
            std::ofstream out_stream_ = std::ofstream(index_path, std::ios::binary | std::ios::out);
            if (!out_stream_) {
                spdlog::error("error in write result, not found index");
                exit(-1);
            }
            out_stream_.write((char *) &n_sample_, sizeof(size_t));
            out_stream_.write((char *) &n_data_item_, sizeof(size_t));
            out_stream_.write((char *) &n_user_, sizeof(size_t));
            out_stream_.write((char *) &n_sample_query_, sizeof(size_t));
            out_stream_.write((char *) &sample_topk_, sizeof(size_t));

            out_stream_.write((char *) known_rank_idx_l_.get(), (int64_t) (n_sample_ * sizeof(int)));
            out_stream_.write((char *) bound_distance_table_.get(), (int64_t) (n_user_ * n_sample_ * sizeof(double)));

            out_stream_.close();
        }

        void LoadIndex(const char *index_basic_dir, const char *dataset_name,
                       const size_t &n_sample, const size_t &n_sample_query, const size_t &sample_topk) {
            char index_path[256];
            sprintf(index_path,
                    "%s/memory_index/QueryRankSampleScoreDistribution-%s-n_sample_%ld-n_sample_query_%ld-sample_topk_%ld.index",
                    index_basic_dir, dataset_name, n_sample, n_sample_query, sample_topk);
            std::ifstream index_stream = std::ifstream(index_path, std::ios::binary | std::ios::in);
            if (!index_stream) {
                spdlog::error("error in reading index");
                exit(-1);
            }

            index_stream.read((char *) &n_sample_, sizeof(size_t));
            index_stream.read((char *) &n_data_item_, sizeof(size_t));
            index_stream.read((char *) &n_user_, sizeof(size_t));
            index_stream.read((char *) &n_sample_query_, sizeof(size_t));
            index_stream.read((char *) &sample_topk_, sizeof(size_t));
            assert(n_sample_ == n_sample && n_sample_query_ == n_sample_query && sample_topk_ == sample_topk);

            known_rank_idx_l_ = std::make_unique<int[]>(n_sample_);
            index_stream.read((char *) known_rank_idx_l_.get(), (int64_t) (sizeof(int) * n_sample_));

            bound_distance_table_ = std::make_unique<double[]>(n_user_ * n_sample_);
            index_stream.read((char *) bound_distance_table_.get(), (int64_t) (sizeof(double) * n_user_ * n_sample_));

            index_stream.close();
        }

        uint64_t IndexSizeByte() const {
            const uint64_t known_rank_idx_size = sizeof(int) * n_sample_;
            const uint64_t bound_distance_table_size = sizeof(double) * n_user_ * n_sample_;
            const uint64_t score_distribution_size = n_user_ * (n_sample_ - 1) * n_sample_score_ / 8;
            return known_rank_idx_size + bound_distance_table_size + score_distribution_size;
        }

    };
}
#endif //REVERSE_K_RANKS_QUERYRANKSEARCHSCOREDISTRIBUTION_HPP
