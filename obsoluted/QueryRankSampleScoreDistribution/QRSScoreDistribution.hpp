//
// Created by BianZheng on 2022/8/12.
//

#ifndef REVERSE_KRANKS_QRSVALUEDISTRIBUTION_HPP
#define REVERSE_KRANKS_QRSVALUEDISTRIBUTION_HPP

#include "../SampleSearch/QueryRankSearchSearchAllRank.hpp"
#include "struct/DistancePair.hpp"

#include <iostream>
#include <memory>
#include <fstream>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class QRSScoreDistribution {

        size_t n_sample_, n_data_item_, n_user_, n_sample_score_;
        std::unique_ptr<int[]> known_rank_idx_l_; // n_sample_
        std::unique_ptr<double[]> bound_distance_table_; // n_user * n_sample_
        std::vector<bool> score_distribution_l_; // n_user * (n_sample_ - 1) * n_sample_score_
    public:

        inline QRSScoreDistribution() {}

        inline QRSScoreDistribution(const int &n_sample, const int &n_sample_score_distribution,
                                    const int &n_data_item, const int &n_user,
                                    const char *dataset_name,
                                    const int &n_sample_query, const int &sample_topk,
                                    const char *index_size_basic_dir = "..") {
            this->n_sample_ = n_sample;
            this->n_data_item_ = n_data_item;
            this->n_user_ = n_user;
            this->n_sample_score_ = n_sample_score_distribution;
            known_rank_idx_l_ = std::make_unique<int[]>(n_sample_);
            bound_distance_table_ = std::make_unique<double[]>(n_user_ * n_sample_);
            score_distribution_l_.resize(n_user_ * (n_sample_ - 1) * n_sample_score_);
            if (n_sample <= 1 || n_sample >= n_data_item) {
                spdlog::error("n_sample too small or too large, program exit");
                exit(-1);
            }
            assert(n_sample > 1);

            PreprocessSampleRank(dataset_name, n_sample_query, sample_topk, index_size_basic_dir);
            spdlog::info("rank bound: n_sample {}", n_sample_);
        }

        inline QRSScoreDistribution(const char *index_path) {
            LoadIndex(index_path);
        }

        void PreprocessSampleRank(const char *dataset_name, const int &n_sample_query, const int &sample_topk,
                                  const char *index_size_basic_dir) {
            SampleQueryDistributionBelowTopk query_distribution_ins((int) n_data_item_, dataset_name,
                                                                    n_sample_query, sample_topk, index_size_basic_dir);
            const uint64_t n_sample_rank = query_distribution_ins.n_distinct_rank_;

            std::vector<int64_t> optimal_dp(n_sample_rank * n_sample_);
            std::vector<int> position_dp(n_sample_rank * n_sample_);
            for (int sampleID = 0; sampleID < n_sample_; sampleID++) {
                if (sampleID != 0 && sampleID % 100 == 0) {
                    spdlog::info("sampleID {}, n_sample {}, progress {:.3f}",
                                 sampleID, n_sample_, (double) sampleID / (double) n_sample_);
                }
#pragma omp parallel for default(none) shared(n_sample_rank, sampleID, optimal_dp, query_distribution_ins, position_dp)
                for (int rankID = 0; rankID < n_sample_rank; rankID++) {
                    if (sampleID == 0) {
                        optimal_dp[rankID * n_sample_ + sampleID] =
                                query_distribution_ins.GetUnpruneCandidate(0, rankID) +
                                query_distribution_ins.GetUnpruneCandidate(rankID + 1);

                        position_dp[rankID * n_sample_ + sampleID] = rankID;
                    } else {
                        optimal_dp[rankID * n_sample_ + sampleID] =
                                optimal_dp[rankID * n_sample_ + sampleID - 1];
                        position_dp[rankID * n_sample_ + sampleID] =
                                position_dp[rankID * n_sample_ + sampleID - 1];
                        for (int prev_rankID = rankID - 1; prev_rankID >= 0; prev_rankID--) {
                            const int64_t unprune_user_candidate =
                                    -query_distribution_ins.GetUnpruneCandidate(prev_rankID + 1) +
                                    query_distribution_ins.GetUnpruneCandidate(prev_rankID + 1, rankID) +
                                    query_distribution_ins.GetUnpruneCandidate(rankID + 1);
                            assert(unprune_user_candidate <= 0);

                            if (optimal_dp[rankID * n_sample_ + sampleID] >
                                optimal_dp[prev_rankID * n_sample_ + sampleID - 1] + unprune_user_candidate) {

                                optimal_dp[rankID * n_sample_ + sampleID] =
                                        optimal_dp[prev_rankID * n_sample_ + sampleID - 1] + unprune_user_candidate;

                                position_dp[rankID * n_sample_ + sampleID] = prev_rankID;
                            }
                            assert(optimal_dp[rankID * n_sample_ + sampleID] >= 0);
                            assert(position_dp[rankID * n_sample_ + sampleID] >= 0);
                        }
                        assert(optimal_dp[rankID * n_sample_ + sampleID] >= 0);
                        assert(position_dp[rankID * n_sample_ + sampleID] >= 0);
                    }
                }
            }

            uint64_t min_cost = UINT64_MAX;
            unsigned int min_cost_idx = -1;
            for (int rankID = 0; rankID < n_sample_rank; rankID++) {
                const uint64_t tmp_cost = optimal_dp[rankID * n_sample_ + n_sample_ - 1];
                if (tmp_cost < min_cost) {
                    min_cost_idx = rankID;
                    min_cost = tmp_cost;
                }
            }
            assert(min_cost_idx != -1);
            std::vector<int> sample_idx_l(n_sample_);
            for (int sampleID = (int) n_sample_ - 1; sampleID >= 0; sampleID--) {
                sample_idx_l[sampleID] = (int) min_cost_idx;

                known_rank_idx_l_[sampleID] = (int) query_distribution_ins.GetRank(min_cost_idx);
                min_cost_idx = position_dp[min_cost_idx * n_sample_ + sampleID];
            }

            for (int rankID = 0; rankID < n_sample_; rankID++) {
                std::cout << known_rank_idx_l_[rankID] << " ";
            }
            std::cout << std::endl;

            std::vector<int> known_distinct_rank_l(n_sample_);
            known_distinct_rank_l.assign(known_rank_idx_l_.get(), known_rank_idx_l_.get() + n_sample_);
            std::sort(known_distinct_rank_l.begin(), known_distinct_rank_l.end());
            known_distinct_rank_l.erase(std::unique(known_distinct_rank_l.begin(), known_distinct_rank_l.end()),
                                        known_distinct_rank_l.end());

            if (query_distribution_ins.n_distinct_rank_ >= n_sample_) {
                assert(known_distinct_rank_l.size() == n_sample_);
            }
            spdlog::info("n_distinct_rank {}, n_sample {}, n_sample_distinct_rank {}",
                         query_distribution_ins.n_distinct_rank_, n_sample_,
                         known_distinct_rank_l.size());
        }

        void LoopPreprocess(const DistancePair *distance_ptr, const int &userID) {
            for (int crankID = 0; crankID < n_sample_; crankID++) {
                const unsigned int rankID = known_rank_idx_l_[crankID];
                bound_distance_table_[n_sample_ * userID + crankID] = distance_ptr[rankID].dist_;
            }

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
                    const DistancePair *iter_begin = distance_ptr;
                    const DistancePair *iter_end = distance_ptr + n_data_item_;

                    const DistancePair *lb_ptr = std::lower_bound(iter_begin, iter_end, sample_IP,
                                                                  [](const DistancePair &arrIP, double queryIP) {
                                                                      return arrIP.dist_ > queryIP;
                                                                  });
                    const int sample_rank = (int) (lb_ptr - iter_begin);
                    assert(0 <= sample_rank && sample_rank <= rank_lb);
                    assert(rank_ub <= pred_sample_rank && pred_sample_rank <= rank_lb);
                    const bool is_larger = sample_rank >= pred_sample_rank;
                    score_distribution_l_[offset + scoreID] = is_larger;
                }
            }

        }

        void LoopPreprocess(const double *distance_ptr, const int &userID) {
            //warning, not test, may have bug
            for (int crankID = 0; crankID < n_sample_; crankID++) {
                unsigned int rankID = known_rank_idx_l_[crankID];
                bound_distance_table_[n_sample_ * userID + crankID] = distance_ptr[rankID];
            }

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
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l) const {
            for (int userID = 0; userID < n_user_; userID++) {
                int lower_rank = rank_lb_l[userID];
                int upper_rank = rank_ub_l[userID];
                assert(upper_rank <= lower_rank);
                double queryIP = queryIP_l[userID];
                int bucketID = 0;

                CoarseBinarySearch(queryIP, userID,
                                   bucketID, lower_rank, upper_rank);

                ScoreDistribution(queryIP, userID, bucketID, lower_rank, upper_rank);

                rank_lb_l[userID] = lower_rank;
                rank_ub_l[userID] = upper_rank;
            }
        }

        void SaveIndex(const char *index_path) {
            std::ofstream out_stream_ = std::ofstream(index_path, std::ios::binary | std::ios::out);
            if (!out_stream_) {
                spdlog::error("error in write result");
                exit(-1);
            }
            out_stream_.write((char *) &n_sample_, sizeof(size_t));
            out_stream_.write((char *) &n_data_item_, sizeof(size_t));
            out_stream_.write((char *) &n_user_, sizeof(size_t));

            out_stream_.write((char *) known_rank_idx_l_.get(), (int64_t) (n_sample_ * sizeof(int)));
            out_stream_.write((char *) bound_distance_table_.get(), (int64_t) (n_user_ * n_sample_ * sizeof(double)));

            out_stream_.close();
        }

        void LoadIndex(const char *index_path) {
            std::ifstream index_stream = std::ifstream(index_path, std::ios::binary | std::ios::in);
            if (!index_stream) {
                spdlog::error("error in reading index");
                exit(-1);
            }

            index_stream.read((char *) &n_sample_, sizeof(size_t));
            index_stream.read((char *) &n_data_item_, sizeof(size_t));
            index_stream.read((char *) &n_user_, sizeof(size_t));

            known_rank_idx_l_ = std::make_unique<int[]>(n_sample_);
            index_stream.read((char *) known_rank_idx_l_.get(), (int64_t) (sizeof(int) * n_sample_));

            bound_distance_table_ = std::make_unique<double[]>(n_user_ * n_sample_);
            index_stream.read((char *) bound_distance_table_.get(), (int64_t) (sizeof(double) * n_user_ * n_sample_));

            index_stream.close();
        }

    };
}
#endif //REVERSE_KRANKS_QRSVALUEDISTRIBUTION_HPP
