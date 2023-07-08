//
// Created by bianzheng on 2023/3/13.
//

#ifndef REVERSE_KRANKS_OPTIMALSAMPLERANKCOMPUTEALL_HPP
#define REVERSE_KRANKS_OPTIMALSAMPLERANKCOMPUTEALL_HPP

#include <iostream>
#include <memory>
#include <fstream>
#include <spdlog/spdlog.h>
#include <set>
#include <map>
#include <numeric>

namespace ReverseMIPS {
    class QueryDistributionFast {
        uint64_t n_user_, n_data_item_;

        uint64_t n_sample_query_;
        uint64_t sample_topk_;

        std::vector<uint64_t> IO_cost_offset_l_;
        std::vector<int64_t> IO_cost_iter_l_;
        std::vector<int> sample_rank_l_;
    public:

        uint64_t n_sample_rank_;
        std::vector<int64_t> IO_cost_sample_1_;

        inline QueryDistributionFast() = default;

        inline QueryDistributionFast(const uint64_t &n_user, const uint64_t &n_data_item,
                                     const char *dataset_name, const std::string &sample_name,
                                     const size_t &n_sample_query, const size_t &sample_topk,
                                     const char *index_basic_dir) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->n_sample_query_ = n_sample_query;
            this->sample_topk_ = sample_topk;
            if (!(sample_name == "OptimalPart" || sample_name == "OptimalUniform")) {
                spdlog::error("not find sample name, program exit");
                exit(-1);
            }
            QueryDistribution query_distribution_(n_user, n_data_item, dataset_name, sample_name,
                                                  n_sample_query, sample_topk, index_basic_dir);
            this->n_sample_rank_ = query_distribution_.n_sample_rank_;
            sample_rank_l_.resize(n_sample_rank_);
            std::memcpy(sample_rank_l_.data(), query_distribution_.sample_rank_l_.get(), sizeof(int) * n_sample_rank_);

            //compute the IO cost when sampleID == 0
            IO_cost_sample_1_.resize(n_sample_rank_);
#pragma omp parallel for default(none) shared(IO_cost_sample_1_, query_distribution_)
            for (int rankID = 0; rankID < n_sample_rank_; rankID++) {
                IO_cost_sample_1_[rankID] =
                        query_distribution_.GetIOCostBegin(rankID) + query_distribution_.GetIOCostEnd(rankID);
//                printf("IO_cost_sample_1_ %lu\n", IO_cost_sample_1_[rankID]);
            }


            IO_cost_offset_l_.resize(n_sample_rank_ - 1);
            for (int sample_rankID = 0; sample_rankID < n_sample_rank_ - 1; sample_rankID++) {
                if (sample_rankID == 0) {
                    IO_cost_offset_l_[sample_rankID] = 0;
                } else {
                    IO_cost_offset_l_[sample_rankID] = IO_cost_offset_l_[sample_rankID - 1] + sample_rankID;
                }
            }

            const uint64_t IO_cost_cache_size = IO_cost_offset_l_[n_sample_rank_ - 2] + n_sample_rank_ - 1;
            IO_cost_iter_l_.resize(IO_cost_cache_size);
            IO_cost_iter_l_.assign(IO_cost_cache_size, -1);
            assert(IO_cost_cache_size == n_sample_rank_ * (n_sample_rank_ - 1) / 2);

#pragma omp parallel for default(none) shared(query_distribution_, IO_cost_iter_l_)
            for (int rankID = 1; rankID < n_sample_rank_; rankID++) {
                const uint64_t start_offset = IO_cost_offset_l_[rankID - 1];
                for (int prev_rankID = 0; prev_rankID < rankID; prev_rankID++) {
                    IO_cost_iter_l_[start_offset + prev_rankID] = query_distribution_.GetTransitIOCost(prev_rankID,
                                                                                                       rankID);
//                    printf("IO_cost_iter_l_ %ld, prev_rankID %d, rankID %d, total_offset %lu\n",
//                           IO_cost_iter_l_[start_offset + prev_rankID], prev_rankID, rankID,
//                           start_offset + prev_rankID);
                }
            }

        }

        int64_t GetTransitIOCost(const int &rankID_ub, const int &rankID_lb) {
            assert(0 <= rankID_ub && rankID_ub < rankID_lb && rankID_lb < n_sample_rank_);
            const uint64_t start_offset = IO_cost_offset_l_[rankID_lb - 1];
            return IO_cost_iter_l_[start_offset + rankID_ub];
        }

        unsigned int GetRank(const unsigned int &rankID) {
            assert(0 <= rankID && rankID < n_sample_rank_);
            return sample_rank_l_[rankID];
        }

    };

    void FindOptimalSampleFast(const size_t &n_user, const size_t &n_data_item, const size_t &n_sample,
                               std::vector<int> &sample_rank_l,
                               const size_t &n_sample_query, const size_t &sample_topk,
                               const char *dataset_name, const char *sample_name,
                               const char *index_basic_dir,
                               size_t &train_io_cost) {
        assert(sample_rank_l.size() == n_sample);

        QueryDistributionFast query_distribution_ins(n_user, n_data_item, dataset_name, sample_name,
                                                     n_sample_query, sample_topk, index_basic_dir);
        const uint64_t n_sample_rank = query_distribution_ins.n_sample_rank_;

        if (n_sample_rank <= n_sample) {
            spdlog::info("the sampled rank is smaller than number of samples, take all samples");

            std::set<int> rank_set;
            for (int sample_rankID = 0; sample_rankID < n_sample_rank; sample_rankID++) {
                rank_set.insert((int) query_distribution_ins.GetRank(sample_rankID));
            }

            const int n_remain_sample = n_sample - (int) rank_set.size();
            const int last_sample =
                    n_sample_query < 500 ? n_data_item : (int) query_distribution_ins.GetRank(n_sample_rank - 1);
//            const int last_sample = n_data_item;
            const double delta = last_sample * 1.0 / n_remain_sample;

            double insert_rank = 0;
            while (rank_set.size() < n_sample && (int) insert_rank < last_sample) {
                if (rank_set.find((int) insert_rank) == rank_set.end()) {
                    rank_set.insert((int) insert_rank);
                    insert_rank += delta;
                } else {
                    insert_rank += 1;
                }
            }
            int insert_rank_int = 0;
            while (rank_set.size() < n_sample && (int) insert_rank_int < n_data_item) {
                if (rank_set.find((int) insert_rank_int) == rank_set.end()) {
                    rank_set.insert((int) insert_rank_int);
                }
                insert_rank_int++;
            }

//            int insert_rank = 0;
//            while (rank_set.size() < n_sample && insert_rank < n_data_item) {
//                if (rank_set.find(insert_rank) == rank_set.end()) {
//                    rank_set.insert(insert_rank);
//                }
//                insert_rank++;
//            }
            assert(rank_set.size() == n_sample);
            std::vector<int> tmp_sample_rank_l;
            tmp_sample_rank_l.reserve(n_sample);
            for (const int &rank: rank_set) {
                tmp_sample_rank_l.emplace_back(rank);
            }
            std::sort(tmp_sample_rank_l.begin(), tmp_sample_rank_l.end());
            for (int sampleID = 0; sampleID < n_sample; sampleID++) {
                sample_rank_l[sampleID] = tmp_sample_rank_l[sampleID];
            }

        } else {
            std::vector<int64_t> optimal_dp(n_sample * n_sample_rank);
            std::vector<int> position_dp(n_sample * n_sample_rank);

            {//sampleID = 0
                const int sampleID = 0;
                std::memcpy(optimal_dp.data() + sampleID * n_sample_rank,
                            query_distribution_ins.IO_cost_sample_1_.data(), sizeof(int64_t) * n_sample_rank);
                for (int rankID = 0; rankID < n_sample_rank; rankID++) {
                    position_dp[sampleID * n_sample_rank + rankID] = rankID;
                }
                std::iota(position_dp.data() + sampleID * n_sample_rank,
                          position_dp.data() + sampleID * n_sample_rank + n_sample_rank, 0);
            }

            for (int sampleID = 1; sampleID < n_sample; sampleID++) {
                if (sampleID % 300 == 0) {
                    spdlog::info("sampleID {}, n_sample {}, progress {:.3f}",
                                 sampleID, n_sample, (double) sampleID / (double) n_sample);
                }
#pragma omp parallel for default(none) shared(n_sample_rank, sampleID, optimal_dp, position_dp, n_sample, query_distribution_ins, n_user, n_sample_query, n_data_item)
                for (int rankID = 0; rankID < n_sample_rank; rankID++) {
                    assert(sampleID != 0);
                    optimal_dp[sampleID * n_sample_rank + rankID] =
                            optimal_dp[(sampleID - 1) * n_sample_rank + rankID];
                    position_dp[sampleID * n_sample_rank + rankID] =
                            position_dp[(sampleID - 1) * n_sample_rank + rankID];

                    for (int prev_rankID = 0; prev_rankID <= rankID - 1; prev_rankID++) {
                        assert(rankID != 0);
                        const int64_t unprune_iocost =
                                query_distribution_ins.GetTransitIOCost(prev_rankID, rankID);
                        assert(unprune_iocost <= 0);

                        if (optimal_dp[sampleID * n_sample_rank + rankID] >
                            optimal_dp[(sampleID - 1) * n_sample_rank + prev_rankID] + unprune_iocost) {

                            optimal_dp[sampleID * n_sample_rank + rankID] =
                                    optimal_dp[(sampleID - 1) * n_sample_rank + prev_rankID] + unprune_iocost;

                            position_dp[sampleID * n_sample_rank + rankID] = prev_rankID;
                        }
                        assert(optimal_dp[sampleID * n_sample_rank + rankID] >= 0);
                        assert(position_dp[sampleID * n_sample_rank + rankID] >= 0);
                    }
                    assert(optimal_dp[sampleID * n_sample_rank + rankID] >= 0);
                    assert(position_dp[sampleID * n_sample_rank + rankID] >= 0);
                }
            }

            uint64_t min_cost = UINT64_MAX;
            unsigned int min_cost_idx = -1;
            for (int rankID = 0; rankID < n_sample_rank; rankID++) {
                const uint64_t tmp_cost = optimal_dp[(n_sample - 1) * n_sample_rank + rankID];
                if (tmp_cost < min_cost) {
                    min_cost_idx = rankID;
                    min_cost = tmp_cost;
                }
            }
            spdlog::info("min IO cost {}, idx {}", min_cost, min_cost_idx);
            train_io_cost = min_cost;
            assert(min_cost_idx != -1);
            for (int sampleID = (int) n_sample - 1; sampleID >= 0; sampleID--) {
                sample_rank_l[sampleID] = (int) query_distribution_ins.GetRank(min_cost_idx);
                min_cost_idx = position_dp[sampleID * n_sample_rank + min_cost_idx];
            }

            bool has_redundant = false;
            std::set<int> rank_set;
            for (int sampleID = 0; sampleID < n_sample; sampleID++) {
                if (rank_set.find((int) sample_rank_l[sampleID]) == rank_set.end()) {
                    rank_set.insert(sample_rank_l[sampleID]);
                } else {
                    has_redundant = true;
                    break;
                }
            }
            rank_set.clear();
            for (int sampleID = 0; sampleID < n_sample; sampleID++) {
                rank_set.insert(sample_rank_l[sampleID]);
            }

            if (has_redundant) {
                const int n_remain_sample = n_sample - (int) rank_set.size();
                const int last_sample = (int) query_distribution_ins.GetRank(n_sample_rank - 1);
                const double delta = last_sample * 1.0 / n_remain_sample;

                double insert_rank = 0;
                while (rank_set.size() < n_sample && (int) insert_rank < last_sample) {
                    if (rank_set.find((int) insert_rank) == rank_set.end()) {
                        rank_set.insert((int) insert_rank);
                        insert_rank += delta;
                    } else {
                        insert_rank += 1;
                    }
                }
                int insert_rank_int = 0;
                while (rank_set.size() < n_sample && (int) insert_rank_int < n_data_item) {
                    if (rank_set.find((int) insert_rank_int) == rank_set.end()) {
                        rank_set.insert((int) insert_rank_int);
                    }
                    insert_rank_int++;
                }
                if (rank_set.size() != n_sample) {
                    spdlog::error("have bug, the size of rank set is less than n_sample");
                    exit(-1);
                }
                assert(rank_set.size() == n_sample);

                std::vector<int> tmp_sample_rank_l;
                tmp_sample_rank_l.reserve(n_sample);
                for (const int &rank: rank_set) {
                    tmp_sample_rank_l.emplace_back(rank);
                }
                std::sort(tmp_sample_rank_l.begin(), tmp_sample_rank_l.end());
                for (int sampleID = 0; sampleID < n_sample; sampleID++) {
                    sample_rank_l[sampleID] = tmp_sample_rank_l[sampleID];
                }
            }


        }

        std::cout << "first 50 rank: ";
        const int end_rankID = std::min((int) n_sample, 50);
        for (int rankID = 0; rankID < end_rankID; rankID++) {
            std::cout << sample_rank_l[rankID] << " ";
        }
        std::cout << std::endl;
        std::cout << "last 50 rank: ";
        const int start_rankID = std::max(0, (int) n_sample - 50);
        for (int rankID = start_rankID; rankID < n_sample; rankID++) {
            std::cout << sample_rank_l[rankID] << " ";
        }
        std::cout << std::endl;

//        for (int rankID = 0; rankID < n_sample; rankID++) {
//            std::cout << sample_rank_l[rankID] << " ";
//        }
//        std::cout << std::endl;

        spdlog::info("optimal sample: n_sample {}", n_sample);
    }

}
#endif //REVERSE_KRANKS_OPTIMALSAMPLERANKCOMPUTEALL_HPP
