//
// Created by BianZheng on 2022/11/3.
//

#ifndef REVERSE_K_RANKS_OPTIMALALLRANK_HPP
#define REVERSE_K_RANKS_OPTIMALALLRANK_HPP

#include <iostream>
#include <memory>
#include <fstream>
#include <spdlog/spdlog.h>
#include <set>
#include <map>
#include <numeric>

namespace ReverseMIPS {
    class QueryDistribution {
        uint64_t n_user_, n_data_item_;
        std::unique_ptr<int[]> sort_kth_rank_l_; // n_sample_query
        std::unique_ptr<int[]> accu_user_rank_l_; // n_sample_query * (n_data_item_ + 1)
        // stores the number of user candidates between [1, rank]
        std::map<int, int> rankID_this_last_pos_m_;
        std::map<int, int> rankID_next_first_pos_m_;

        uint64_t n_sample_query_;
        uint64_t sample_topk_;
    public:

        uint64_t n_sample_rank_;
        std::unique_ptr<int[]> sample_rank_l_; // n_sample_rank_

        inline QueryDistribution() = default;

        inline QueryDistribution(const uint64_t &n_user, const uint64_t &n_data_item,
                                 const char *dataset_name, const std::string &sample_name,
                                 const size_t &n_sample_query, const size_t &sample_topk,
                                 const char *index_basic_dir) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->n_sample_query_ = n_sample_query;
            this->sample_topk_ = sample_topk;
            this->sort_kth_rank_l_ = std::make_unique<int[]>(n_sample_query_);
            this->accu_user_rank_l_ = std::make_unique<int[]>(n_sample_query_ * (n_data_item_ + 1));

            //init sort_kth_rank_l_, accu_user_rank_l_
            ReadQueryDistribution(index_basic_dir, dataset_name);

            std::vector<int> sample_rank_l;
            if (sample_name == "OptimalPart") {
                std::set<int> rank_s;
//            this->sample_rank_l_ = std::make_unique<int[]>(n_data_item);

                for (int sampleID = 0; sampleID < n_sample_query_; sampleID++) {
                    const int rank = sort_kth_rank_l_[sampleID];
                    rank_s.insert(rank);
                    if (rank != 0) {
                        rank_s.insert(rank - 1);
                    }
                }
                sample_rank_l.reserve(rank_s.size());
                for (const int &rank: rank_s) {
                    sample_rank_l.emplace_back(rank);
                }
                assert(sample_rank_l.size() == rank_s.size());

            } else if (sample_name == "OptimalAll") {
                sample_rank_l.resize(n_data_item_);
                std::iota(sample_rank_l.begin(), sample_rank_l.end(), 0);

            } else if (sample_name == "OptimalUniform") {

                const int end_sample_rank = (int) n_data_item - 1;
                const double delta = (end_sample_rank - 0) * 1.0 / n_sample_query_;
                sample_rank_l.resize(n_sample_query_);
                for (int sampleID = 0; sampleID < n_sample_query_; sampleID++) {
                    sample_rank_l[sampleID] = std::floor(sampleID * delta);
                }
                std::sort(sample_rank_l.data(), sample_rank_l.data() + n_sample_query_);

            } else {
                spdlog::error("not find sample name, program exit");
                exit(-1);
            }

            std::sort(sample_rank_l.begin(), sample_rank_l.end());
            this->n_sample_rank_ = sample_rank_l.size();
            if (sample_name == "OptimalAll") {
                assert(this->n_sample_rank_ == n_data_item_);
            }
            sample_rank_l_ = std::make_unique<int[]>(n_sample_rank_);
            std::memcpy(sample_rank_l_.get(), sample_rank_l.data(), n_sample_rank_ * sizeof(int));

            BuildRankIDMap();

            spdlog::info("n_sample_query {}, sample_topk {}, n_sample_rank {}",
                         n_sample_query_, sample_topk_, n_sample_rank_);

            /*
             * {
                std::vector<int> sample_queryID_l(n_sample_query_);
                assert(sample_queryID_l.size() == n_sample_query_);
                char resPath[256];
                std::sprintf(resPath, "%s/index/query_distribution/%s-sample-itemID-n_sample_query_%ld-sample_topk_%ld.txt",
                             index_basic_dir, dataset_name, n_sample_query_, sample_topk_);

                std::ifstream in_stream = std::ifstream(resPath, std::ios::binary | std::ios::in);
                if (!in_stream.is_open()) {
                    spdlog::error("error in open file");
                    exit(-1);
                }

                in_stream.read((char *) sample_queryID_l.data(), sizeof(int) * n_sample_query_);
                printf("sample queryID l\n");
                for (int sample_queryID = 0; sample_queryID < n_sample_query_; sample_queryID++) {
                    printf("%d ", sample_queryID_l[sample_queryID]);
                }
                printf("\n");
            }
             */

//            const int n_print = 10; //n_print <= n_sample_query_
//            for (int sample_queryID = 0; sample_queryID < n_print; sample_queryID++) {
//                printf("%4d ", sample_queryID);
//            }
//            printf("\n------------------\n");
//            for (int sample_queryID = 0; sample_queryID < n_print; sample_queryID++) {
//                printf("%4d ", sort_kth_rank_l_[sample_queryID]);
//            }
//            printf("\n------------------\n");
//            for (int sample_queryID = 0; sample_queryID < n_print; sample_queryID++) {
//                for (int sample_queryID2 = 0; sample_queryID2 < n_print; sample_queryID2++) {
//                    int print_thing;
//                    if (sample_queryID2 == 0) {
//                    print_thing = accu_user_rank_l_[sample_queryID * n_sample_query_ + sample_queryID2];
//                    } else {
//                        print_thing = accu_user_rank_l_[sample_queryID * n_sample_query_ + sample_queryID2] -
//                                      accu_user_rank_l_[sample_queryID * n_sample_query_ + sample_queryID2 - 1];
//                    }
//                    if (sample_queryID > sample_queryID2) {
//                        assert(accu_user_rank_l_[sample_queryID * n_sample_query_ + sample_queryID2] == 0);
//                    }
//
//                    if (sample_queryID == sample_queryID2) {
//                        printf("\033[0;31m"); //Set the text to the color red
//                        printf("%4d ", print_thing);
//                        printf("\033[0m"); //Resets the text to default color
//
//                    } else {
//                        printf("%4d ", print_thing);
//                    }
//                }
//                printf("\n");
//            }

        }

        void ReadQueryDistribution(const char *index_basic_dir, const char *dataset_name) {

            {
                char sort_kth_rank_path[512];
                sprintf(sort_kth_rank_path,
                        "%s/query_distribution/%s-n_sample_item_%ld-sample_topk_%ld/sort_kth_rank_l.index",
                        index_basic_dir, dataset_name, n_sample_query_, sample_topk_);

                std::ifstream sort_kth_rank_stream = std::ifstream(sort_kth_rank_path, std::ios::binary | std::ios::in);
                if (!sort_kth_rank_stream) {
                    spdlog::error("error in reading index");
                    exit(-1);
                }

                sort_kth_rank_stream.seekg(0, std::ios::end);
                std::ios::pos_type ss = sort_kth_rank_stream.tellg();
                auto fsize = (size_t) ss;
                assert(fsize == sizeof(int) * n_sample_query_);
                sort_kth_rank_stream.seekg(0, std::ios::beg);

                sort_kth_rank_stream.read((char *) sort_kth_rank_l_.get(), sizeof(int) * n_sample_query_);
                sort_kth_rank_stream.close();
            }

            {
                char accu_user_rank_path[512];
                sprintf(accu_user_rank_path,
                        "%s/query_distribution/%s-n_sample_item_%ld-sample_topk_%ld/accu_n_user_rank_l.index",
                        index_basic_dir, dataset_name, n_sample_query_, sample_topk_);

                std::ifstream accu_user_rank_stream = std::ifstream(accu_user_rank_path,
                                                                    std::ios::binary | std::ios::in);
                if (!accu_user_rank_stream) {
                    spdlog::error("error in reading index");
                    exit(-1);
                }

                accu_user_rank_stream.seekg(0, std::ios::end);
                std::ios::pos_type ss = accu_user_rank_stream.tellg();
                auto fsize = (size_t) ss;
                assert(fsize == sizeof(int) * n_sample_query_ * (n_data_item_ + 1));
                accu_user_rank_stream.seekg(0, std::ios::beg);

                accu_user_rank_stream.read((char *) accu_user_rank_l_.get(),
                                           sizeof(int) * n_sample_query_ * (n_data_item_ + 1));
                accu_user_rank_stream.close();
            }
        }

        void BuildRankIDMap() {
            for (int rankID = 0; rankID < n_sample_rank_; rankID++) {
                const int rank_this = sample_rank_l_[rankID];
                const int rank_next = sample_rank_l_[rankID] + 1;
                int *rank_this_ptr = std::lower_bound(sort_kth_rank_l_.get(),
                                                      sort_kth_rank_l_.get() + n_sample_query_,
                                                      rank_this,
                                                      [](const int &info, const int &value) {
                                                          return info <=
                                                                 value; //first position that is > than the value
                                                      });

                int *rank_next_ptr = std::lower_bound(sort_kth_rank_l_.get(),
                                                      sort_kth_rank_l_.get() + n_sample_query_,
                                                      rank_next,
                                                      [](const int &info, const int &value) {
                                                          return info < value; // first position that is == to the value
                                                      });
                int queryID_rank_this = (int) (rank_this_ptr - sort_kth_rank_l_.get());
                int queryID_rank_next = (int) (rank_next_ptr - sort_kth_rank_l_.get());

                assert(0 <= queryID_rank_this & queryID_rank_this <= n_sample_query_);

                rankID_this_last_pos_m_[rankID] = queryID_rank_this;
                rankID_next_first_pos_m_[rankID] = queryID_rank_next;
            }
        }

        int64_t GetTransitIOCostExceptPublicTerm(const int &rankID_ub, const int &rankID_lb) {
            int64_t first = GetIOCostEnd(rankID_ub);
            int64_t second = GetIOCostBetween(rankID_ub, rankID_lb);
            return -first + second;
        }

        // it is Upsilon([rank_ub + 1, rank_lb])
        int64_t GetIOCostBetween(const int &rankID_ub, const int &rankID_lb) {
            assert(0 <= rankID_ub && rankID_ub < rankID_lb && rankID_lb < n_sample_rank_);
            const int rank_ub = sample_rank_l_[rankID_ub] + 1;
            const int rank_lb = sample_rank_l_[rankID_lb];
            assert(1 <= rank_ub && rank_ub <= rank_lb && rank_lb <= n_data_item_);

            const int &queryID_begin = rankID_next_first_pos_m_[rankID_ub];
            const int &queryID_end = rankID_this_last_pos_m_[rankID_lb];
            assert(0 <= queryID_begin && queryID_begin <= n_sample_query_);
            assert(0 <= queryID_end && queryID_end <= n_sample_query_);
            if (queryID_begin >= n_sample_query_) {
                return 0;
            }
            if (queryID_begin > queryID_end) {
                assert(rank_ub >= rank_lb);
                return 0;
            }
            assert(0 <= queryID_begin && queryID_begin <= queryID_end && queryID_end <= n_sample_query_);

            int64_t n_unprune_user = 0;
            for (int queryID = queryID_begin; queryID < queryID_end; queryID++) {

                int64_t n_user_candidate = accu_user_rank_l_[queryID * (n_data_item_ + 1) + rank_lb] -
                                           accu_user_rank_l_[queryID * (n_data_item_ + 1) + rank_ub - 1];
                assert(n_user_candidate >= 0);
                n_unprune_user += n_user_candidate;
            }
            assert(sample_rank_l_[rankID_lb] - sample_rank_l_[rankID_ub] > 0);
//            n_unprune_user *= (sample_rank_l_[rankID_lb] - sample_rank_l_[rankID_ub]);

            return n_unprune_user;
        }

        // it is Upsilon([rank+1, n_data_item])
        int64_t GetIOCostEnd(const int &rankID) {
            assert(rankID < n_sample_rank_);
            const int rank = sample_rank_l_[rankID] + 1;
            const int queryID_start = rankID_next_first_pos_m_[rankID];

            assert(0 <= queryID_start & queryID_start <= n_sample_query_);
            assert(1 <= rank && rank <= n_data_item_);
            if (queryID_start >= n_sample_query_) {
                return 0;
            }

            int64_t n_unprune_user = 0;
            for (int queryID = queryID_start; queryID < n_sample_query_; queryID++) {
                int64_t n_user_candidate = (int64_t) n_user_ -
                                           accu_user_rank_l_[queryID * (n_data_item_ + 1) + rank - 1];
                assert(accu_user_rank_l_[queryID * (n_data_item_ + 1) + n_data_item_] == n_user_);
                assert(n_user_candidate >= 0);
                n_unprune_user += n_user_candidate;
            }
            assert(n_data_item_ - sample_rank_l_[rankID] > 0);
//            n_unprune_user *= ((int64_t) n_data_item_ - sample_rank_l_[rankID]);

            return n_unprune_user;
        }

        // Upsilon([0, rank])
        int64_t GetIOCostBegin(const int &rankID) {
            assert(rankID < n_sample_rank_);
            const int rank = sample_rank_l_[rankID];
            const int queryID_end = rankID_this_last_pos_m_[rankID];
            assert(0 <= queryID_end && queryID_end <= n_sample_query_);

            int64_t n_unprune_user = 0;
            for (int queryID = 0; queryID < queryID_end; queryID++) {
                int64_t n_user_candidate = accu_user_rank_l_[queryID * (n_data_item_ + 1) + rank];
                assert(n_user_candidate >= 0);
                n_unprune_user += n_user_candidate;
            }
            assert(rank + 1 > 0);
//            n_unprune_user *= (rank + 1);

            return n_unprune_user;
        }

        int64_t GetTransitIOCost(const int &rankID_ub, const int rankID_lb) {
            assert(0 <= rankID_ub && rankID_ub < rankID_lb && rankID_lb < n_sample_rank_);
            int64_t first = GetIOCostEnd(rankID_ub);
            int64_t second = GetIOCostBetween(rankID_ub, rankID_lb);
            int64_t public_term = GetIOCostEnd(rankID_lb);
            return -first + second + public_term;
        }

        unsigned int GetRank(const unsigned int &rankID) {
            assert(0 <= rankID && rankID < n_sample_rank_);
            return sample_rank_l_[rankID];
        }

    };

    void FindOptimalSample(const size_t &n_user, const size_t &n_data_item, const size_t &n_sample,
                           std::vector<int> &sample_rank_l,
                           const size_t &n_sample_query, const size_t &sample_topk,
                           const char *dataset_name, const char *sample_name,
                           const char *index_basic_dir,
                           size_t &train_io_cost) {
        assert(sample_rank_l.size() == n_sample);

        QueryDistribution query_distribution_ins(n_user, n_data_item, dataset_name, sample_name,
                                                 n_sample_query, sample_topk, index_basic_dir);
        const uint64_t n_sample_rank = query_distribution_ins.n_sample_rank_;

        if (n_sample_rank <= n_sample) {
            spdlog::info("the sampled rank is smaller than number of samples, take all samples");

            std::set<int> rank_set;
            for (int sample_rankID = 0; sample_rankID < n_sample_rank; sample_rankID++) {
                rank_set.insert((int) query_distribution_ins.GetRank(sample_rankID));
            }
            int insert_rank = 0;
            while (rank_set.size() < n_sample && insert_rank < n_data_item) {
                if (rank_set.find(insert_rank) == rank_set.end()) {
                    rank_set.insert(insert_rank);
                }
                insert_rank++;
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

        } else {
            std::vector<int64_t> optimal_dp(n_sample_rank * n_sample);
            std::vector<int> position_dp(n_sample_rank * n_sample);

            {//sampleID = 0
                const int sampleID = 0;
                for (int rankID = 0; rankID < n_sample_rank; rankID++) {
                    optimal_dp[rankID * n_sample + sampleID] =
                            query_distribution_ins.GetIOCostBegin(rankID) +
                            query_distribution_ins.GetIOCostEnd(rankID);

                    position_dp[rankID * n_sample + sampleID] = rankID;
                }
            }

            for (int sampleID = 1; sampleID < n_sample; sampleID++) {
                if (sampleID % 10 == 0) {
                    spdlog::info("sampleID {}, n_sample {}, progress {:.3f}",
                                 sampleID, n_sample, (double) sampleID / (double) n_sample);
                }
#pragma omp parallel for default(none) shared(n_sample_rank, sampleID, optimal_dp, position_dp, n_sample, query_distribution_ins, n_user, n_sample_query, n_data_item)
                for (int rankID = 0; rankID < n_sample_rank; rankID++) {
                    assert(sampleID != 0);
                    optimal_dp[rankID * n_sample + sampleID] =
                            optimal_dp[rankID * n_sample + sampleID - 1];
                    position_dp[rankID * n_sample + sampleID] =
                            position_dp[rankID * n_sample + sampleID - 1];

                    const int64_t iocost_public_term = query_distribution_ins.GetIOCostEnd(rankID);
                    int64_t prev_iocost = -(int64_t) n_user * n_sample_query * (n_data_item + 1); // negative inf

                    for (int prev_rankID = 0; prev_rankID <= rankID - 1; prev_rankID++) {
                        if (optimal_dp[rankID * n_sample + sampleID] <
                            optimal_dp[prev_rankID * n_sample + sampleID - 1] + prev_iocost) {
                            continue;
                        }

                        const int64_t unprune_iocost =
                                query_distribution_ins.GetTransitIOCostExceptPublicTerm(
                                        prev_rankID, rankID) + iocost_public_term;
                        assert(unprune_iocost <= 0);

                        if (optimal_dp[rankID * n_sample + sampleID] >
                            optimal_dp[prev_rankID * n_sample + sampleID - 1] + unprune_iocost) {

                            optimal_dp[rankID * n_sample + sampleID] =
                                    optimal_dp[prev_rankID * n_sample + sampleID - 1] + unprune_iocost;

                            position_dp[rankID * n_sample + sampleID] = prev_rankID;

                            prev_iocost = unprune_iocost;
                        }
                        assert(optimal_dp[rankID * n_sample + sampleID] >= 0);
                        assert(position_dp[rankID * n_sample + sampleID] >= 0);
                    }
//                        if (rankID - 1 >= 0 && sampleID < rankID) {
//                            if (optimal_dp[rankID * n_sample + sampleID] >=
//                                optimal_dp[rankID * n_sample + sampleID - 1]) {
//                                printf("this sample %lu, prev sample %lu, sampleID %d, rankID %d, position_dp %d, rank %d\n",
//                                       optimal_dp[rankID * n_sample + sampleID],
//                                       optimal_dp[rankID * n_sample + sampleID - 1],
//                                       sampleID, rankID,
//                                       position_dp[rankID * n_sample + sampleID],
//                                       query_distribution_ins.GetRank(rankID));
//                            }
//
//                            assert(optimal_dp[rankID * n_sample + sampleID] <
//                                   optimal_dp[rankID * n_sample + sampleID - 1]);
//                        }
                    assert(optimal_dp[rankID * n_sample + sampleID] >= 0);
                    assert(position_dp[rankID * n_sample + sampleID] >= 0);
                }
            }

            uint64_t min_cost = UINT64_MAX;
            unsigned int min_cost_idx = -1;
            for (int rankID = 0; rankID < n_sample_rank; rankID++) {
                const uint64_t tmp_cost = optimal_dp[rankID * n_sample + n_sample - 1];
                if (tmp_cost < min_cost) {
                    min_cost_idx = rankID;
                    min_cost = tmp_cost;
                }
            }
            spdlog::info("min IO cost {}, idx {}", min_cost, min_cost_idx);
            train_io_cost = min_cost;
            assert(min_cost_idx != -1);
            std::vector<int> sample_idx_l(n_sample);
            for (int sampleID = (int) n_sample - 1; sampleID >= 0; sampleID--) {
                sample_idx_l[sampleID] = (int) min_cost_idx;

                sample_rank_l[sampleID] = (int) query_distribution_ins.GetRank(min_cost_idx);
                min_cost_idx = position_dp[min_cost_idx * n_sample + sampleID];
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
#endif //REVERSE_K_RANKS_OPTIMALALLRANK_HPP
