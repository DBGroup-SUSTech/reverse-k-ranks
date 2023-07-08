//
// Created by BianZheng on 2022/11/7.
//

#ifndef REVERSE_K_RANKS_OPTIMALBRUTEFORCE_HPP
#define REVERSE_K_RANKS_OPTIMALBRUTEFORCE_HPP

#include <vector>

namespace ReverseMIPS {
    class QueryDistributionBruteForce {
        uint64_t n_user_, n_data_item_;
        std::unique_ptr<int[]> sort_kth_rank_l_; // n_sample_query
        std::unique_ptr<int[]> accu_user_rank_l_; // n_sample_query * (n_data_item_ + 1)
        // stores the number of user candidates between [1, rank]

        uint64_t n_sample_query_;
        uint64_t sample_topk_;
    public:

        inline QueryDistributionBruteForce() = default;

        inline QueryDistributionBruteForce(const uint64_t &n_user, const uint64_t &n_data_item,
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

            spdlog::info("n_sample_query {}, sample_topk {}",
                         n_sample_query_, sample_topk_);

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

        void ComputeIOCost(const std::vector<int> &sample_rank_l, const int &n_sample, size_t &io_cost) {

            io_cost = 0;
            assert(sample_rank_l.size() == n_sample);
            for (int sampleID = 0; sampleID < n_sample_query_; sampleID++) {

                const int kth_rank = sort_kth_rank_l_[sampleID];
                const int *itvID_ptr = std::lower_bound(sample_rank_l.data(),
                                                        sample_rank_l.data() + n_sample,
                                                        kth_rank,
                                                        [](const int &info, const int &value) {
                                                            return info <
                                                                   value; //first position that is == the value
                                                        });
                const int itvID = itvID_ptr - sample_rank_l.data();
                if (itvID == n_sample) {
                    const int rank_ub = sample_rank_l[n_sample - 1];
                    assert(rank_ub > 0);
                    const size_t tmp_io_cost = n_user_ -
                                               accu_user_rank_l_[sampleID * (n_data_item_ + 1) + rank_ub - 1];
                    io_cost += tmp_io_cost;
                    assert(tmp_io_cost >= 0);

                } else if (itvID == 0) {
                    const int rank_lb = sample_rank_l[0];
                    const size_t tmp_io_cost = accu_user_rank_l_[sampleID * (n_data_item_ + 1) + rank_lb];
                    io_cost += tmp_io_cost;
                    assert(tmp_io_cost >= 0);

                } else {
                    const int rank_lb = sample_rank_l[itvID];
                    const int rank_ub = sample_rank_l[itvID - 1];

                    const int lower_user =
                            rank_ub - 1 < 0 ? 0 : accu_user_rank_l_[sampleID * (n_data_item_ + 1) + rank_ub - 1];
                    const size_t tmp_io_cost = accu_user_rank_l_[sampleID * (n_data_item_ + 1) + rank_lb] -
                                               lower_user;
                    io_cost += tmp_io_cost;
                    assert(tmp_io_cost >= 0);

                }

            }

        }

    };

    struct Combinations {
        typedef std::vector<int> combination_t;

        // initialize status
        Combinations(int N, int R) :
                completed(N < 1 || R > N),
                generated(0),
                N(N), R(R) {
            for (int c = 0; c < R; ++c)
                curr.push_back(c);
        }

        // true while there are more solutions
        bool completed;

        // count how many generated
        size_t generated;

        // get current and compute next combination
        combination_t next() {
            combination_t ret = curr;

            // find what to increment
            completed = true;
            for (int i = R - 1; i >= 0; --i)
                if (curr[i] < N - R + i) {
                    int j = curr[i] + 1;
                    while (i <= R - 1)
                        curr[i++] = j++;
                    completed = false;
                    ++generated;
                    break;
                }

            return ret;
        }

    private:

        int N, R;
        combination_t curr;
    };


    void FindOptimalSampleBruteForce(const size_t &n_user, const size_t &n_data_item, const size_t &n_sample,
                                     std::vector<int> &sample_rank_l,
                                     const size_t &n_sample_query, const size_t &sample_topk,
                                     const char *dataset_name, const char *sample_name,
                                     const char *index_basic_dir,
                                     size_t &train_io_cost) {

        QueryDistributionBruteForce query_distribution_ins(n_user, n_data_item, dataset_name, sample_name,
                                                           n_sample_query, sample_topk, index_basic_dir);

        size_t min_io_cost = SIZE_MAX;
        std::vector<int> min_sample_rank_l(n_sample, -1);

        Combinations comb((int) n_data_item, (int) n_sample);
        while (!comb.completed) {
            const std::vector<int> &tmp_sample_rank_l = comb.next();
            size_t tmp_io_cost = 0;
            query_distribution_ins.ComputeIOCost(tmp_sample_rank_l, (int) n_sample, tmp_io_cost);
            if (min_io_cost > tmp_io_cost) {
                min_io_cost = tmp_io_cost;
                min_sample_rank_l.assign(tmp_sample_rank_l.begin(), tmp_sample_rank_l.end());
            }

        }

        spdlog::info("min_io_cost {}", min_io_cost);

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

//        for (int sampleID = 0; sampleID < n_sample; sampleID++) {
//            std::cout << min_sample_rank_l[sampleID] << " ";
//        }
//        std::cout << std::endl;
        sample_rank_l.assign(min_sample_rank_l.begin(), min_sample_rank_l.end());

    }
}

#endif //REVERSE_K_RANKS_OPTIMALBRUTEFORCE_HPP
