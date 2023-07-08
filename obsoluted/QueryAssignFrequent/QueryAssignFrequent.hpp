//
// Created by BianZheng on 2022/7/30.
//

#ifndef REVERSE_KRANKS_QUERYASSIGNFREQUENT_HPP
#define REVERSE_KRANKS_QUERYASSIGNFREQUENT_HPP

#include "alg/TopkMaxHeap.hpp"
#include "alg/DiskIndex/ComputeRank/BaseIPBound.hpp"
#include "struct/DistancePair.hpp"
#include "struct/UserRankElement.hpp"

#include <memory>
#include <spdlog/spdlog.h>
#include <numeric>

namespace ReverseMIPS {

    class QueryAssignFrequent {
        inline int FineBinarySearch(const double &queryIP, const int &userID,
                                    const int &base_rank,
                                    const int &read_count) const {
            if (read_count == 0) {
                return base_rank + 1;
            }
            const double *cache_ptr = disk_cache_.get();
            auto iter_begin = cache_ptr;
            auto iter_end = cache_ptr + read_count;

            auto lb_ptr = std::lower_bound(iter_begin, iter_end, queryIP,
                                           [](const double &arrIP, double queryIP) {
                                               return arrIP > queryIP;
                                           });
            return (int) (lb_ptr - iter_begin) + base_rank + 1;
        }

        inline void ReadDisk(const int &user_offset, const int &start_idx, const int &read_count) {
            uint64_t offset = sizeof(double) * (user_offset * n_data_item_ + start_idx);
            uint64_t read_count_offset = sizeof(double) * read_count;
            index_stream_.seekg(offset, std::ios::beg);
            index_stream_.read((char *) disk_cache_.get(), read_count_offset);
        }

    public:
        int n_data_item_, n_user_, n_store_user_;
        const char *index_path_;
        BaseIPBound exact_rank_ins_;
        //if -1, then not store; if 1, then it store
        std::vector<int> store_user_offset_l_;

        TimeRecord read_disk_record_, exact_rank_refinement_record_;
        double read_disk_time_, exact_rank_refinement_time_;

        //variable in build index
        std::ofstream out_stream_;

        //variable in retrieval
        std::ifstream index_stream_;
        std::unique_ptr<double[]> disk_cache_;
        int n_candidate_;
        std::vector<UserRankElement> user_topk_cache_l_;


        inline QueryAssignFrequent() {}

        inline QueryAssignFrequent(const int &n_user, const int &n_data_item, const int &vec_dim,
                                   const char *index_path,
                                   const size_t &index_size_gb,
                                   const char *query_distribution_path,
                                   const int &n_sample_query, const int &sample_topk) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->index_path_ = index_path;
            this->exact_rank_ins_ = BaseIPBound(n_data_item, vec_dim);

            this->disk_cache_ = std::make_unique<double[]>(n_data_item);
            this->user_topk_cache_l_.resize(n_user);

            ComputeParameter(n_user, n_data_item, index_size_gb);

            LoadQueryDistribution(query_distribution_path, n_sample_query, sample_topk);
        }

        void ComputeParameter(const int &n_user, const int &n_data_item, const size_t &index_size_gb) {
            const uint64_t n_data_item_int64 = n_data_item;
            const uint64_t n_user_int64 = n_user;
            uint64_t single_user_size = sizeof(double) * n_data_item_int64;
            uint64_t capacity_size_byte = index_size_gb * 1024 * 1024 * 1024;
            uint64_t origin_size_byte = sizeof(double) * n_user_int64 * n_data_item_int64;
            uint64_t n_store_user_big_size = capacity_size_byte / single_user_size;
            n_store_user_ = (int) n_store_user_big_size;
            if (origin_size_byte <= capacity_size_byte) {
                spdlog::info("index size larger than the whole score table, use whole table setting");
                n_store_user_ = n_user;
//                n_store_user_ = n_user / 5 * 4;
            }
            assert(n_store_user_ <= n_user_);
            spdlog::info("n_store_user {}", n_store_user_);
        }

        void LoadQueryDistribution(const char *query_distribution_path,
                                   const int &n_sample_query, const int &sample_topk) {

            std::vector<UserRankElement> query_rank_l(n_sample_query * sample_topk);
            std::ifstream index_stream = std::ifstream(query_distribution_path, std::ios::binary | std::ios::in);
            if (!index_stream) {
                spdlog::error("error in writing index");
            }
            index_stream.read((char *) query_rank_l.data(), sizeof(UserRankElement) * n_sample_query * sample_topk);
            index_stream.close();

            std::vector<int> user_freq_l(n_user_);
            user_freq_l.assign(n_user_, 0);
            const int total_n_sample = n_sample_query * sample_topk;
            for (int sampleID = 0; sampleID < total_n_sample; sampleID++) {
                const UserRankElement element = query_rank_l[sampleID];
                user_freq_l[element.userID_]++;
            }

            std::vector<int> freq_userID_l(n_user_);
            std::iota(freq_userID_l.data(), freq_userID_l.data() + n_user_, 0);
            std::sort(freq_userID_l.data(), freq_userID_l.data() + n_user_,
                      [&](int i1, int i2) { return user_freq_l[i1] > user_freq_l[i2]; });

            std::sort(freq_userID_l.begin(), freq_userID_l.begin() + n_store_user_, std::less());

            store_user_offset_l_.resize(n_user_);
            store_user_offset_l_.assign(n_user_, -1);
            int store_offset = 0;
            for (int freqID = 0; freqID < n_store_user_; freqID++) {
                const int userID = freq_userID_l[freqID];
                store_user_offset_l_[userID] = store_offset;
                store_offset++;
            }

        }

        void BuildIndexPreprocess() {
            out_stream_ = std::ofstream(index_path_, std::ios::binary | std::ios::out);
            if (!out_stream_) {
                spdlog::error("error in write result");
                exit(-1);
            }
        }

        void PreprocessData(VectorMatrix &user, VectorMatrix &data_item) {
            exact_rank_ins_.PreprocessData(user, data_item);
        };

        //should guarantee the evaluating sequence is increase
        void BuildIndexLoop(const DistancePair *distance_cache, const int &userID) {
            std::vector<double> distance_double(n_data_item_);
            for (int i = 0; i < n_data_item_; i++) {
                distance_double[i] = distance_cache[i].dist_;
            }
            BuildIndexLoop(distance_double.data(), userID);
        }

        //should guarantee the evaluating sequence is increase
        void BuildIndexLoop(const double *distance_cache, const int &userID) {
            // distance_cache: write_every * n_data_item_, n_write <= write_every
            if (store_user_offset_l_[userID] != -1) {
                uint64_t offset = sizeof(double) * n_data_item_;
                out_stream_.write((char *) distance_cache, offset);
            }
        }

        void RetrievalPreprocess() {
            read_disk_time_ = 0;
            exact_rank_refinement_time_ = 0;
            index_stream_ = std::ifstream(this->index_path_, std::ios::binary | std::ios::in);
            if (!index_stream_) {
                spdlog::error("error in writing index");
            }
        }

        void PreprocessQuery(const double *query_vecs, const int &vec_dim, double *query_write_vecs) {
            memcpy(query_write_vecs, query_vecs, vec_dim * sizeof(double));
        }

        void GetRank(const std::vector<double> &queryIP_l,
                     const std::vector<int> &rank_lb_l, const std::vector<int> &rank_ub_l,
                     const VectorMatrix &user, const VectorMatrix &data_item,
                     std::vector<bool> &prune_l, TopkLBHeap &topk_lb_heap, size_t &n_compute) {

            //read disk and fine binary search
            n_compute = 0;
            n_candidate_ = 0;
            int topk_lb_rank = topk_lb_heap.Front();
            topk_lb_heap.Reset();
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID] && rank_lb_l[userID] + 1 <= topk_lb_rank) {
                    continue;
                }
                const int rank = GetSingleRank(queryIP_l[userID], rank_lb_l[userID], rank_ub_l[userID], userID, user,
                                               data_item, n_compute);
                topk_lb_heap.Update(rank);
                prune_l[userID] = true;
            }
            assert(topk_lb_heap.Front() != -1);
            topk_lb_rank = topk_lb_heap.Front();

            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID] || rank_ub_l[userID] > topk_lb_rank) {
                    continue;
                }
                GetSingleRank(queryIP_l[userID], rank_lb_l[userID], rank_ub_l[userID], userID, user, data_item,
                              n_compute);
            }

            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_candidate_,
                      std::less());
        }

        int GetSingleRank(const double &queryIP, const int &rank_lb, const int &rank_ub, const int &userID,
                          const VectorMatrix &user, const VectorMatrix &data_item,
                          size_t &n_compute) {
            const int user_offset = store_user_offset_l_[userID];
            int rank;
            if (user_offset == -1) {
                const double *user_vecs = user.getVector(userID);
                rank = exact_rank_ins_.QueryRankByCandidate(user_vecs, userID, data_item, queryIP);
                rank++;
                n_compute += n_data_item_;
            } else {
                assert(0 <= user_offset && user_offset < n_store_user_);
                int end_idx = rank_lb;
                int start_idx = rank_ub;
                assert(0 <= start_idx && start_idx <= end_idx && end_idx <= n_data_item_);

                int base_rank = start_idx;
                int read_count = end_idx - start_idx;

                read_disk_record_.reset();
                ReadDisk(user_offset, start_idx, read_count);
                read_disk_time_ += read_disk_record_.get_elapsed_time_second();
                exact_rank_refinement_record_.reset();
                rank = FineBinarySearch(queryIP, userID, base_rank, read_count);
                exact_rank_refinement_time_ += exact_rank_refinement_record_.get_elapsed_time_second();
            }

            user_topk_cache_l_[n_candidate_] = UserRankElement(userID, rank, queryIP);
            n_candidate_++;
            return rank;
        }

        void FinishRetrieval() {
            index_stream_.close();
        }

        std::string IndexInfo() {
            std::string info = "ReadAll do not need to calculate the information";
            return info;
        }

    };
}
#endif //REVERSE_KRANKS_QUERYASSIGNFREQUENT_HPP
