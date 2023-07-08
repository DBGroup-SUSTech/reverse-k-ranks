//
// Created by BianZheng on 2022/4/12.
//

#ifndef REVERSE_K_RANKS_READALL_HPP
#define REVERSE_K_RANKS_READALL_HPP

#include "alg/TopkMaxHeap.hpp"
#include "struct/DistancePair.hpp"
#include "struct/UserRankElement.hpp"
#include "util/TimeMemory.hpp"

#include <memory>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class ReadAll {
        int n_data_item_, n_user_;
        const char *index_path_;

        inline int FineBinarySearch(const float &queryIP, const int &userID,
                                    const int &base_rank,
                                    const int &read_count) const {
            if (read_count == 0) {
                return base_rank + 1;
            }
            const float *cache_ptr = disk_cache_.get();
            auto iter_begin = cache_ptr;
            auto iter_end = cache_ptr + read_count;

            auto lb_ptr = std::lower_bound(iter_begin, iter_end, queryIP,
                                           [](const float &arrIP, float queryIP) {
                                               return arrIP > queryIP;
                                           });
            return (int) (lb_ptr - iter_begin) + base_rank + 1;
        }

    public:

        TimeRecord read_disk_record_, exact_rank_refinement_record_;
        double read_disk_time_, exact_rank_refinement_time_;

        //variable in build index
        std::ofstream out_stream_;

        //variable in retrieval
        std::ifstream index_stream_;
        std::unique_ptr<float[]> disk_cache_;
        int n_refine_user_;
        std::vector<UserRankElement> user_topk_cache_l_;


        inline ReadAll() {}

        inline ReadAll(const int &n_user, const int &n_data_item, const char *index_path) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->index_path_ = index_path;
            this->disk_cache_ = std::make_unique<float[]>(n_data_item_);
            this->user_topk_cache_l_.resize(n_user);
        }

        void
        BuildIndexPreprocess() {
            out_stream_ = std::ofstream(index_path_, std::ios::binary | std::ios::out);
            if (!out_stream_) {
                spdlog::error("error in write result");
                exit(-1);
            }
        }

        void BuildIndexLoop(const DistancePair *distance_cache, const int &batch_n_user = 1) {
            std::vector<float> distance_double(batch_n_user * n_data_item_);
            const uint32_t n_total_cpy = batch_n_user * n_data_item_;
            for (int i = 0; i < n_total_cpy; i++) {
                distance_double[i] = distance_cache[i].dist_;
            }
            BuildIndexLoop(distance_double.data(), batch_n_user);
        }

        void BuildIndexLoop(const float *distance_cache, const int &batch_n_user = 1) {
            // distance_cache: write_every * n_data_item_, n_write <= write_every
            int64_t offset = (int64_t) n_data_item_ * batch_n_user;
            offset *= sizeof(float);
            out_stream_.write((char *) distance_cache, offset);
        }

        void FinishBuildIndex() {
            out_stream_.close();
        }

        void RetrievalPreprocess() {
            read_disk_time_ = 0;
            exact_rank_refinement_time_ = 0;
            index_stream_ = std::ifstream(this->index_path_, std::ios::binary | std::ios::in);
            if (!index_stream_) {
                spdlog::error("error in writing index");
            }
        }

        inline void ReadDisk(const int &userID, const int &start_idx, const int &read_count) {
            int64_t offset = (int64_t) userID * n_data_item_ + start_idx;
            offset *= sizeof(float);
            int64_t read_count_offset = read_count * sizeof(float);
            index_stream_.seekg(offset, std::ios::beg);
            index_stream_.read((char *) disk_cache_.get(), read_count_offset);
        }

        inline void ReadDiskNoCache(const int &userID, std::vector<float> &distance_l) {
            assert(distance_l.size() == n_data_item_);
            int64_t offset = (int64_t) userID * n_data_item_;
            offset *= sizeof(float);
            int64_t read_count_offset = n_data_item_ * sizeof(float);
            index_stream_.seekg(offset, std::ios::beg);
            index_stream_.read((char *) distance_l.data(), read_count_offset);
        }

        void GetRank(const std::vector<float> &queryIP_l,
                     const std::vector<int> &rank_lb_l, const std::vector<int> &rank_ub_l,
                     const std::vector<int> &refine_seq_l, const int &refine_user_size, const int &remain_n_result,
                     size_t &io_cost, size_t &ip_cost,
                     double &read_disk_time, double &rank_computation_time) {

            n_refine_user_ = 0;
            io_cost = 0;
            ip_cost = 0;
            read_disk_time = 0;
            rank_computation_time = 0;

            assert(remain_n_result <= refine_user_size);
            TopkMaxHeap heap(remain_n_result);

            TimeRecord record;
            record.reset();
            for (int refineID = 0; refineID < refine_user_size; refineID++) {
                const int userID = refine_seq_l[refineID];
                assert(rank_ub_l[userID] <= rank_lb_l[userID]);
                if (heap.Front() != -1 && heap.Front() < rank_ub_l[userID]) {
                    continue;
                }
                const int rank = GetSingleRank(queryIP_l[userID], rank_lb_l[userID], rank_ub_l[userID], userID,
                                               io_cost, ip_cost, read_disk_time, rank_computation_time);

                user_topk_cache_l_[n_refine_user_] = UserRankElement(userID, rank, queryIP_l[userID]);
                n_refine_user_++;
                heap.Update(rank);

            }

            assert(0 <= remain_n_result && remain_n_result <= n_refine_user_ && n_refine_user_ <= n_user_);
            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_refine_user_,
                      std::less());
        }

        int GetSingleRank(const float &queryIP, const int &rank_lb, const int &rank_ub, const int &userID,
                          size_t &io_cost, size_t &ip_cost,
                          double &read_disk_time, double &rank_computation_time) {
            if (rank_lb == rank_ub) {
                int rank = rank_lb;
                return rank;
            }
            int end_idx = rank_lb;
            int start_idx = rank_ub;
            assert(0 <= start_idx && start_idx <= end_idx && end_idx <= n_data_item_);

            int base_rank = start_idx;
            int read_count = end_idx - start_idx;

            assert(0 <= read_count && read_count <= n_data_item_);

            read_disk_record_.reset();
            ReadDisk(userID, start_idx, read_count);
            const double tmp_read_disk = read_disk_record_.get_elapsed_time_second();
            io_cost += read_count;
            read_disk_time += tmp_read_disk;
            read_disk_time_ += tmp_read_disk;
            exact_rank_refinement_record_.reset();
            int rank = FineBinarySearch(queryIP, userID, base_rank, read_count);
            exact_rank_refinement_time_ += exact_rank_refinement_record_.get_elapsed_time_second();

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
#endif //REVERSE_K_RANKS_READALL_HPP
