//
// Created by BianZheng on 2022/4/12.
//

#ifndef REVERSE_K_RANKS_READALLDIRECTIO_HPP
#define REVERSE_K_RANKS_READALLDIRECTIO_HPP

#include "alg/TopkMaxHeap.hpp"
#include "struct/DistancePair.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"
#include "util/TimeMemory.hpp"

#include <memory>
#include <spdlog/spdlog.h>
#include <fcntl.h>
#include <unistd.h>

namespace ReverseMIPS {

#define DISK_PAGE_SIZE 512

    using namespace std;

    class DiskData {
    public:
        const float *data_;
        size_t n_ele_;

        inline DiskData() = default;

        inline DiskData(const char *data, const size_t &n_ele, const int64_t &offset_byte) {
            this->data_ = (const float *) (data + offset_byte);
            this->n_ele_ = n_ele;
        }
    };

    class ReadDiskIndex {
        int fileID_;
        char *disk_cache_;
        const char *index_path_;
        int64_t n_user_, n_data_item_;
        int64_t io_ele_unit_, disk_cache_capacity_;
        TimeRecord record_;

    public:

        inline ReadDiskIndex() = default;

        inline ReadDiskIndex(const int64_t &n_user, const int64_t &n_data_item, const char *index_path) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->index_path_ = index_path;
            this->io_ele_unit_ = DISK_PAGE_SIZE / sizeof(float);
            this->disk_cache_capacity_ = (n_data_item_ / io_ele_unit_ + 2) * io_ele_unit_;
            assert(disk_cache_capacity_ >= n_data_item);
        }

        void StartRead() {
            fileID_ = open(index_path_, O_DIRECT | O_RDONLY, 0660);
            if (fileID_ == -1) {
                perror("open");
            }

            disk_cache_ = (char *) aligned_alloc(DISK_PAGE_SIZE, disk_cache_capacity_ * sizeof(float));
            assert(disk_cache_capacity_ * sizeof(float) % DISK_PAGE_SIZE == 0);
            if (!disk_cache_) {
                perror("aligned_alloc");
            }
        }

        DiskData ReadDisk(const int &userID, const int &start_idx, const int &read_count,
                          double &read_disk_time, size_t &io_cost) {
            assert(0 <= start_idx && start_idx < n_data_item_);
            assert(0 <= read_count && read_count <= n_data_item_);
            assert(0 <= start_idx + read_count && start_idx + read_count <= n_data_item_);
            const int64_t ele_offset = userID * n_data_item_ + start_idx;
            //make alignment
            const int64_t ele_actual_offset = ele_offset - (ele_offset % io_ele_unit_);

            const int64_t ele_pad_read_count = ele_offset % io_ele_unit_ + read_count;
            const int64_t ele_actual_read_count =
                    io_ele_unit_ *
                    (ele_pad_read_count / io_ele_unit_ + (ele_pad_read_count % io_ele_unit_ == 0 ? 0 : 1));
            assert(ele_actual_offset <= n_data_item_ * (userID + 1));
            assert(ele_actual_offset + ele_actual_read_count >= ele_offset + read_count);
            assert(ele_actual_read_count <= disk_cache_capacity_);

            record_.reset();
            off_t seek_offset = lseek(fileID_, sizeof(float) * ele_actual_offset, SEEK_SET);
            ssize_t read_chars = read(fileID_, disk_cache_, ele_actual_read_count * sizeof(float));
            read_disk_time = record_.get_elapsed_time_second();
            io_cost = ele_actual_read_count;

            assert(seek_offset == sizeof(float) * ele_actual_offset);
            assert(read_chars != -1 && (read_chars == 0 || read_count <= read_chars));
            assert(ele_actual_read_count * sizeof(float) % DISK_PAGE_SIZE == 0);

            DiskData res(disk_cache_, read_count, ele_offset % io_ele_unit_ * sizeof(float));
            return res;
        }

        void FinishRead() {
            close(fileID_);
            free(disk_cache_);
        }

    };

    class ReadAllDirectIO {
        int n_data_item_, n_user_;
        const char *index_path_;

        inline int FineBinarySearch(const float &queryIP, const int &userID,
                                    const int &base_rank,
                                    const int &read_count) const {
            if (read_count == 0) {
                return base_rank + 1;
            }
            const float *cache_ptr = disk_data_.data_;
            auto iter_begin = cache_ptr;
            auto iter_end = cache_ptr + read_count;
            assert(disk_data_.n_ele_ == read_count);

            auto lb_ptr = std::lower_bound(iter_begin, iter_end, queryIP,
                                           [](const float &arrIP, float queryIP) {
                                               return arrIP > queryIP;
                                           });
            return (int) (lb_ptr - iter_begin) + base_rank + 1;
        }

    public:

        TimeRecord record_;

        //variable in retrieval
        ReadDiskIndex read_disk_index_;
        DiskData disk_data_;

        int n_refine_user_;
        int n_read_disk_user_;
        std::vector<UserRankElement> user_topk_cache_l_;


        inline ReadAllDirectIO() {}

        inline ReadAllDirectIO(const int &n_user, const int &n_data_item, const char *index_path) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->index_path_ = index_path;
            this->user_topk_cache_l_.resize(n_user);
            this->read_disk_index_ = ReadDiskIndex(n_user, n_data_item, index_path);
        }

        void RetrievalPreprocess() {
            read_disk_index_.StartRead();
        }

        void GetRank(const std::vector<float> &queryIP_l,
                     const std::vector<int> &rank_lb_l, const std::vector<int> &rank_ub_l,
                     const std::vector<char> &prune_l, const std::vector<char> &result_l,
                     const int &refine_user_size, const int &remain_n_result,
                     size_t &io_cost, double &read_disk_time, double &refine_rank_time) {

            n_refine_user_ = 0;
            n_read_disk_user_ = 0;
            io_cost = 0;
            read_disk_time = 0;
            refine_rank_time = 0;

            assert(remain_n_result <= refine_user_size);
            TopkMaxHeap heap(remain_n_result);
            assert(refine_user_size == 0 || refine_user_size > 1);

            if (remain_n_result == 0) {
                return;
            }

//            TimeRecord record;
//            record.reset();
            for (int userID = 0; userID < n_user_; userID++) {
                if (result_l[userID] || prune_l[userID]) {
                    continue;
                }
                assert(rank_ub_l[userID] <= rank_lb_l[userID]);
                if (heap.Front() != -1 && heap.Front() < rank_ub_l[userID]) {
                    continue;
                }
                int rank;
                if (rank_lb_l[userID] == rank_ub_l[userID]) {
                    rank = rank_ub_l[userID];
                } else {
                    rank = GetSingleRank(queryIP_l[userID], rank_lb_l[userID], rank_ub_l[userID], userID,
                                         io_cost, read_disk_time, refine_rank_time);
                    n_read_disk_user_++;
                }
                user_topk_cache_l_[n_refine_user_] = UserRankElement(userID, rank, queryIP_l[userID]);
                n_refine_user_++;
                heap.Update(rank);

//                if (n_refine_user_ % 7500 == 0) {
//                    const double progress = n_refine_user_ / (0.01 * refine_user_size);
//                    spdlog::info(
//                            "compute rank {:.2f}%, io_cost {}, read_disk_time {:.3f}s, rank_compute_time {:.3f}s, {:.2f}s/iter Mem: {} Mb",
//                            progress, io_cost, read_disk_time, rank_computation_time,
//                            record.get_elapsed_time_second(), get_current_RSS() / 1000000);
//                    record.reset();
//                }
            }

            assert(0 <= remain_n_result && remain_n_result <= n_refine_user_ && n_refine_user_ <= n_user_);
            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_refine_user_,
                      std::less());

        }

        int GetSingleRank(const float &queryIP, const int &rank_lb, const int &rank_ub, const int &userID,
                          size_t &io_cost, double &read_disk_time, double &refine_rank_time) {
            assert(rank_lb > rank_ub);
            int end_idx = rank_lb;
            int start_idx = rank_ub;
            assert(0 <= start_idx && start_idx <= end_idx && end_idx <= n_data_item_);

            int base_rank = start_idx;
            int read_count = end_idx - start_idx;

            assert(0 <= read_count && read_count <= n_data_item_);

            double tmp_read_disk_time = 0;
            size_t tmp_io_cost = 0;
            disk_data_ = read_disk_index_.ReadDisk(userID, start_idx, read_count,
                                                   tmp_read_disk_time, tmp_io_cost);
            io_cost += tmp_io_cost;
            read_disk_time += tmp_read_disk_time;
            record_.reset();
            int rank = FineBinarySearch(queryIP, userID, base_rank, read_count);
            refine_rank_time += record_.get_elapsed_time_second();

            return rank;
        }

        void FinishRetrieval() {
            read_disk_index_.FinishRead();
        }

        std::string IndexInfo() {
            std::string info = "ReadAllDirectIO do not need to calculate the information";
            return info;
        }

    };
}
#endif //REVERSE_K_RANKS_READALLDIRECTIO_HPP
