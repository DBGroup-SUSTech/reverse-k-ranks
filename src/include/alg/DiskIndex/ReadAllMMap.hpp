//
// Created by BianZheng on 2022/4/12.
//

#ifndef REVERSE_K_RANKS_READALLMMAP_HPP
#define REVERSE_K_RANKS_READALLMMAP_HPP

#include "alg/TopkMaxHeap.hpp"
#include "struct/DistancePair.hpp"
#include "struct/UserRankElement.hpp"
#include "util/TimeMemory.hpp"

#include <memory>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <cassert>
#include <fcntl.h>
#include <sys/mman.h>
//#include <sys/stat.h>
#include <unistd.h>

namespace ReverseMIPS {

    template<class T>
    struct FileSpan {
        T *data;
        size_t size;
    };

    class FileReader {
    private:
        int return_val_;
        const char *last_call_;
        char *filename_;
        size_t file_length_;
    public:
        FileReader(const std::filesystem::path &path) noexcept
                : return_val_(0), last_call_(""), filename_(nullptr), file_length_(0) {

            //test whether can read
            auto fd = open(path.c_str(), O_RDONLY);
            if (fd == -1) {
                last_call_ = "open";
                spdlog::error("open file fail");
                return_val_ = -1;
                exit(-1);
            }

            std::ifstream stat_stream = std::ifstream(path, std::ios::binary | std::ios::in);
            if (!stat_stream) {
                spdlog::error("error in reading index");
                exit(-1);
            }

            stat_stream.seekg(0, std::ios::end);
            std::ios::pos_type ss = stat_stream.tellg();
            auto fsize = (size_t) ss;
            stat_stream.close();

            //get the file statistic
//            struct stat st;
//            auto ret = fstat(fd, &st);
//            if (ret == -1) {
//                last_call_ = "fstat";
//                return_val_ = -1;
//                spdlog::error("cannot read the file stat");
//                close(fd);
//                exit(-1);
//            }

            //get the length and mmap information
//            file_length_ = st.st_size;
            file_length_ = fsize;
            filename_ = (char *) mmap64(nullptr, file_length_, PROT_READ, MAP_SHARED, fd, 0);
            if (filename_ == MAP_FAILED) {
                last_call_ = "mmap64";
                return_val_ = -1;
                spdlog::info("mmap fail");
                close(fd);
                exit(-1);
            }
            close(fd);
        }

        FileReader() = default;

        FileReader(FileReader &&r) noexcept
                : return_val_(r.return_val_), last_call_(r.last_call_), filename_(r.filename_),
                  file_length_(r.file_length_) {
            r.filename_ = nullptr;
        }

        FileReader &operator=(FileReader &&rhs) noexcept {
            auto tmp_ret_val = rhs.return_val_;
            auto tmp_last_call = rhs.last_call_;
            auto tmp_addr = rhs.filename_;
            auto tmp_length = rhs.file_length_;
            rhs.filename_ = nullptr;
            using namespace std;
            swap(this->return_val_, tmp_ret_val);
            swap(this->last_call_, tmp_last_call);
            swap(this->filename_, tmp_addr);
            swap(this->file_length_, tmp_length);
            if (tmp_addr) {
                munmap(tmp_addr, tmp_length);
            }
            return *this;
        }

        FileReader(const FileReader &) = delete;

        FileReader &operator=(const FileReader &) = delete;

        ~FileReader() noexcept {
            if (filename_) {
                munmap(filename_, file_length_);
            }
        }

        template<class T>
        FileSpan<T> read(off64_t offset_byte, size_t ele_size) noexcept {
            assert(offset_byte >= 0);
            auto begin = reinterpret_cast<T *>(filename_ + offset_byte);
            auto actual_ele_size = static_cast<size_t>(std::min(file_length_, offset_byte + ele_size * sizeof(T)) -
                                                       std::min(file_length_, (size_t) offset_byte)) / sizeof(T);
            return FileSpan<T>{begin, actual_ele_size};
        }

        operator bool() noexcept {
            return return_val_ == 0;
        }

        const char *last_call() noexcept {
            return last_call_;
        }

    };

    class ReadAllMMap {
        int n_data_item_, n_user_;
        const char *index_path_;

        inline int FineBinarySearch(const float &queryIP, const int &userID,
                                    const int &base_rank,
                                    const int &read_count) const {
            if (read_count == 0) {
                return base_rank + 1;
            }
            const float *cache_ptr = disk_cache_.data;
            if (disk_cache_.size != read_count) {
                spdlog::error("array overflow, program exit");
                exit(-1);
            }
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

        //variable in retrieval
        FileReader file_reader_;
        FileSpan<float> disk_cache_;
        int n_refine_user_;
        std::vector<UserRankElement> user_topk_cache_l_;


        inline ReadAllMMap() {}

        inline ReadAllMMap(const int &n_user, const int &n_data_item, const char *index_path) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->index_path_ = index_path;
            this->user_topk_cache_l_.resize(n_user);
        }

        void RetrievalPreprocess() {
            read_disk_time_ = 0;
            exact_rank_refinement_time_ = 0;
            file_reader_ = FileReader(this->index_path_);
        }

        inline void ReadDisk(const int &userID, const int &start_idx, const int &read_count) {
            int64_t offset = (int64_t) userID * n_data_item_ + start_idx;
            offset *= sizeof(float);
//            int64_t read_count_offset = read_count * sizeof(float);
            disk_cache_ = file_reader_.read<float>(offset, read_count);
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

//                if (n_refine_user_ % 7500 == 0) {
//                    const double progress = n_refine_user_ / (0.01 * refine_user_size);
//                    spdlog::info(
//                            "compute rank {:.2f}%, io_cost {}, ip_cost {}, read_disk_time {:.3f}s, rank_compute_time {:.3f}s, {:.2f}s/iter Mem: {} Mb",
//                            progress, io_cost, ip_cost, read_disk_time, rank_computation_time,
//                            record.get_elapsed_time_second(), get_current_RSS() / 1000000);
//                    record.reset();
//                }
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
        }

        std::string IndexInfo() {
            std::string info = "ReadAllMMap do not need to calculate the information";
            return info;
        }

    };
}
#endif //REVERSE_K_RANKS_READALLMMAP_HPP
