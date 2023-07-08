//
// Created by bianzheng on 2022/5/3.
//

#ifndef REVERSE_KRANKS_TOPIP_HPP
#define REVERSE_KRANKS_TOPIP_HPP

#include "alg/DiskIndex/ComputeRank/BaseIPBound.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/DistancePair.hpp"
#include "util/TimeMemory.hpp"

#include <fstream>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {
    class TopTIP {

        inline double ReadDisk(const int &userID, const int &start_idx, const int &read_count) {
            read_disk_record_.reset();
            assert(0 <= start_idx + read_count && start_idx + read_count <= topt_);
            int64_t offset = (int64_t) userID * topt_ + start_idx;
            offset *= sizeof(double);
            index_stream_.seekg(offset, std::ios::beg);
            int64_t read_count_byte = read_count * sizeof(double);

            assert(0 <= offset + read_count_byte && offset + read_count_byte <= n_user_ * topt_ * sizeof(double));

            system("# sync; echo 3 > /proc/sys/vm/drop_caches");
            index_stream_.read((char *) disk_cache_.get(), read_count_byte);
            const double tmp_read_disk_time = read_disk_record_.get_elapsed_time_second();
            return tmp_read_disk_time;
        }

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

        int BelowTopt(const double &queryIP, const int &rank_lb, const int &rank_ub, const int &userID,
                      size_t &io_cost, size_t &ip_cost,
                      double &read_disk_time, double &rank_compute_time) {
            int end_idx = rank_lb;
            int start_idx = rank_ub;
            assert(0 <= start_idx && start_idx <= end_idx && end_idx <= topt_);

            int base_rank = start_idx;
            int read_count = end_idx - start_idx;

            assert(0 <= start_idx + read_count && start_idx + read_count <= topt_);

            const double tmp_read_disk_time = ReadDisk(userID, start_idx, read_count);
            io_cost += read_count;
            read_disk_time += tmp_read_disk_time;
            read_disk_time_ += tmp_read_disk_time;

            exact_rank_record_.reset();
            int rank = FineBinarySearch(queryIP, userID, base_rank, read_count);
            const double tmp_rank_compute_time = exact_rank_record_.get_elapsed_time_second();
            exact_rank_time_ += tmp_rank_compute_time;
            ip_cost += 0;
            return rank;
        }

        int AboveTopt(const double &queryIP,
                      const int &userID, const double *user_vecs, const VectorMatrix &item,
                      size_t &io_cost, size_t &ip_cost,
                      double &read_disk_time, double &rank_compute_time) {

            exact_rank_record_.reset();
            int rank = exact_rank_ins_.QueryRankByCandidate(user_vecs, userID, item, queryIP);
            rank++;
            const double tmp_rank_compute_time = exact_rank_record_.get_elapsed_time_second();
            rank_compute_time += tmp_rank_compute_time;
            exact_rank_time_ += tmp_rank_compute_time;
            ip_cost += n_data_item_;
            io_cost += 0;
            return rank;
        }

    public:
        int n_data_item_, n_user_, vec_dim_, topt_;
        BaseIPBound exact_rank_ins_;
        const char *index_path_;

        TimeRecord read_disk_record_, exact_rank_record_;
        double read_disk_time_, exact_rank_time_;

        //variable in build index
        std::ofstream out_stream_;

        //variable in retrieval
        std::ifstream index_stream_;
        std::unique_ptr<double[]> disk_cache_;
        int n_refine_user_;
        std::vector<UserRankElement> user_topk_cache_l_;

        inline TopTIP() = default;

        inline TopTIP(const int &n_user, const int &n_data_item, const int &vec_dim, const char *index_path,
                      const int &topt) {
            this->exact_rank_ins_ = BaseIPBound(n_data_item, vec_dim);
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->vec_dim_ = vec_dim;
            this->index_path_ = index_path;

            if (topt <= 0 || topt > n_data_item) {
                spdlog::error("topt is invalid, consider change topt_perc");
                exit(-1);
            }

            this->topt_ = topt;
            if (topt_ > n_data_item_) {
                spdlog::error("top-t larger than n_data_item, program exit");
                exit(-1);
            }
            spdlog::info("topt {}", topt_);

            this->disk_cache_ = std::make_unique<double[]>(topt);
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

        void PreprocessData(VectorMatrix &user, VectorMatrix &data_item) {
            exact_rank_ins_.PreprocessData(user, data_item);
        };

        void BuildIndexLoop(const DistancePair *distance_cache) {
            // distance_cache: write_every * n_data_item_, n_write <= write_every
            for (int candID = 0; candID < topt_; candID++) {
                disk_cache_[candID] = distance_cache[candID].dist_;
            }
            out_stream_.write((char *) disk_cache_.get(), topt_ * sizeof(double));
        }

        void BuildIndexLoop(const double *distance_cache) {
            // distance_cache: write_every * n_data_item_, n_write <= write_every
            out_stream_.write((char *) distance_cache, topt_ * sizeof(double));
        }

        void FinishBuildIndex() {
            out_stream_.close();
        }

        void PreprocessQuery(const double *query_vecs, const int &vec_dim, double *query_write_vecs) {
            exact_rank_ins_.PreprocessQuery(query_vecs, vec_dim, query_write_vecs);
        }

        void RetrievalPreprocess() {
            read_disk_time_ = 0;
            exact_rank_time_ = 0;
            index_stream_ = std::ifstream(this->index_path_, std::ios::binary | std::ios::in);
            if (!index_stream_) {
                spdlog::error("error in writing index");
            }
        }

        void GetRank(const std::vector<double> &queryIP_l,
                     const std::vector<int> &rank_lb_l, const std::vector<int> &rank_ub_l,
                     const VectorMatrix &user, const VectorMatrix &item,
                     const std::vector<int> &refine_seq_l, const int &refine_user_size, const int &remain_n_result,
                     size_t &io_cost, size_t &ip_cost,
                     double &read_disk_time, double &rank_compute_time) {
            assert(n_user_ == queryIP_l.size());
            assert(n_user_ == rank_lb_l.size() && n_user_ == rank_ub_l.size());

            n_refine_user_ = 0;
            io_cost = 0;
            ip_cost = 0;
            read_disk_time = 0;
            rank_compute_time = 0;

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
                                               user, item,
                                               io_cost, ip_cost, read_disk_time, rank_compute_time);

                user_topk_cache_l_[n_refine_user_] = UserRankElement(userID, rank, queryIP_l[userID]);
                n_refine_user_++;
                heap.Update(rank);

                if (n_refine_user_ % 7500 == 0) {
                    const double progress = n_refine_user_ / (0.01 * refine_user_size);
                    spdlog::info(
                            "compute rank {:.2f}%, io_cost {}, ip_cost {}, read_disk_time {:.3f}s, rank_compute_time {:.3f}s, {:.2f}s/iter Mem: {} Mb",
                            progress, io_cost, ip_cost, read_disk_time, rank_compute_time,
                            record.get_elapsed_time_second(), get_current_RSS() / 1000000);
                    record.reset();
                }

            }


            assert(0 <= remain_n_result && remain_n_result <= n_refine_user_ && n_refine_user_ <= n_user_);
            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_refine_user_,
                      std::less());
        }

        int GetSingleRank(const double &queryIP, const int &rank_lb, const int &rank_ub, const int &userID,
                          const VectorMatrix &user, const VectorMatrix &item,
                          size_t &io_cost, size_t &ip_cost, double &read_disk_time, double &rank_compute_time) {
            assert(rank_ub <= rank_lb);
            if (rank_ub == rank_lb) {
                int rank = rank_ub;
                return rank;
            }
            int rank;
            if (rank_lb <= topt_) {
                //retrieval the top-t like before
                rank = BelowTopt(queryIP, rank_lb, rank_ub, userID,
                                 io_cost, ip_cost,
                                 read_disk_time, rank_compute_time);
            } else {//topt_ < rank_ub
                const double *user_vecs = user.getVector(userID);
                rank = AboveTopt(queryIP, userID, user_vecs, item,
                                 io_cost, ip_cost,
                                 read_disk_time, rank_compute_time);
            }
            return rank;
        }

        void FinishRetrieval() {
            index_stream_.close();
        }

        std::string IndexInfo() {
            std::string info = "Exact rank method_name: " + exact_rank_ins_.method_name;
            return info;
        }

    };
}

#endif //REVERSE_KRANKS_TOPIP_HPP
