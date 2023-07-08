//
// Created by BianZheng on 2022/9/6.
//

#ifndef REVERSE_K_RANKS_READALLID_HPP
#define REVERSE_K_RANKS_READALLID_HPP

#include "alg/TopkMaxHeap.hpp"
#include "alg/DiskIndex/ComputeRank/BaseIPBound.hpp"

#include "struct/DistancePair.hpp"
#include "struct/UserRankElement.hpp"
#include "util/TimeMemory.hpp"

#include <memory>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class ReadAllID {

        inline void ReadDisk(const int &userID, const int &start_idx, const int &read_count) {
            int64_t offset = (int64_t) userID * n_data_item_ + start_idx;
            offset *= sizeof(int);
            int64_t read_count_offset = read_count * sizeof(int);
            index_stream_.seekg(offset, std::ios::beg);
            system("# sync; echo 3 > /proc/sys/vm/drop_caches");
            index_stream_.read((char *) disk_cache_.get(), read_count_offset);
        }

        void
        BuildIndexPreprocess() {
            out_stream_ = std::ofstream(index_path_, std::ios::binary | std::ios::out);
            if (!out_stream_) {
                spdlog::error("error in write result");
                exit(-1);
            }
        }

    public:

        int n_data_item_, n_user_;
        BaseIPBound exact_rank_ins_;
        const char *index_path_;

        TimeRecord read_disk_record_, exact_rank_refinement_record_;
        double read_disk_time_, exact_rank_refinement_time_;

        //variable in build index
        std::ofstream out_stream_;

        //variable in retrieval
        std::ifstream index_stream_;
        std::unique_ptr<int[]> disk_cache_;
        std::unique_ptr<double[]> candIP_cache_;
        int n_candidate_;
        std::vector<UserRankElement> user_topk_cache_l_;


        inline ReadAllID() {}

        inline ReadAllID(const int &n_user, const int &n_data_item, const int &vec_dim, const char *index_path) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;

            this->exact_rank_ins_ = BaseIPBound(n_data_item, vec_dim);

            this->index_path_ = index_path;
            this->disk_cache_ = std::make_unique<int[]>(n_data_item_);
            this->candIP_cache_ = std::make_unique<double[]>(n_data_item_);
            this->user_topk_cache_l_.resize(n_user);

            BuildIndexPreprocess();
        }

        void PreprocessData(VectorMatrix &user, VectorMatrix &data_item) {
            exact_rank_ins_.PreprocessData(user, data_item);
        };

        void BuildIndexLoop(const DistancePair *distance_cache) {
            for (int i = 0; i < n_data_item_; i++) {
                disk_cache_[i] = distance_cache[i].ID_;
            }
            // distance_cache: write_every * n_data_item_, n_write <= write_every
            int64_t offset = (int64_t) n_data_item_ * sizeof(int);
            out_stream_.write((char *) disk_cache_.get(), offset);
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

        void PreprocessQuery(const double *query_vecs, const int &vec_dim, double *query_write_vecs) {
            memcpy(query_write_vecs, query_vecs, vec_dim * sizeof(double));
        }

        void GetRank(const std::vector<double> &queryIP_l,
                     const std::vector<int> &rank_lb_l, const std::vector<int> &rank_ub_l,
                     const VectorMatrix &user, const VectorMatrix &item,
                     std::vector<bool> &prune_l, TopkLBHeap &topk_lb_heap,
                     size_t &io_cost, size_t &ip_cost,
                     double &read_disk_time, double &rank_computation_time) {
            io_cost = 0;
            ip_cost = 0;
            read_disk_time = 0;
            rank_computation_time = 0;

            //read disk and fine binary search
            n_candidate_ = 0;
            int topk_lb_rank = topk_lb_heap.Front();
            topk_lb_heap.Reset();
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID] || rank_lb_l[userID] >= topk_lb_rank) {
                    continue;
                }
                const int rank = GetSingleRank(queryIP_l[userID], rank_lb_l[userID], rank_ub_l[userID], userID,
                                               user, item,
                                               io_cost, ip_cost, read_disk_time, rank_computation_time);
                topk_lb_heap.Update(rank);
                prune_l[userID] = true;
            }

            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID] || (topk_lb_heap.Front() != -1 && rank_ub_l[userID] > topk_lb_heap.Front())) {
                    continue;
                }
                const int rank = GetSingleRank(queryIP_l[userID], rank_lb_l[userID], rank_ub_l[userID], userID,
                                               user, item,
                                               io_cost, ip_cost, read_disk_time, rank_computation_time);
                topk_lb_heap.Update(rank);
            }

            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_candidate_,
                      std::less());
        }

        void GetRank(const std::vector<double> &queryIP_l,
                     const std::vector<int> &rank_lb_l, const std::vector<int> &rank_ub_l,
                     const std::vector<bool> &prune_l,
                     const VectorMatrix &user, const VectorMatrix &item,
                     size_t &io_cost, size_t &ip_cost,
                     double &read_disk_time, double &rank_computation_time) {
            io_cost = 0;
            ip_cost = 0;
            read_disk_time = 0;
            rank_computation_time = 0;

            //read disk and fine binary search
            n_candidate_ = 0;
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID]) {
                    continue;
                }
                GetSingleRank(queryIP_l[userID], rank_lb_l[userID], rank_ub_l[userID], userID,
                              user, item,
                              io_cost, ip_cost, read_disk_time, rank_computation_time);
            }

            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_candidate_,
                      std::less());
        }

        int GetSingleRank(const double &queryIP, const int &rank_lb, const int &rank_ub, const int &userID,
                          const VectorMatrix &user, const VectorMatrix &item,
                          size_t &io_cost, size_t &ip_cost,
                          double &read_disk_time, double &rank_computation_time) {
            if (rank_lb == rank_ub) {
                int rank = rank_lb;
                user_topk_cache_l_[n_candidate_] = UserRankElement(userID, rank, queryIP);
                n_candidate_++;
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
            const double *user_vecs = user.getVector(userID);
            int rank = exact_rank_ins_.QueryRankByCandidate(user_vecs, disk_cache_.get(), item, read_count, queryIP,
                                                            base_rank);
            rank++;
            const double tmp_ip_time = exact_rank_refinement_record_.get_elapsed_time_second();
            rank_computation_time += tmp_ip_time;
            exact_rank_refinement_time_ += tmp_ip_time;
            ip_cost += read_count;

            user_topk_cache_l_[n_candidate_] = UserRankElement(userID, rank, queryIP);
            n_candidate_++;
            return rank;
        }

        void FinishRetrieval() {
            index_stream_.close();
        }

        std::string IndexInfo() {
            std::string info = "ReadAllID do not need to calculate the information";
            return info;
        }

    };
}
#endif //REVERSE_K_RANKS_READALLID_HPP
