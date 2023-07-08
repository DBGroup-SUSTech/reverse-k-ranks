//
// Created by BianZheng on 2022/6/20.
//

#ifndef REVERSE_K_RANKS_TOPTID_HPP
#define REVERSE_K_RANKS_TOPTID_HPP

#include "alg/DiskIndex/ComputeRank/BaseIPBound.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "struct/DistancePair.hpp"
#include "struct/UserRankElement.hpp"
#include "util/TimeMemory.hpp"

#include <fstream>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {
    class TopTID {

        inline void ReadDisk(const int &userID, const int &start_idx, const int &read_count) {
            assert(0 <= start_idx + read_count && start_idx + read_count <= topt_);
            int64_t offset = (int64_t) userID * topt_ + start_idx;
            offset *= sizeof(int);
            index_stream_.seekg(offset, std::ios::beg);
            int64_t read_count_byte = read_count * sizeof(int);

            assert(0 <= offset + read_count_byte && offset + read_count_byte <= n_user_ * topt_ * sizeof(int));

            index_stream_.read((char *) disk_cache_.get(), read_count_byte);
        }

        inline void ComputeCandIP(const VectorMatrix &item, const double *user_vecs, const int &read_count) {
            for (int candID = 0; candID < read_count; candID++) {
                const double *item_vecs = item.getVector(disk_cache_[candID]);
                const double IP = InnerProduct(item_vecs, user_vecs, vec_dim_);
                candIP_cache_[candID] = IP;
            }
        }

        inline int FineBinarySearch(const double &queryIP, const int &userID,
                                    const int &base_rank,
                                    const int &read_count) const {
            if (read_count == 0) {
                return base_rank + 1;
            }
            const double *cache_ptr = candIP_cache_.get();
            auto iter_begin = cache_ptr;
            auto iter_end = cache_ptr + read_count;

            auto lb_ptr = std::lower_bound(iter_begin, iter_end, queryIP,
                                           [](const double &arrIP, double queryIP) {
                                               return arrIP > queryIP;
                                           });
            return (int) (lb_ptr - iter_begin) + base_rank + 1;
        }

        void BelowTopt(const VectorMatrix &item, const double *user_vecs, const double &queryIP,
                       const int &rank_lb, const int &rank_ub, const int &userID, size_t &n_compute) {
            int end_idx = rank_lb;
            int start_idx = rank_ub;
            assert(0 <= start_idx && start_idx <= end_idx && end_idx <= topt_);

            int base_rank = start_idx;
            int read_count = end_idx - start_idx;

            assert(0 <= start_idx + read_count && start_idx + read_count <= topt_);

            read_disk_record_.reset();
            ReadDisk(userID, start_idx, read_count);
            read_disk_time_ += read_disk_record_.get_elapsed_time_second();
            exact_rank_record_.reset();
            ComputeCandIP(item, user_vecs, read_count);
            n_compute += read_count;
            int rank = FineBinarySearch(queryIP, userID, base_rank, read_count);
            exact_rank_time_ += exact_rank_record_.get_elapsed_time_second();

            user_topk_cache_l_[n_candidate_] = UserRankElement(userID, rank, queryIP);
            n_candidate_++;
        }

        void BetweenTopt(const double &queryIP,
                         const int &rank_lb, const int &rank_ub,
                         const int &userID, const double *user_vecs, const VectorMatrix &item,
                         size_t &n_compute) {
            //check the data is in topt
            int end_idx = topt_;
            int start_idx = rank_ub;
            assert(0 <= start_idx && start_idx <= end_idx && end_idx <= topt_);

            int base_rank = start_idx;
            int read_count = end_idx - start_idx;

            assert(0 <= read_count && read_count <= topt_);

            read_disk_record_.reset();
            ReadDisk(userID, start_idx, read_count);
            read_disk_time_ += read_disk_record_.get_elapsed_time_second();
            exact_rank_record_.reset();
            ComputeCandIP(item, user_vecs, read_count);
            n_compute += read_count;
            int rank = FineBinarySearch(queryIP, userID, base_rank, read_count);
            exact_rank_time_ += exact_rank_record_.get_elapsed_time_second();

            if (rank == topt_ + 1) {
                exact_rank_record_.reset();
                rank = exact_rank_ins_.QueryRankByCandidate(user_vecs, userID, item, queryIP);
                rank++;
                n_compute += n_data_item_;
                exact_rank_time_ += exact_rank_record_.get_elapsed_time_second();
            }

            user_topk_cache_l_[n_candidate_] = UserRankElement(userID, rank, queryIP);
            n_candidate_++;
        }

        void
        AboveTopt(const double &queryIP,
                  const int &userID, const double *user_vecs, const VectorMatrix &item,
                  size_t &n_compute) {
            exact_rank_record_.reset();
            int rank = exact_rank_ins_.QueryRankByCandidate(user_vecs, userID, item, queryIP);
            rank++;
            n_compute += n_data_item_;
            exact_rank_time_ += exact_rank_record_.get_elapsed_time_second();

            user_topk_cache_l_[n_candidate_] = UserRankElement(userID, rank, queryIP);
            n_candidate_++;
        }

    public:
        int n_data_item_, n_user_, vec_dim_, topt_;
        const char *index_path_;
        BaseIPBound exact_rank_ins_;

        TimeRecord read_disk_record_, exact_rank_record_;
        double read_disk_time_, exact_rank_time_;

        //variable in build index
        std::ofstream out_stream_;

        //variable in retrieval
        std::ifstream index_stream_;
        std::unique_ptr<int[]> disk_cache_;
        std::unique_ptr<double[]> candIP_cache_;
        int n_candidate_;
        std::vector<UserRankElement> user_topk_cache_l_;

        inline TopTID() = default;

        inline TopTID(const int &n_user, const int &n_data_item, const int &vec_dim,
                      const char *index_path,
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

            this->disk_cache_ = std::make_unique<int[]>(topt);
            this->candIP_cache_ = std::make_unique<double[]>(topt);
            this->user_topk_cache_l_.resize(n_user);
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

        void BuildIndexLoop(const DistancePair *distance_cache) {
            // distance_cache: write_every * n_data_item_, n_write <= write_every
            for (int candID = 0; candID < topt_; candID++) {
                disk_cache_[candID] = distance_cache[candID].ID_;
            }
            out_stream_.write((char *) disk_cache_.get(), sizeof(int) * topt_);
        }

        void FinishBuildIndex() {
            out_stream_.close();
        }

        void RetrievalPreprocess() {
            read_disk_time_ = 0;
            exact_rank_time_ = 0;
            index_stream_ = std::ifstream(this->index_path_, std::ios::binary | std::ios::in);
            if (!index_stream_) {
                spdlog::error("error in reading index");
            }
        }

        void PreprocessQuery(const double *query_vecs, const int &vec_dim, double *query_write_vecs) {
            exact_rank_ins_.PreprocessQuery(query_vecs, vec_dim, query_write_vecs);
        }

        void GetRank(const std::vector<double> &queryIP_l,
                     const std::vector<int> &rank_lb_l, const std::vector<int> &rank_ub_l,
                     const std::vector<bool> &prune_l, const VectorMatrix &user, const VectorMatrix &item,
                     size_t &n_compute) {
            assert(n_user_ == queryIP_l.size());
            assert(n_user_ == rank_lb_l.size() && n_user_ == rank_ub_l.size());
            assert(n_user_ == prune_l.size());

            n_candidate_ = 0;
            n_compute = 0;
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID]) {
                    continue;
                }
                const double *user_vecs = user.getVector(userID);

                const int rank_lb = rank_lb_l[userID];
                const int rank_ub = rank_ub_l[userID];
                const double queryIP = queryIP_l[userID];
                assert(rank_ub <= rank_lb);
                if (rank_lb <= topt_) {
                    //retrieval the top-t like before
                    BelowTopt(item, user_vecs, queryIP, rank_lb, rank_ub, userID, n_compute);
                } else if (rank_ub <= topt_ && topt_ <= rank_lb) {
                    BetweenTopt(queryIP, rank_lb, rank_ub, userID, user_vecs, item, n_compute);
                } else if (topt_ < rank_ub) {
                    AboveTopt(queryIP, userID, user_vecs, item, n_compute);
                } else {
                    spdlog::error("have bug in get rank, topt ID IP");
                }
            }

            assert(0 <= n_candidate_ && n_candidate_ <= n_user_);
            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_candidate_,
                      std::less());

        };

        void FinishRetrieval() {
            index_stream_.close();
        }

        std::string IndexInfo() {
            std::string info = "Exact rank method_name: " + exact_rank_ins_.method_name;
            return info;
        }

    };
}
#endif //REVERSE_K_RANKS_TOPTID_HPP
