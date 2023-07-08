//
// Created by BianZheng on 2022/6/20.
//

#ifndef REVERSE_K_RANKS_IDIndex_HPP
#define REVERSE_K_RANKS_IDIndex_HPP

#include "CandidatesIO.hpp"
#include "struct/DistancePair.hpp"
#include "alg/SpaceInnerProduct.hpp"

namespace ReverseMIPS {
    class DiskIndex {

        void BuildIndexPreprocess() {
            out_stream_ = std::ofstream(index_path_, std::ios::binary | std::ios::out);
            if (!out_stream_) {
                spdlog::error("error in write result");
                exit(-1);
            }
        }

        inline void ReadDisk(const int &userID, const int &start_idx, const int &read_count) {
            assert(0 <= start_idx + read_count && start_idx + read_count <= n_data_item_);
            int64_t offset = (int64_t) userID * n_data_item_ + start_idx;
            offset *= sizeof(int);
            index_stream_.seekg(offset, std::ios::beg);
            int64_t read_count_byte = read_count * sizeof(int);

            assert(0 <= offset + read_count_byte && offset + read_count_byte <= n_user_ * n_data_item_ * sizeof(int));

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

    public:
        int n_data_item_, n_user_, vec_dim_;
        const char *index_path_;

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

        inline DiskIndex() = default;

        inline DiskIndex(const int &n_user, const int &n_data_item, const int &vec_dim, const char *index_path) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->vec_dim_ = vec_dim;
            this->index_path_ = index_path;

            this->disk_cache_ = std::make_unique<int[]>(n_data_item_);
            this->candIP_cache_ = std::make_unique<double[]>(n_data_item_);
            this->user_topk_cache_l_.resize(n_user);

            BuildIndexPreprocess();
        }

        void BuildIndexLoop(const DistancePair *distance_cache, const int &n_write) {
            // distance_cache: write_every * n_data_item_, n_write <= write_every
            std::vector<int> distID_l(n_data_item_);
            for (int writeID = 0; writeID < n_write; writeID++) {
                for (int candID = 0; candID < n_data_item_; candID++) {
                    distID_l[candID] = distance_cache[writeID * n_data_item_ + candID].ID_;
                }
                out_stream_.write((char *) distID_l.data(), sizeof(int) * n_data_item_);
            }
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
                     const std::vector<bool> &prune_l, const VectorMatrix &user, const VectorMatrix &item,
                     std::vector<ItemCandidates> &ID_candidate_l) {
            assert(n_user_ == queryIP_l.size());
            assert(n_user_ == rank_lb_l.size() && n_user_ == rank_ub_l.size());
            assert(n_user_ == prune_l.size());

            n_candidate_ = 0;
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID]) {
                    continue;
                }
                const double *user_vecs = user.getVector(userID);

                const int rank_lb = rank_lb_l[userID];
                const int rank_ub = rank_ub_l[userID];
                const double queryIP = queryIP_l[userID];
                assert(rank_ub <= rank_lb);

                int end_idx = rank_lb;
                int start_idx = rank_ub;
                assert(0 <= start_idx && start_idx <= end_idx && end_idx <= n_data_item_);

                int base_rank = start_idx;
                int read_count = end_idx - start_idx;

                assert(0 <= start_idx + read_count && start_idx + read_count <= n_data_item_);

                read_disk_record_.reset();
                ReadDisk(userID, start_idx, read_count);
                read_disk_time_ += read_disk_record_.get_elapsed_time_second();

                std::vector<int> itemID_l(read_count);
                itemID_l.assign(disk_cache_.get(), disk_cache_.get() + read_count);
                std::sort(itemID_l.begin(), itemID_l.end());
                ID_candidate_l.emplace_back(itemID_l, userID, rank_lb, rank_ub);

                exact_rank_record_.reset();
                ComputeCandIP(item, user_vecs, read_count);
                int rank = FineBinarySearch(queryIP, userID, base_rank, read_count);
                exact_rank_time_ += exact_rank_record_.get_elapsed_time_second();

                user_topk_cache_l_[n_candidate_] = UserRankElement(userID, rank, queryIP);
                n_candidate_++;

            }

            assert(0 <= n_candidate_ && n_candidate_ <= n_user_);
            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_candidate_,
                      std::less());

        };

        void FinishRetrieval() {
            index_stream_.close();
        }
    };
}
#endif //REVERSE_K_RANKS_IDIndex_HPP
