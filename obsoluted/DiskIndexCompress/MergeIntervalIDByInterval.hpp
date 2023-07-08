//
// Created by BianZheng on 2022/6/27.
//

#ifndef REVERSE_KRANKS_MERGEINTERVAL_HPP
#define REVERSE_KRANKS_MERGEINTERVAL_HPP

#include "alg/SpaceInnerProduct.hpp"
//#include "alg/Cluster/KMeansParallel.hpp"
#include "alg/Cluster/GreedyMergeMinClusterSize.hpp"
#include "alg/DiskIndex/ComputeRank/PartIntPartNorm.hpp"
#include "struct/DistancePair.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/UserRankBound.hpp"
#include "util/TimeMemory.hpp"
#include <set>
#include <cfloat>
#include <memory>
#include <spdlog/spdlog.h>
#include <armadillo>

namespace ReverseMIPS {

    class MergeIntervalIDByInterval {
    public:
        //index variable
        int n_user_, n_data_item_, vec_dim_, n_merge_user_;
        //n_cache_rank_: stores how many intervals for each merged user
        std::vector<uint32_t> merge_label_l_; // n_user, stores which cluster the user belons to
        PartIntPartNorm exact_rank_ins_;
        const char *index_path_;

        //record time memory
        TimeRecord read_disk_record_, exact_rank_refinement_record_;
        double read_disk_time_, exact_rank_refinement_time_;

        //variable in build index
        std::ofstream out_stream_;
        //n_data_item, stores the UserRankBound in the disk, used for build index and retrieval
        std::vector<UserRankBound<unsigned char>> disk_write_cache_l_;

        //variable in retrieval
        std::ifstream index_stream_;
        int n_candidate_;
        std::vector<bool> is_compute_l_;
        std::vector<UserRankElement> user_topk_cache_l_; //n_user, used for sort the element to return the top-k
        std::vector<std::pair<unsigned char, unsigned char>> disk_retrieval_cache_l_;
        std::vector<bool> item_cand_l_;

        inline MergeIntervalIDByInterval() {}

        inline MergeIntervalIDByInterval(const VectorMatrix &user,
                                         const char *index_path, const int &n_data_item,
                                         const int &n_merge_user) {
            this->exact_rank_ins_ = PartIntPartNorm(user.n_vector_, n_data_item, user.vec_dim_);;
            this->n_user_ = user.n_vector_;
            this->n_data_item_ = n_data_item;
            this->vec_dim_ = user.vec_dim_;
            this->index_path_ = index_path;
            this->n_merge_user_ = n_merge_user;

            spdlog::info("n_merge_user {}", n_merge_user_);

            this->merge_label_l_.resize(n_user_);
            this->disk_write_cache_l_.resize(n_data_item_);
            for (int itemID = 0; itemID < n_data_item; itemID++) {
                this->disk_write_cache_l_[itemID].Reset();
            }
            this->is_compute_l_.resize(n_merge_user_);
            this->user_topk_cache_l_.resize(n_user_);
            this->disk_retrieval_cache_l_.resize(n_data_item_);
            this->item_cand_l_.resize(n_data_item_);

            BuildIndexPreprocess(user);
        }

        void
        BuildIndexPreprocess(const VectorMatrix &user) {
            merge_label_l_ = GreedyMergeMinClusterSize::ClusterLabel(user, n_merge_user_);

//            printf("cluster size\n");
//            for (int mergeID = 0; mergeID < n_merge_user_; mergeID++) {
//                int count = 0;
//                for (int userID = 0; userID < n_user_; userID++) {
//                    if (merge_label_l_[userID] == mergeID) {
//                        count++;
//                    }
//                }
//                printf("%d ", count);
//            }
//            printf("\n");

            out_stream_ = std::ofstream(index_path_, std::ios::binary | std::ios::out);
            if (!out_stream_) {
                spdlog::error("error in write result");
                exit(-1);
            }
        }

        void PreprocessData(VectorMatrix &user, VectorMatrix &data_item) {
            exact_rank_ins_.PreprocessData(user, data_item);
        };

        std::vector<std::vector<int>> &BuildIndexMergeUser() {
            static std::vector<std::vector<int>> eval_seq_l(n_merge_user_);
            for (int labelID = 0; labelID < n_merge_user_; labelID++) {
                std::vector<int> &eval_l = eval_seq_l[labelID];
                for (int userID = 0; userID < n_user_; userID++) {
                    if (merge_label_l_[userID] == labelID) {
                        eval_l.push_back(userID);
                    }
                }
            }
            return eval_seq_l;
        }

        void BuildIndexLoop(const std::vector<unsigned char> &itvID_l, const int &userID) {
            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                int itvID = itvID_l[itemID];
                disk_write_cache_l_[itemID].Merge(itvID);
            }
        }

        void WriteIndex() {
            //get the number of users in each bucket, assign into the cache_bkt_vec
            assert(disk_write_cache_l_.size() == n_data_item_);

            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                assert(disk_write_cache_l_[itemID].is_assign_);
            }

            assert(disk_retrieval_cache_l_.size() == n_data_item_);
            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                std::pair<unsigned char, unsigned char> itvID_bound_pair = disk_write_cache_l_[itemID].GetRankBound();
                disk_retrieval_cache_l_[itemID] = itvID_bound_pair;
            }

            out_stream_.write((char *) disk_retrieval_cache_l_.data(),
                              (std::streamsize) (n_data_item_ * sizeof(std::pair<unsigned char, unsigned char>)));

            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                disk_write_cache_l_[itemID].Reset();
            }

        }

        void FinishWrite() {
            out_stream_.close();
        }

        inline void RetrievalPreprocess() {
            read_disk_time_ = 0;
            exact_rank_refinement_time_ = 0;
            index_stream_ = std::ifstream(this->index_path_, std::ios::binary | std::ios::in);
            if (!index_stream_) {
                spdlog::error("error in writing index");
            }
        }

        void PreprocessQuery(const double *query_vecs, const int &vec_dim, double *query_write_vecs) {
            exact_rank_ins_.PreprocessQuery(query_vecs, vec_dim, query_write_vecs);
        }

        void GetRank(const std::vector<double> &queryIP_l,
                     const std::vector<int> &rank_lb_l, const std::vector<int> &rank_ub_l,
                     const std::vector<std::pair<double, double>> &queryIPbound_l,
                     const std::vector<int> &itvID_l,
                     const std::vector<bool> &prune_l, const VectorMatrix &user, const VectorMatrix &item) {
            is_compute_l_.assign(n_merge_user_, false);

            //read disk and fine binary search
            n_candidate_ = 0;
            for (int iter_userID = 0; iter_userID < n_user_; iter_userID++) {
                if (prune_l[iter_userID]) {
                    continue;
                }
                int iter_labelID = (int) merge_label_l_[iter_userID];
                if (is_compute_l_[iter_labelID]) {
                    continue;
                }
                read_disk_record_.reset();
                ReadDisk(iter_labelID);
                read_disk_time_ += read_disk_record_.get_elapsed_time_second();
                for (int userID = iter_userID; userID < n_user_; userID++) {
                    if (prune_l[userID]) {
                        continue;
                    }
                    int user_labelID = (int) merge_label_l_[userID];
                    if (user_labelID != iter_labelID) {
                        continue;
                    }
                    assert(0 <= rank_ub_l[userID] && rank_ub_l[userID] <= rank_lb_l[userID] &&
                           rank_lb_l[userID] <= n_data_item_);
                    exact_rank_refinement_record_.reset();

                    const int base_rank = rank_ub_l[userID];
                    const double queryIP = queryIP_l[userID];
                    int loc_rk;
                    if (rank_lb_l[userID] == rank_ub_l[userID]) {
                        loc_rk = 0;
                    } else {
                        const double *user_vecs = user.getVector(userID);

                        item_cand_l_.assign(n_data_item_, false);

                        const int query_itvID_lb = itvID_l[userID];
                        const int query_itvID_ub = itvID_l[userID];
                        assert(query_itvID_ub <= query_itvID_lb);

                        for (int itemID = 0; itemID < n_data_item_; itemID++) {
                            const unsigned char item_itvID_lb = disk_retrieval_cache_l_[itemID].first;
                            const unsigned char item_itvID_ub = disk_retrieval_cache_l_[itemID].second;
                            assert(0 <= item_itvID_ub && item_itvID_ub <= item_itvID_lb &&
                                   item_itvID_lb <= n_data_item_);
                            bool bottom_query = item_itvID_lb < query_itvID_ub;
                            bool top_query = query_itvID_lb < item_itvID_ub;
                            if (bottom_query || top_query) {
                                continue;
                            }
                            item_cand_l_[itemID] = true;
                        }

                        loc_rk = exact_rank_ins_.QueryRankByCandidate(queryIPbound_l[userID], queryIP,
                                                                      user_vecs, userID,
                                                                      item, item_cand_l_);
                    }

                    int rank = base_rank + loc_rk + 1;
                    exact_rank_refinement_time_ += exact_rank_refinement_record_.get_elapsed_time_second();

                    user_topk_cache_l_[n_candidate_] = UserRankElement(userID, rank, queryIP);
                    n_candidate_++;

                }
                is_compute_l_[iter_labelID] = true;
            }

            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_candidate_,
                      std::less());

        }

        inline void ReadDisk(const int &labelID) {
            uint64_t offset = n_data_item_ * labelID;
            std::basic_istream<char>::off_type offset_byte = offset * sizeof(std::pair<unsigned char, unsigned char>);
            index_stream_.seekg(offset_byte, std::ios::beg);
            index_stream_.read((char *) disk_retrieval_cache_l_.data(),
                               (std::streamsize) (n_data_item_ * sizeof(std::pair<unsigned char, unsigned char>)));
        }

        void FinishRetrieval() {
            index_stream_.close();
        }

    };
}
#endif //REVERSE_KRANKS_MERGEINTERVAL_HPP
