//
// Created by BianZheng on 2022/6/30.
//

#ifndef REVERSE_K_RANKS_MERGEINTERVALIDBYBITMAP_HPP
#define REVERSE_K_RANKS_MERGEINTERVALIDBYBITMAP_HPP

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

    class Bitmap {
    private:
        std::vector<unsigned char> bit_l_;
        int n_num_, arr_size_;
        uint64_t arr_size_byte_;
        int64_t user_col_size_byte_;
    public:

        Bitmap() = default;

        Bitmap(const int &n_num, const int &check_arr_size, const int &n_interval) {
            this->n_num_ = n_num;
            int arr_size = n_num / 8;
            arr_size += n_num % 8 == 0 ? 0 : 1;
            this->arr_size_ = arr_size;
            assert(check_arr_size == arr_size_);
            bit_l_.resize(arr_size);
            bit_l_.assign(arr_size, 0);
            this->user_col_size_byte_ = arr_size * n_interval * sizeof(unsigned char);
            this->arr_size_byte_ = arr_size * sizeof(unsigned char);
        }

        void Add(const int &num) {
            assert(num < n_num_);
            const int num_offset = num / 8;
            const int bit_offset = num % 8;
            assert(0 <= num_offset && num_offset <= arr_size_);
            assert(0 <= bit_offset && bit_offset <= 8 * sizeof(unsigned char));
            bit_l_[num_offset] |= (1 << bit_offset);
        }

        bool Find(const int &num) {
            const int num_offset = num / 8;
            const int bit_offset = num % 8;
            assert(0 <= num_offset && num_offset <= arr_size_);
            assert(0 <= bit_offset && bit_offset <= 8 * sizeof(unsigned char));
            if ((bit_l_[num_offset]) & (1 << bit_offset)) {//find
                return true;
            } else {
                return false;
            }
        }

        void WriteDisk(std::ofstream &out_stream) {
            out_stream.write((char *) bit_l_.data(),
                             (std::streamsize) (arr_size_byte_));
            bit_l_.assign(arr_size_, 0);
        }

        inline void ReadDisk(std::ifstream &index_stream, const int &labelID, const int &itvID) {
            std::basic_istream<char>::off_type offset_byte = user_col_size_byte_ * labelID +
                                                             arr_size_byte_ * itvID;
            index_stream.seekg(offset_byte, std::ios::beg);
            index_stream.read((char *) bit_l_.data(),
                              (std::streamsize) (arr_size_byte_));
        }
    };

    class MergeIntervalIDByBitmap {
    public:
        //index variable
        int n_user_, n_data_item_, vec_dim_, n_interval_;
        int n_merge_user_, bitmap_size_byte_;
        //n_cache_rank_: stores how many intervals for each merged user
        std::vector<uint32_t> merge_label_l_; // n_user, stores which cluster the user belons to
        PartIntPartNorm exact_rank_ins_;
        const char *index_path_;

        //record time memory
        TimeRecord read_disk_record_, exact_rank_refinement_record_;
        double read_disk_time_, exact_rank_refinement_time_;

        //variable in build index
        std::ofstream out_stream_;
        //n_interval, stores the bitmap in the disk, used for build index
        std::vector<Bitmap> disk_cache_write_l_;

        //variable in retrieval
        std::ifstream index_stream_;
        int n_candidate_;
        std::vector<UserRankElement> user_topk_cache_l_; //n_user, used for sort the element to return the top-k
        Bitmap retrieval_bitmap_;
        std::vector<bool> item_cand_l_;

        inline MergeIntervalIDByBitmap() {}

        inline MergeIntervalIDByBitmap(const VectorMatrix &user,
                                       const char *index_path, const int &n_data_item, const int &n_interval,
                                       const int &n_merge_user, const int &bitmap_size_byte) {
            this->exact_rank_ins_ = PartIntPartNorm(user.n_vector_, n_data_item, user.vec_dim_);
            this->n_user_ = user.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->index_path_ = index_path;
            this->n_data_item_ = n_data_item;

            this->n_interval_ = n_interval;
            this->n_merge_user_ = n_merge_user;
            this->bitmap_size_byte_ = bitmap_size_byte;
            assert(bitmap_size_byte_ == (n_data_item / 8 + (n_data_item % 8 == 0 ? 0 : 1)));

            this->merge_label_l_.resize(n_user_);
            this->disk_cache_write_l_.resize(n_interval);
            for (int itvID = 0; itvID < n_interval; itvID++) {
                this->disk_cache_write_l_[itvID] = Bitmap(n_data_item, bitmap_size_byte_, n_interval_);
            }
            this->user_topk_cache_l_.resize(n_user_);
            this->retrieval_bitmap_ = Bitmap(n_data_item, bitmap_size_byte_, n_interval_);
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
                assert(itvID <= n_interval_);
                if (itvID == n_interval_) {
                    continue;
                }
                disk_cache_write_l_[itvID].Add(itemID);
                assert(disk_cache_write_l_[itvID].Find(itemID));
            }
        }

        void WriteIndex() {
            //get the number of users in each bucket, assign into the cache_bkt_vec
            assert(disk_cache_write_l_.size() == n_interval_);

            for (int itvID = 0; itvID < n_interval_; itvID++) {
                disk_cache_write_l_[itvID].WriteDisk(out_stream_);
            }
        }

        void FinishWrite() {
            out_stream_.close();
            std::ifstream index_stream(index_path_, std::ios::binary);;
            index_stream.seekg(0, std::ios::end);
            std::ios::pos_type ss = index_stream.tellg();
            auto fsize = (size_t) ss;
            assert(fsize == n_merge_user_ * n_interval_ * bitmap_size_byte_ * sizeof(unsigned char));
            index_stream.close();
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
                     const std::vector<bool> &prune_l, const std::vector<int> &itvID_l,
                     const VectorMatrix &user, const VectorMatrix &item, const int queryID) {

            n_candidate_ = 0;
            //read disk and fine binary search
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID]) {
                    continue;
                }
                int base_rank = rank_ub_l[userID];
                const double queryIP = queryIP_l[userID];
                int loc_rk;
                if (rank_lb_l[userID] == rank_ub_l[userID]) {
                    loc_rk = 0;
                } else {
                    const int user_labelID = (int) merge_label_l_[userID];
                    const int user_itvID = itvID_l[userID];
                    assert(0 <= user_itvID);
                    assert(0 <= user_labelID && user_labelID < n_merge_user_);
                    const double *user_vecs = user.getVector(userID);
                    if (user_itvID >= n_interval_) {
                        exact_rank_refinement_record_.reset();
                        loc_rk = ComputeRankByAll(user_vecs, item, queryIP);
                        base_rank = 0;
                        exact_rank_refinement_time_ += exact_rank_refinement_record_.get_elapsed_time_second();

                    } else if (user_itvID < n_interval_) {
                        read_disk_record_.reset();
                        retrieval_bitmap_.ReadDisk(index_stream_, user_labelID, user_itvID);
                        read_disk_time_ += read_disk_record_.get_elapsed_time_second();

                        item_cand_l_.assign(n_data_item_, false);
                        assert(0 <= rank_ub_l[userID] && rank_ub_l[userID] <= rank_lb_l[userID] &&
                               rank_lb_l[userID] <= n_data_item_);

                        for (int itemID = 0; itemID < n_data_item_; itemID++) {
                            item_cand_l_[itemID] = retrieval_bitmap_.Find(itemID);
                        }

                        exact_rank_refinement_record_.reset();
                        loc_rk = exact_rank_ins_.QueryRankByCandidate(queryIPbound_l[userID], queryIP,
                                                                      user_vecs, userID,
                                                                      item, item_cand_l_);
                        exact_rank_refinement_time_ += exact_rank_refinement_record_.get_elapsed_time_second();
                    }

                }

                int rank = base_rank + loc_rk + 1;

                user_topk_cache_l_[n_candidate_] = UserRankElement(userID, rank, queryIP);
                n_candidate_++;
            }

            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_candidate_,
                      std::less());

        }

        int ComputeRankByAll(const double *user_vecs, const VectorMatrix &item,
                             const double &queryIP) const {

            //calculate all the IP, then get the lower bound
            //make different situation by the information
            int rank = 0;
            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                double ip = InnerProduct(item.getVector(itemID), user_vecs, vec_dim_);
                if (queryIP < ip) {
                    rank++;
                }
            }

            return rank;
        }

        void FinishRetrieval() {
            index_stream_.close();
        }

    };
}
#endif //REVERSE_K_RANKS_MERGEINTERVALIDBYBITMAP_HPP
