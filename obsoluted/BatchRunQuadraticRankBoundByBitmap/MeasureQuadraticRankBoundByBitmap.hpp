//
// Created by BianZheng on 2022/7/25.
//

#ifndef REVERSE_K_RANKS_MEASUREQUADRATICRANKBOUNDBYBITMAP_HPP
#define REVERSE_K_RANKS_MEASUREQUADRATICRANKBOUNDBYBITMAP_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "alg/TopkMaxHeap.hpp"
//#include "alg/Cluster/KMeansParallel.hpp"
#include "alg/Cluster/GreedyMergeMinClusterSize.hpp"
#include "alg/DiskIndex/ComputeRank/BaseIPBound.hpp"
#include "../ScoreSample/ScoreSearch.hpp"
#include "alg/RankBoundRefinement/PruneCandidateByBound.hpp"
#include "struct/DistancePair.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/UserRankBound.hpp"
#include "util/TimeMemory.hpp"
#include <cfloat>
#include <memory>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>

namespace ReverseMIPS::MeasureQuadraticRankBoundByBitmap {

    class MQRBBitmap {
    public:
        std::unique_ptr<unsigned char[]> bit_l_;
        int bitmap_size_;

        MQRBBitmap() = default;

        MQRBBitmap(const int &n_data_item) {
            const int bitmap_size = n_data_item / 8 + (n_data_item % 8 == 0 ? 0 : 1);
            this->bitmap_size_ = bitmap_size;
            bit_l_ = std::make_unique<unsigned char[]>(bitmap_size_);
            std::memset(bit_l_.get(), 0, bitmap_size_ * sizeof(unsigned char));
        }

        bool Find(const int &itemID) const {
            const int num_offset = itemID / 8;
            const int bit_offset = itemID % 8;
            assert(0 <= num_offset && num_offset <= bitmap_size_);
            assert(0 <= bit_offset && bit_offset <= 8 * sizeof(unsigned char));
            if ((bit_l_[num_offset]) & (1 << bit_offset)) {//find
                return true;
            } else {
                return false;
            }
        }

        int Count(const int n_data_item) {
            int count = 0;
            for (int itemID = 0; itemID < n_data_item; itemID++) {
                const int num_offset = itemID / 8;
                const int bit_offset = itemID % 8;
                bool is_candidate = bit_l_[num_offset] & (1 << bit_offset);
                if (is_candidate) {
                    count++;
                }
            }
            return count;
        }

        void AssignVector(const int n_data_item, std::vector<bool> &candidate_l) const {
            assert(candidate_l.size() <= bitmap_size_ * 8);
            assert(n_data_item == candidate_l.size());
            int num_offset = 0;
            char bit_offset = 0;
            for (int itemID = 0; itemID < n_data_item; itemID++) {
                const bool is_appear = (bit_l_[num_offset]) & (1 << bit_offset);
                candidate_l[itemID] = is_appear;
                bit_offset++;
                if (bit_offset % 8 == 0) {
                    bit_offset = 0;
                    num_offset++;
                }
            }
        }

    };

    class MQRBBitmapDiskCache {
    private:
        std::unique_ptr<unsigned char[]> bit_l_;
        int n_data_item_, n_rank_bound_, bitmap_size_;
        uint64_t bitmap_size_byte_, user_size_byte_;
    public:

        MQRBBitmapDiskCache() = default;

        MQRBBitmapDiskCache(const int &n_data_item, const int &n_rank_bound) {
            this->n_data_item_ = n_data_item;
            this->n_rank_bound_ = n_rank_bound;
            const int bitmap_size = n_data_item / 8 + (n_data_item % 8 == 0 ? 0 : 1);
            this->bitmap_size_ = bitmap_size;
            this->user_size_byte_ = bitmap_size_ * n_rank_bound * sizeof(unsigned char);
            this->bitmap_size_byte_ = bitmap_size_ * sizeof(unsigned char);
            bit_l_ = std::make_unique<unsigned char[]>(bitmap_size_ * n_rank_bound);
            std::memset(bit_l_.get(), 0, user_size_byte_);
        }

        void Add(const int &itemID, const int &rank_boundID) {
            assert(rank_boundID <= n_rank_bound_);
            assert(itemID < n_data_item_);
            const int num_offset = itemID / 8;
            const int bit_offset = itemID % 8;
            assert(0 <= num_offset && num_offset <= bitmap_size_byte_);
            assert(0 <= bit_offset && bit_offset <= 8 * sizeof(unsigned char));
            bit_l_[rank_boundID * bitmap_size_ + num_offset] |= (1 << bit_offset);
        }

        int Count(const int &rank_boundID) {
            int count = 0;
            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                const int num_offset = itemID / 8;
                const int bit_offset = itemID % 8;
                bool is_candidate = bit_l_[rank_boundID * bitmap_size_ + num_offset] & (1 << bit_offset);
                if (is_candidate) {
                    count++;
                }
            }
            return count;
        }

        void WriteDisk(std::ofstream &out_stream) {
            out_stream.write((char *) bit_l_.get(),
                             (std::streamsize) (user_size_byte_));
            std::memset(bit_l_.get(), 0, user_size_byte_);
        }

        inline void ReadDisk(std::ifstream &index_stream, const int &labelID,
                             const int &start_idx, const int &read_count, MQRBBitmap &bitmap) {
            assert(start_idx + read_count <= n_rank_bound_);
            std::basic_istream<char>::off_type offset_byte = user_size_byte_ * labelID +
                                                             bitmap_size_byte_ * start_idx;
            index_stream.seekg(offset_byte, std::ios::beg);
            index_stream.read((char *) bit_l_.get(),
                              (std::streamsize) (bitmap_size_byte_ * read_count));

            assert(bitmap_size_ == bitmap.bitmap_size_);
            std::memset(bitmap.bit_l_.get(), 0, bitmap.bitmap_size_ * sizeof(unsigned char));

            for (int rank = 0; rank < read_count; rank++) {
                int bitmap_offset = rank * bitmap_size_;
                for (int bitmapID = 0; bitmapID < bitmap_size_; bitmapID++) {
                    bitmap.bit_l_[bitmapID] |= bit_l_[bitmap_offset + bitmapID];
                }
            }

        }

    };

    class MergeQuadraticRankBoundByBitmap {
    public:
        //index variable
        size_t n_user_, n_data_item_, vec_dim_;
        size_t topt_, n_rank_bound_, n_merge_user_, bitmap_size_byte_;
        size_t sample_unit_;
        std::unique_ptr<int[]> known_rank_idx_l_; // n_rank_bound
        std::vector<uint32_t> merge_label_l_; // n_user, stores which cluster the user belons to
        BaseIPBound exact_rank_ins_;
        const char *index_path_;

        //record time memory
        TimeRecord read_disk_record_, exact_rank_refinement_record_;
        double read_disk_time_, exact_rank_refinement_time_;
        size_t n_compute_lower_bound_, n_compute_upper_bound_, n_total_compute_;
        size_t n_total_candidate_;

        //variable in build index
        MQRBBitmapDiskCache disk_cache_;
        MQRBBitmap candidate_bitmap_;

        //variable in retrieval
        std::ifstream index_stream_;
        std::vector<bool> item_cand_l_;

        inline MergeQuadraticRankBoundByBitmap() {}

        inline MergeQuadraticRankBoundByBitmap(const VectorMatrix &user,
                                               const int &n_data_item, const char *index_path,
                                               const int &n_rank_bound, const int &n_merge_user) {
            this->exact_rank_ins_ = BaseIPBound(n_data_item, user.vec_dim_);;
            this->n_user_ = user.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->n_data_item_ = n_data_item;
            this->index_path_ = index_path;
            this->topt_ = n_data_item / 2;
            this->n_rank_bound_ = n_rank_bound;
            this->n_merge_user_ = n_merge_user;
            const uint64_t bitmap_size_byte =
                    (n_data_item / 8 + (n_data_item % 8 == 0 ? 0 : 1)) * sizeof(unsigned char);
            this->bitmap_size_byte_ = (int) bitmap_size_byte;

            this->sample_unit_ = topt_ / (n_rank_bound * n_rank_bound);

            spdlog::info("topt {}, n_rank_bound {}, n_merge_user {}, bitmap_size_byte {}, sample_unit {}",
                         topt_, n_rank_bound_, n_merge_user_, bitmap_size_byte_, sample_unit_);
            if (sample_unit_ <= 0) {
                spdlog::error("n_rank_bound too large, consider a smaller n_rank_bound");
                exit(-1);
            }

            this->merge_label_l_.resize(n_user_);
            this->disk_cache_ = MQRBBitmapDiskCache(n_data_item, n_rank_bound);
            this->candidate_bitmap_ = MQRBBitmap(n_data_item);
            this->known_rank_idx_l_ = std::make_unique<int[]>(n_rank_bound);
            this->item_cand_l_.resize(n_data_item);

        }

        void PreprocessData(VectorMatrix &user, VectorMatrix &data_item) {
            exact_rank_ins_.PreprocessData(user, data_item);
        };

        inline void RetrievalPreprocess() {
            read_disk_time_ = 0;
            exact_rank_refinement_time_ = 0;

            n_compute_lower_bound_ = 0;
            n_compute_upper_bound_ = 0;
            n_total_compute_ = 0;
            n_total_candidate_ = 0;

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
                     const std::vector<bool> &prune_l, const VectorMatrix &user, const VectorMatrix &item) {

            //read disk and fine binary search
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID]) {
                    continue;
                }
                const int user_labelID = (int) merge_label_l_[userID];
                std::pair<int, int> rank_bound_pair = std::make_pair(rank_lb_l[userID], rank_ub_l[userID]);
                std::pair<size_t, size_t> n_compute = ReadDisk(rank_bound_pair, user_labelID, item_cand_l_);
                n_compute_upper_bound_ += n_compute.second;
                n_compute_lower_bound_ += n_compute.first;
                n_total_compute_ += n_data_item_;
                n_total_candidate_++;
            }

        }

        inline std::pair<size_t, size_t> ReadDisk(const std::pair<int, int> &rank_bound_pair, const int &user_labelID,
                                                  std::vector<bool> &item_cand_l) {
            assert(0 <= rank_bound_pair.second && rank_bound_pair.second <= rank_bound_pair.first &&
                   rank_bound_pair.first <= n_data_item_);
            if (rank_bound_pair.first == rank_bound_pair.second) {
                return std::make_pair(0, 0);
            }
            if (rank_bound_pair.first >= topt_) {
                item_cand_l.assign(n_data_item_, true);
                return std::make_pair(n_data_item_, n_data_item_);
            }
            item_cand_l.assign(n_data_item_, false);

            const int rank_lb = rank_bound_pair.first;
            const int rank_ub = rank_bound_pair.second;
            int bucket_lb = std::floor(std::sqrt(1.0 * (rank_lb + 1) / sample_unit_));
            int bucket_ub = std::floor(std::sqrt(1.0 * (rank_ub + 1) / sample_unit_));
            if ((rank_ub + 1) % sample_unit_ == 0 && rank_ub != 0) {
                bucket_ub--;
            }

            if (bucket_lb >= n_rank_bound_) {
                bucket_lb = n_rank_bound_ - 1;
            }
            const int read_count = bucket_lb - bucket_ub + 1;
            assert(0 <= bucket_ub && bucket_ub <= bucket_lb && bucket_lb < n_rank_bound_);
            disk_cache_.ReadDisk(index_stream_, user_labelID,
                                 bucket_ub, read_count,
                                 candidate_bitmap_);

            candidate_bitmap_.AssignVector(n_data_item_, item_cand_l);
            int n_cand = 0;
            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                if (item_cand_l[itemID]) {
                    n_cand++;
                }
            }
            return std::make_pair(n_cand, n_cand);
        }

//        inline std::pair<size_t, size_t> ReadDisk(const std::pair<int, int> &rank_bound_pair, const int &user_labelID,
//                                                  std::vector<bool> &item_cand_l) {
//            assert(0 <= rank_bound_pair.second && rank_bound_pair.second <= rank_bound_pair.first &&
//                   rank_bound_pair.first <= n_data_item_);
//            if (rank_bound_pair.first == rank_bound_pair.second) {
//                return std::make_pair(0, 0);
//            }
//            if (rank_bound_pair.first >= topt_) {
//                item_cand_l.assign(n_data_item_, true);
//                return std::make_pair(n_data_item_, n_data_item_);
//            }
//            item_cand_l.assign(n_data_item_, false);
//
//            return std::make_pair(0, n_data_item_);
//        }

        void FinishRetrieval() {
            index_stream_.close();
        }

        std::string IndexInfo() {
            std::string info = "Exact rank method_name: " + exact_rank_ins_.method_name;
            return info;
        }

        void LoadMemoryIndex(const char *index_path) {
            std::ifstream index_stream = std::ifstream(index_path, std::ios::binary | std::ios::in);
            if (!index_stream) {
                spdlog::error("error in reading index");
                exit(-1);
            }

            known_rank_idx_l_ = std::make_unique<int[]>(n_rank_bound_);
            index_stream.read((char *) known_rank_idx_l_.get(), (int64_t) (sizeof(int) * n_rank_bound_));

            merge_label_l_.resize(n_user_);
            index_stream.read((char *) merge_label_l_.data(), (int64_t) (sizeof(uint32_t) * n_user_));

            index_stream.close();
        }

    };

    class Index : public BaseMeasureIndex {
        void ResetTimer() {
            read_disk_time_ = 0;
            inner_product_time_ = 0;
            rank_bound_refinement_time_ = 0;
            exact_rank_refinement_time_ = 0;
            rank_search_prune_ratio_ = 0;
        }

    public:
        //for rank search, store in memory
        ScoreSearch rank_bound_ins_;
        //read all instance
        MergeQuadraticRankBoundByBitmap disk_ins_;

        VectorMatrix user_, data_item_;
        int vec_dim_, n_data_item_, n_user_;
        double inner_product_time_, rank_bound_refinement_time_, read_disk_time_, exact_rank_refinement_time_;
        size_t n_compute_lower_bound_, n_compute_upper_bound_, n_total_compute_;
        size_t n_total_candidate_;
        TimeRecord inner_product_record_, rank_bound_refinement_record_;
        double rank_search_prune_ratio_;

        //temporary retrieval variable
        std::vector<bool> prune_l_;
        std::vector<std::pair<double, double>> queryIPbound_l_;
        std::vector<double> queryIP_l_;
        std::vector<int> rank_lb_l_;
        std::vector<int> rank_ub_l_;
        std::vector<int> itvID_l_;
        std::unique_ptr<double[]> query_cache_;

        Index(
                // score search
                ScoreSearch &rank_bound_ins,
                //disk index
                MergeQuadraticRankBoundByBitmap &disk_ins,
                //general retrieval
                VectorMatrix &user, VectorMatrix &data_item) {
            //hash search
            this->rank_bound_ins_ = std::move(rank_bound_ins);
            //read disk
            this->disk_ins_ = std::move(disk_ins);
            //general retrieval
            this->n_user_ = user.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->user_ = std::move(user);
            this->n_data_item_ = data_item.n_vector_;
            this->data_item_ = std::move(data_item);
            assert(0 < this->user_.vec_dim_);

            //retrieval variable
            this->prune_l_.resize(n_user_);
            this->queryIPbound_l_.resize(n_user_);
            this->queryIP_l_.resize(n_user_);
            this->rank_lb_l_.resize(n_user_);
            this->rank_ub_l_.resize(n_user_);
            this->itvID_l_.resize(n_user_);
            this->query_cache_ = std::make_unique<double[]>(vec_dim_);

        }

        void Retrieval(const VectorMatrix &query_item, const int &topk, const int &n_eval_query_item) override {
            ResetTimer();
            disk_ins_.RetrievalPreprocess();

            if (topk > user_.n_vector_) {
                spdlog::error("top-k is too large, program exit");
                exit(-1);
            }

            if (n_eval_query_item > query_item.n_vector_) {
                spdlog::info("n_eval_query_item larger than n_query, program exit");
                exit(-1);
            }
            const int n_query_item = n_eval_query_item;

            // store queryIP
            TopkLBHeap topkLbHeap(topk);
            for (int queryID = 0; queryID < n_query_item; queryID++) {
                prune_l_.assign(n_user_, false);
                rank_lb_l_.assign(n_user_, n_data_item_);
                rank_ub_l_.assign(n_user_, 0);
                topkLbHeap.Reset();

                const double *tmp_query_vecs = query_item.getVector(queryID);
                double *query_vecs = query_cache_.get();
                disk_ins_.PreprocessQuery(tmp_query_vecs, vec_dim_, query_vecs);

                //calculate the exact IP
                inner_product_record_.reset();
                for (int userID = 0; userID < n_user_; userID++) {
                    if (prune_l_[userID]) {
                        continue;
                    }
                    queryIP_l_[userID] = InnerProduct(user_.getVector(userID), query_vecs, vec_dim_);
                }
                this->inner_product_time_ += inner_product_record_.get_elapsed_time_second();

                //rank bound refinement
                rank_bound_refinement_record_.reset();
                rank_bound_ins_.RankBound(queryIP_l_, rank_lb_l_, rank_ub_l_, queryIPbound_l_,
                                          itvID_l_);
                PruneCandidateByBound(rank_lb_l_, rank_ub_l_,
                                      n_user_,
                                      prune_l_, topkLbHeap);
                rank_bound_refinement_time_ += rank_bound_refinement_record_.get_elapsed_time_second();
                int n_candidate = 0;
                for (int userID = 0; userID < n_user_; userID++) {
                    if (!prune_l_[userID]) {
                        n_candidate++;
                    }
                }
                assert(n_candidate >= topk);
                rank_search_prune_ratio_ += 1.0 * (n_user_ - n_candidate) / n_user_;

                //read disk and fine binary search
                disk_ins_.GetRank(queryIP_l_, rank_lb_l_, rank_ub_l_, queryIPbound_l_, prune_l_, user_, data_item_);
            }
            disk_ins_.FinishRetrieval();

            exact_rank_refinement_time_ = disk_ins_.exact_rank_refinement_time_;
            read_disk_time_ = disk_ins_.read_disk_time_;

            n_compute_lower_bound_ = disk_ins_.n_compute_lower_bound_ / n_query_item;
            n_compute_upper_bound_ = disk_ins_.n_compute_upper_bound_ / n_query_item;
            n_total_compute_ = disk_ins_.n_total_compute_ / n_query_item;
            n_total_candidate_ = disk_ins_.n_total_candidate_ / n_query_item;

            rank_search_prune_ratio_ /= n_query_item;
        }

        std::string
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &ms_per_query) override {
            // int topk;
            //double total_time,
            //          inner_product_time, rank_bound_refinement_time_
            //          read_disk_time_, exact_rank_refinement_time_,
            //          rank_search_prune_ratio_
            //double ms_per_query;
            //unit: second

            char buff[1024];
            sprintf(buff,
                    "top%d retrieval time:\n\ttotal %.3fs\n\tinner product %.3fs, memory index search %.3fs\n\trank search prune ratio %.4f\n\tn_compute_lower_bound %ld, n_compute_upper_bound %ld, n_total_compute %ld\n\tn_total_candidate %ld\n\tmillion second per query %.3fms",
                    topk, retrieval_time,
                    inner_product_time_, rank_bound_refinement_time_,
                    rank_search_prune_ratio_,
                    n_compute_lower_bound_, n_compute_upper_bound_, n_total_compute_,
                    n_total_candidate_,
                    ms_per_query);
            std::string str(buff);
            return str;
        }

        std::string BuildIndexStatistics() override {
            uint64_t file_size = std::filesystem::file_size(disk_ins_.index_path_);
            char buffer[512];
            double index_size_gb =
                    1.0 * file_size / (1024 * 1024 * 1024);
            sprintf(buffer, "Build Index Info: index size %.3f GB", index_size_gb);
            std::string index_size_str(buffer);

            std::string disk_index_str = "Exact rank name: " + disk_ins_.IndexInfo();
            return index_size_str + "\n" + disk_index_str;
        }

    };
}
#endif //REVERSE_K_RANKS_MEASUREQUADRATICRANKBOUNDBYBITMAP_HPP
