//
// Created by BianZheng on 2022/7/6.
//

#ifndef REVERSE_K_RANKS_MERGEQUADRATICRANKBOUNDBYBITMAP_HPP
#define REVERSE_K_RANKS_MERGEQUADRATICRANKBOUNDBYBITMAP_HPP

#include "alg/SpaceInnerProduct.hpp"
//#include "alg/Cluster/KMeansParallel.hpp"
#include "alg/Cluster/GreedyMergeMinClusterSize.hpp"
#include "alg/DiskIndex/ComputeRank/BaseIPBound.hpp"
#include "struct/DistancePair.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/UserRankBound.hpp"
#include "util/TimeMemory.hpp"
#include <set>
#include <cfloat>
#include <memory>
#include <spdlog/spdlog.h>
#include <fstream>

namespace ReverseMIPS {

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

        //variable in build index
        std::ofstream out_stream_;
        MQRBBitmapDiskCache disk_cache_;
        MQRBBitmap candidate_bitmap_;

        //variable in retrieval
        std::ifstream index_stream_;
        std::vector<bool> item_cand_l_;
        int n_candidate_;
        std::vector<UserRankElement> user_topk_cache_l_; //n_user, used for sort the element to return the top-k

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
            this->user_topk_cache_l_.resize(n_user_);
            this->known_rank_idx_l_ = std::make_unique<int[]>(n_rank_bound);
            this->item_cand_l_.resize(n_data_item);

        }

        void
        BuildIndexPreprocess(const VectorMatrix &user) {
//            merge_label_l_.assign(n_user_, 0);
            merge_label_l_ = GreedyMergeMinClusterSize::ClusterLabel(user, (int) n_merge_user_);

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

            for (int rank = (int) sample_unit_, idx = 1;
                 rank < topt_ && idx <= n_rank_bound_;
                 rank = (int) ((idx + 1) * (idx + 1) * sample_unit_), idx++) {
                known_rank_idx_l_[idx - 1] = rank;
                assert(idx <= n_rank_bound_);
            }
            known_rank_idx_l_[n_rank_bound_ - 1] = (int) topt_;

//            for (int rankID = 0; rankID < n_rank_bound_; rankID++) {
//                std::cout << known_rank_idx_l_[rankID] << " ";
//            }
//            std::cout << std::endl;

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
            for (int userID = 0; userID < n_user_; userID++) {
                uint32_t labelID = merge_label_l_[userID];
                assert(0 <= labelID && labelID < n_merge_user_);
                eval_seq_l[labelID].push_back(userID);
            }
            return eval_seq_l;
        }

        void BuildIndexLoop(const DistancePair *distance_pair_l, const int &userID) {
#pragma omp parallel for default(none) shared(distance_pair_l)
            for (int crank = 0; crank < n_rank_bound_; crank++) {
                const int low_rank = crank == 0 ? 0 : known_rank_idx_l_[crank - 1];
                const int high_rank = known_rank_idx_l_[crank];
                for (int rank = low_rank; rank < high_rank; rank++) {
                    int itemID = distance_pair_l[rank].ID_;
#pragma omp critical
                    disk_cache_.Add(itemID, crank);
                }
            }
        }

        void WriteIndex() {
            std::vector<bool> cand_l(n_data_item_);
            for (int crank = 0; crank < n_rank_bound_; crank++) {
                cand_l.assign(n_data_item_, false);
                const int cand_count = disk_cache_.Count(crank);
                const int low_rank = crank == 0 ? 0 : known_rank_idx_l_[crank - 1];
                const int high_rank = known_rank_idx_l_[crank];
                const int expect_count = high_rank - low_rank;
                assert(cand_count >= expect_count);
            }
            disk_cache_.WriteDisk(out_stream_);
        }

        void FinishBuildIndex() {
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
                     const std::vector<bool> &prune_l, const VectorMatrix &user, const VectorMatrix &item,
                     size_t& n_compute) {

            //read disk and fine binary search
            n_compute = 0;
            n_candidate_ = 0;
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID]) {
                    continue;
                }
                const int user_labelID = (int) merge_label_l_[userID];
                read_disk_record_.reset();
                std::pair<int, int> rank_bound_pair = std::make_pair(rank_lb_l[userID], rank_ub_l[userID]);
                ReadDisk(rank_bound_pair, user_labelID, item_cand_l_, n_compute);
                read_disk_time_ += read_disk_record_.get_elapsed_time_second();

                assert(0 <= rank_ub_l[userID] && rank_ub_l[userID] <= rank_lb_l[userID] &&
                       rank_lb_l[userID] <= n_data_item_);
                exact_rank_refinement_record_.reset();
                const double queryIP = queryIP_l[userID];
                const int base_rank = rank_ub_l[userID];

                int loc_rk;
                if (rank_lb_l[userID] == rank_ub_l[userID]) {
                    loc_rk = 0;
                } else {
                    const double *user_vecs = user.getVector(userID);
                    loc_rk = exact_rank_ins_.QueryRankByCandidate(queryIPbound_l[userID], queryIP,
                                                                  user_vecs, userID,
                                                                  item, item_cand_l_);
                }
                int rank = base_rank + loc_rk + 1;
                exact_rank_refinement_time_ += exact_rank_refinement_record_.get_elapsed_time_second();

                user_topk_cache_l_[n_candidate_] = UserRankElement(userID, rank, queryIP);
                n_candidate_++;
            }

            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_candidate_,
                      std::less());

        }

        inline void ReadDisk(const std::pair<int, int> &rank_bound_pair, const int &user_labelID,
                             std::vector<bool> &item_cand_l, size_t &n_compute) {
            assert(0 <= rank_bound_pair.second && rank_bound_pair.second <= rank_bound_pair.first &&
                   rank_bound_pair.first <= n_data_item_);
            if (rank_bound_pair.first == rank_bound_pair.second) {
                return;
            }
            if (rank_bound_pair.first >= topt_) {
                item_cand_l.assign(n_data_item_, true);
                return;
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
            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                if (item_cand_l[itemID]) {
                    n_compute++;
                }
            }
        }

        void FinishRetrieval() {
            index_stream_.close();
        }

        std::string IndexInfo() {
            std::string info = "Exact rank method_name: " + exact_rank_ins_.method_name;
            return info;
        }

        void SaveMemoryIndex(const char *index_path) {
            std::ofstream out_stream = std::ofstream(index_path, std::ios::binary | std::ios::out);
            if (!out_stream) {
                spdlog::error("error in write result");
                exit(-1);
            }
            out_stream.write((char *) known_rank_idx_l_.get(), (int64_t) (n_rank_bound_ * sizeof(int)));
            out_stream.write((char *) merge_label_l_.data(), (int64_t) (n_user_ * sizeof(uint32_t)));

            out_stream.close();
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
}
#endif //REVERSE_K_RANKS_MERGEQUADRATICRANKBOUNDBYBITMAP_HPP
