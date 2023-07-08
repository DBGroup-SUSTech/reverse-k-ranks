//
// Created by BianZheng on 2022/7/25.
//

#ifndef REVERSE_KRANKS_MEASUREMERGERANKBYINTERVAL_HPP
#define REVERSE_KRANKS_MEASUREMERGERANKBYINTERVAL_HPP

#include "alg/TopkMaxHeap.hpp"
#include "alg/SpaceInnerProduct.hpp"
//#include "alg/Cluster/KMeansParallel.hpp"
#include "alg/Cluster/GreedyMergeMinClusterSize.hpp"
#include "alg/RankBoundRefinement/PruneCandidateByBound.hpp"
#include "alg/DiskIndex/ComputeRank/BaseIPBound.hpp"
#include "struct/DistancePair.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/UserRankBound.hpp"
#include "util/TimeMemory.hpp"
#include <cfloat>
#include <memory>
#include <fstream>
#include <spdlog/spdlog.h>
#include <filesystem>

namespace ReverseMIPS::MeasureMergeRankByInterval {

    class MergeRankByInterval {
    public:
        //index variable
        int n_user_, n_data_item_, vec_dim_, n_merge_user_;
        //n_cache_rank_: stores how many intervals for each merged user
        std::vector<uint32_t> merge_label_l_; // n_user, stores which cluster the user belons to
        BaseIPBound exact_rank_ins_;
        const char *index_path_;

        //record time memory
        TimeRecord read_disk_record_, exact_rank_refinement_record_;
        double read_disk_time_, exact_rank_refinement_time_;
        size_t n_compute_lower_bound_, n_compute_upper_bound_, n_total_compute_;
        size_t n_total_user_candidate_;

        //variable in build index
        std::ofstream out_stream_;
        //n_data_item, stores the UserRankBound in the disk, used for build index and retrieval
        std::vector<UserRankBound<int>> disk_write_cache_l_;

        //variable in retrieval
        std::ifstream index_stream_;
        std::vector<bool> is_compute_l_;
        std::vector<std::pair<int, int>> disk_retrieval_cache_l_;
        std::vector<bool> item_cand_l_;

        inline MergeRankByInterval() {}

        inline MergeRankByInterval(const VectorMatrix &user,
                                   const int &n_data_item, const char *index_path,
                                   const int &n_merge_user) {
            this->exact_rank_ins_ = BaseIPBound(n_data_item, user.vec_dim_);;
            this->n_user_ = user.n_vector_;
            this->n_data_item_ = n_data_item;
            this->vec_dim_ = user.vec_dim_;
            this->index_path_ = index_path;
            this->n_merge_user_ = n_merge_user;

            spdlog::info("n_merge_user {}", n_merge_user_);

            this->merge_label_l_.resize(n_user_);
            this->disk_write_cache_l_.resize(n_data_item);
            for (int itemID = 0; itemID < n_data_item; itemID++) {
                this->disk_write_cache_l_[itemID].Reset();
            }
            this->is_compute_l_.resize(n_merge_user_);
            this->disk_retrieval_cache_l_.resize(n_data_item);
            this->item_cand_l_.resize(n_data_item_);
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
            n_total_user_candidate_ = 0;

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
                     const int &n_user_candidate, uint64_t &n_item_candidate) {
            is_compute_l_.assign(n_merge_user_, false);
            n_item_candidate = 0;
            uint64_t user_candidate = 0;

            //read disk and fine binary search
            TimeRecord record;
            record.reset();
            double batch_read_disk_time = 0;
            double batch_decode_time = 0;
            for (int iter_userID = 0; iter_userID < n_user_; iter_userID++) {
                if (prune_l[iter_userID]) {
                    continue;
                }
                int iter_labelID = (int) merge_label_l_[iter_userID];
                assert(0 <= iter_labelID && iter_labelID < n_merge_user_);
                if (is_compute_l_[iter_labelID]) {
                    continue;
                }
                read_disk_record_.reset();
                ReadDisk(iter_labelID);
                const double tmp_read_disk_time = read_disk_record_.get_elapsed_time_second();
                batch_read_disk_time += tmp_read_disk_time;
                read_disk_time_ += tmp_read_disk_time;
                for (int userID = iter_userID; userID < n_user_; userID++) {
                    if (prune_l[userID]) {
                        continue;
                    }
                    int user_labelID = (int) merge_label_l_[userID];
                    assert(0 <= user_labelID && user_labelID < n_merge_user_);
                    if (user_labelID != iter_labelID) {
                        continue;
                    }
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
                        std::pair<int, int> query_rank_bound = std::make_pair(rank_lb_l[userID], rank_ub_l[userID]);

                        item_cand_l_.assign(n_data_item_, false);

                        const int query_rank_lb = rank_lb_l[userID];
                        const int query_rank_ub = rank_ub_l[userID];
                        assert(query_rank_ub <= query_rank_lb);

                        for (int itemID = 0; itemID < n_data_item_; itemID++) {
                            const int item_rank_lb = disk_retrieval_cache_l_[itemID].first;
                            const int item_rank_ub = disk_retrieval_cache_l_[itemID].second;
                            assert(0 <= item_rank_ub && item_rank_ub <= item_rank_lb &&
                                   item_rank_lb <= n_data_item_);
                            bool bottom_query = item_rank_lb < query_rank_ub;
                            bool top_query = query_rank_lb < item_rank_ub;
                            if (bottom_query || top_query) {
                                continue;
                            }
                            item_cand_l_[itemID] = true;
                            n_compute_lower_bound_++;
                            n_compute_upper_bound_++;
                            n_item_candidate++;
                        }

                    }
                    const double tmp_batch_decode_time = exact_rank_refinement_record_.get_elapsed_time_second();
                    batch_decode_time += tmp_batch_decode_time;
                    exact_rank_refinement_time_ += tmp_batch_decode_time;

                    n_total_compute_ += n_data_item_;
                    n_total_user_candidate_++;
                    user_candidate++;

                    if (user_candidate % 2500 == 0) {
                        std::cout << "compute rank " << (double) user_candidate / (0.01 * n_user_candidate)
                                  << " %, "
                                  << "n_total_item_candidate " << n_item_candidate << ", "
                                  << "read_disk_time " << batch_read_disk_time << ", "
                                  << "decode_time " << batch_decode_time << ", "
                                  << record.get_elapsed_time_second() << " s/iter" << " Mem: "
                                  << get_current_RSS() / 1000000 << " Mb \n";
                        batch_read_disk_time = 0;
                        batch_decode_time = 0;
                        record.reset();
                    }
                }
                is_compute_l_[iter_labelID] = true;
            }

        }

        inline void ReadDisk(const size_t &labelID) {
            uint64_t offset = n_data_item_ * labelID;
            uint64_t offset_byte = offset * sizeof(std::pair<int, int>);
            index_stream_.seekg(offset_byte, std::ios::beg);
            index_stream_.read((char *) disk_retrieval_cache_l_.data(),
                               (std::streamsize) (n_data_item_ * sizeof(std::pair<int, int>)));
        }

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
        MergeRankByInterval disk_ins_;

        VectorMatrix user_, data_item_;
        int vec_dim_, n_data_item_, n_user_;
        double inner_product_time_, rank_bound_refinement_time_, read_disk_time_, exact_rank_refinement_time_;
        TimeRecord inner_product_record_, rank_bound_refinement_record_;
        double rank_search_prune_ratio_;
        size_t n_compute_lower_bound_, n_compute_upper_bound_, n_total_compute_;
        size_t n_total_user_candidate_;

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
                MergeRankByInterval &disk_ins,
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

        void Retrieval(const VectorMatrix &query_item, const int &topk, const int &n_eval_query_item,
                       uint64_t *n_item_candidate_l) override {
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
                uint64_t n_compute = 0;
                disk_ins_.GetRank(queryIP_l_, rank_lb_l_, rank_ub_l_, queryIPbound_l_, prune_l_, user_, data_item_,
                                  n_candidate, n_compute);
                n_item_candidate_l[queryID] = n_compute;

                spdlog::info("finish queryID {} n_user_candidate {} n_item_candidate {}", queryID, n_candidate, n_compute);
            }
            disk_ins_.FinishRetrieval();

            exact_rank_refinement_time_ = disk_ins_.exact_rank_refinement_time_ / n_query_item;
            read_disk_time_ = disk_ins_.read_disk_time_ / n_query_item;

            n_compute_lower_bound_ = disk_ins_.n_compute_lower_bound_ / n_query_item;
            n_compute_upper_bound_ = disk_ins_.n_compute_upper_bound_ / n_query_item;
            n_total_compute_ = disk_ins_.n_total_compute_ / n_query_item;
            n_total_user_candidate_ = disk_ins_.n_total_user_candidate_ / n_query_item;

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
                    "top%d retrieval time:\n\ttotal %.3fs\n\tinner product %.3fs, memory index search %.3fs, read disk %.3fs\n\trank search prune ratio %.4f\n\tn_compute_lower_bound %ld, n_compute_upper_bound %ld, n_total_compute %ld\n\tn_total_user_candidate %ld\n\tmillion second per query %.3fms",
                    topk, retrieval_time,
                    inner_product_time_, rank_bound_refinement_time_, read_disk_time_,
                    rank_search_prune_ratio_,
                    n_compute_lower_bound_, n_compute_upper_bound_, n_total_compute_,
                    n_total_user_candidate_,
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
#endif //REVERSE_KRANKS_MEASUREMERGERANKBYINTERVAL_HPP
