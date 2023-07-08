//
// Created by BianZheng on 2022/7/25.
//

#ifndef REVERSE_K_RANKS_MEASURETOPTIP_HPP
#define REVERSE_K_RANKS_MEASURETOPTIP_HPP

#include "BatchRun/MeasureDiskIndex/BaseMeasureIndex.hpp"

#include "alg/TopkMaxHeap.hpp"
#include "alg/DiskIndex/ComputeRank/BaseIPBound.hpp"
#include "alg/RankBoundRefinement/PruneCandidateByBound.hpp"

#include <filesystem>

namespace ReverseMIPS::MeasureTopTIP {
    class MeasureTopTIP {

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
        size_t n_compute_lower_bound_, n_compute_upper_bound_, n_total_compute_;
        size_t n_below_topt_, n_between_topt_, n_above_topt_, n_total_user_candidate_;

        inline MeasureTopTIP() = default;

        inline MeasureTopTIP(const int &n_user, const int &n_data_item, const int &vec_dim, const char *index_path,
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

        }

        void PreprocessData(VectorMatrix &user, VectorMatrix &data_item) {
            exact_rank_ins_.PreprocessData(user, data_item);
        };

        void PreprocessQuery(const double *query_vecs, const int &vec_dim, double *query_write_vecs) {
            exact_rank_ins_.PreprocessQuery(query_vecs, vec_dim, query_write_vecs);
        }

        void RetrievalPreprocess() {
            read_disk_time_ = 0;
            exact_rank_time_ = 0;

            n_compute_lower_bound_ = 0;
            n_compute_upper_bound_ = 0;
            n_total_compute_ = 0;

            n_below_topt_ = 0;
            n_between_topt_ = 0;
            n_above_topt_ = 0;
            n_total_user_candidate_ = 0;

            index_stream_ = std::ifstream(this->index_path_, std::ios::binary | std::ios::in);
            if (!index_stream_) {
                spdlog::error("error in writing index");
            }
        }

        void GetRank(const std::vector<double> &queryIP_l,
                     const std::vector<int> &rank_lb_l, const std::vector<int> &rank_ub_l,
                     const std::vector<bool> &prune_l, const VectorMatrix &user, const VectorMatrix &item,
                     const int &n_user_candidate, uint64_t &n_item_candidate) {
            assert(n_user_ == queryIP_l.size());
            assert(n_user_ == rank_lb_l.size() && n_user_ == rank_ub_l.size());
            assert(n_user_ == prune_l.size());

            n_item_candidate = 0;
            uint64_t user_candidate =0 ;
            TimeRecord record;
            record.reset();
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
                    n_compute_lower_bound_ += 0;
                    n_compute_upper_bound_ += 0;
                    n_item_candidate += 0;
                    n_below_topt_++;
                } else if (rank_ub <= topt_ && topt_ <= rank_lb) {
                    n_compute_lower_bound_ += 0;
                    n_compute_upper_bound_ += n_data_item_;
                    n_item_candidate += 0;
                    n_between_topt_++;

                } else if (topt_ < rank_ub) {
                    n_compute_lower_bound_ += n_data_item_;
                    n_compute_upper_bound_ += n_data_item_;
                    n_item_candidate += n_data_item_;
                    n_above_topt_++;
                } else {
                    spdlog::error("have bug in get rank, topt ID IP");
                }

                n_total_user_candidate_++;
                user_candidate++;
                n_total_compute_ += n_data_item_;

                if (user_candidate % 2500 == 0) {
                    std::cout << "compute rank " << (double) user_candidate / (0.01 * n_user_candidate)
                              << " %, "
                              << "n_total_item_candidate " << n_item_candidate << ", "
                              << record.get_elapsed_time_second() << " s/iter" << " Mem: "
                              << get_current_RSS() / 1000000 << " Mb \n";
                    record.reset();
                }
            }

        };

        void FinishRetrieval() {
            index_stream_.close();
        }

        std::string IndexInfo() {
            std::string info = "Exact rank method_name: " + exact_rank_ins_.method_name;
            return info;
        }

    };

    class Index : public BaseMeasureIndex {
        void ResetTimer() {
            inner_product_time_ = 0;
            hash_search_time_ = 0;
            read_disk_time_ = 0;
            exact_rank_time_ = 0;
            hash_prune_ratio_ = 0;
        }

    public:
        //for hash search, store in memory
        RankSearch rank_bound_ins_;
        //read all instance
        MeasureTopTIP disk_ins_;

        VectorMatrix user_, data_item_;
        int vec_dim_, n_data_item_, n_user_;
        double inner_product_time_, hash_search_time_, read_disk_time_, exact_rank_time_;
        TimeRecord inner_product_record_, hash_search_record_;
        size_t n_compute_lower_bound_, n_compute_upper_bound_, n_total_compute_;
        size_t n_below_topt_, n_between_topt_, n_above_topt_, n_total_user_candidate_;
        double hash_prune_ratio_;

        //temporary retrieval variable
        std::vector<std::pair<double, double>> IPbound_l_;
        std::vector<bool> prune_l_;
        std::vector<double> queryIP_l_;
        std::vector<int> rank_lb_l_;
        std::vector<int> rank_ub_l_;
        std::unique_ptr<double[]> query_cache_;

        Index(
                // hash search
                RankSearch &rank_bound_ins,
                //disk index
                MeasureTopTIP &disk_ins,
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
            this->IPbound_l_.resize(n_user_);
            this->prune_l_.resize(n_user_);
            this->queryIP_l_.resize(n_user_);
            this->rank_lb_l_.resize(n_user_);
            this->rank_ub_l_.resize(n_user_);
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
                IPbound_l_.assign(n_user_, std::pair<double, double>(-std::numeric_limits<double>::max(),
                                                                     std::numeric_limits<double>::max()));
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

                //coarse binary search
                hash_search_record_.reset();
                rank_bound_ins_.RankBound(queryIP_l_, rank_lb_l_, rank_ub_l_, IPbound_l_);
                PruneCandidateByBound(rank_lb_l_, rank_ub_l_,
                                      n_user_,
                                      prune_l_, topkLbHeap);
                hash_search_time_ += hash_search_record_.get_elapsed_time_second();
                int n_candidate = 0;
                for (int userID = 0; userID < n_user_; userID++) {
                    if (!prune_l_[userID]) {
                        n_candidate++;
                    }
                }
                assert(n_candidate >= topk);
                hash_prune_ratio_ += 1.0 * (n_user_ - n_candidate) / n_user_;

                //read disk and fine binary search
                uint64_t n_compute = 0;
                disk_ins_.GetRank(queryIP_l_, rank_lb_l_, rank_ub_l_, prune_l_, user_, data_item_, n_candidate,
                                  n_compute);
                n_item_candidate_l[queryID] = n_compute;

                spdlog::info("finish queryID {} n_user_candidate {} n_item_candidate {}", queryID, n_candidate,
                             n_compute);
            }
            disk_ins_.FinishRetrieval();

            exact_rank_time_ = disk_ins_.exact_rank_time_;
            read_disk_time_ = disk_ins_.read_disk_time_;

            n_compute_lower_bound_ = disk_ins_.n_compute_lower_bound_ / n_query_item;
            n_compute_upper_bound_ = disk_ins_.n_compute_upper_bound_ / n_query_item;
            n_total_compute_ = disk_ins_.n_total_compute_ / n_query_item;

            n_below_topt_ = disk_ins_.n_below_topt_ / n_query_item;
            n_between_topt_ = disk_ins_.n_between_topt_ / n_query_item;
            n_above_topt_ = disk_ins_.n_above_topt_ / n_query_item;
            n_total_user_candidate_ = disk_ins_.n_total_user_candidate_ / n_query_item;

            hash_prune_ratio_ /= n_query_item;
        }

        std::string
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &ms_per_query) override {
            // int topk;
            //double total_time,
            //          inner_product_time, hash_search_time_
            //          read_disk_time_, exact_rank_time_,
            //          hash_prune_ratio_
            //double ms_per_query;
            //unit: second

            char buff[1024];
            sprintf(buff,
                    "top%d retrieval time:\n\ttotal %.3fs\n\tinner product %.3fs, memory index search %.3fs\n\tmemory index prune ratio %.4f\n\tn_compute_lower_bound %ld, n_compute_upper_bound %ld, n_total_compute %ld\n\tn_below_topt %ld, n_between_topt %ld, n_above_topt %ld, n_total_candidate %ld\n\tmillion second per query %.3fms",
                    topk, retrieval_time,
                    inner_product_time_, hash_search_time_,
                    hash_prune_ratio_,
                    n_compute_lower_bound_, n_compute_upper_bound_, n_total_compute_,
                    n_below_topt_, n_between_topt_, n_above_topt_, n_total_user_candidate_,
                    ms_per_query);
            std::string str(buff);
            return str;
        }

        std::string BuildIndexStatistics() override {
            uint64_t file_size = std::filesystem::file_size(disk_ins_.index_path_);
            char buffer[512];
            double index_size_gb = 1.0 * file_size / (1024 * 1024 * 1024);
            sprintf(buffer, "Build Index Info: index size %.3f GB", index_size_gb);
            std::string index_size_str(buffer);

            std::string disk_index_str = "Exact rank name: " + disk_ins_.IndexInfo();
            return index_size_str + "\n" + disk_index_str;
        };

    };
}

#endif //REVERSE_K_RANKS_MEASURETOPTIP_HPP
