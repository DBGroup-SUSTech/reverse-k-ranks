//
// Created by bianzheng on 2022/5/3.
//

#ifndef REVERSE_KRANKS_COMPRESSTOPTIPBRUTEFROCE_HPP
#define REVERSE_KRANKS_COMPRESSTOPTIPBRUTEFROCE_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "alg/TopkMaxHeap.hpp"
#include "../TopT/TopTIP.hpp"
#include "alg/RankBoundRefinement/PruneCandidateByBound.hpp"
#include "ScoreSearch.hpp"

#include "score_computation/ComputeScoreTable.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/MethodBase.hpp"
#include "util/TimeMemory.hpp"
#include "util/VectorIO.hpp"
#include "util/FileIO.hpp"
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <set>
#include <cassert>
#include <spdlog/spdlog.h>
#include <filesystem>

namespace ReverseMIPS::CompressTopTIPBruteForce {

    class Index : public BaseIndex {
        void ResetTimer() {
            inner_product_time_ = 0;
            memory_index_search_time_ = 0;
            read_disk_time_ = 0;
            exact_rank_time_ = 0;
            total_io_cost_ = 0;
            total_ip_cost_ = 0;
            user_prune_ratio_ = 0;
        }

    public:
        //for hash search, store in memory
        ScoreSearch rank_bound_ins_;
        //read all instance
        TopTIP disk_ins_;

        VectorMatrix user_, data_item_;
        int vec_dim_, n_data_item_, n_user_;
        double inner_product_time_, memory_index_search_time_, read_disk_time_, exact_rank_time_;
        TimeRecord inner_product_record_, memory_index_search_record_;
        uint64_t total_io_cost_, total_ip_cost_;
        double user_prune_ratio_;

        //temporary retrieval variable
        std::vector<bool> prune_l_;
        std::vector<double> queryIP_l_;
        std::vector<int> rank_lb_l_;
        std::vector<int> rank_ub_l_;
        std::unique_ptr<double[]> query_cache_;

        Index(
                // hash search
                ScoreSearch &rank_bound_ins,
                //disk index
                TopTIP &disk_ins,
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
            this->queryIP_l_.resize(n_user_);
            this->rank_lb_l_.resize(n_user_);
            this->rank_ub_l_.resize(n_user_);
            this->query_cache_ = std::make_unique<double[]>(vec_dim_);

        }

        std::vector<std::vector<UserRankElement>>
        Retrieval(const VectorMatrix &query_item, const int &topk, const int &n_execute_query) override {
            ResetTimer();
            disk_ins_.RetrievalPreprocess();

            if (n_execute_query > query_item.n_vector_) {
                spdlog::error("n_execute_query larger than n_query_item, program exit");
                exit(-1);
            }

            if (topk > user_.n_vector_) {
                spdlog::error("top-k is too large, program exit");
                exit(-1);
            }

            spdlog::info("n_query_item {}", n_execute_query);
            const int n_query_item = n_execute_query;
            std::vector<std::vector<UserRankElement>> query_heap_l(n_query_item);
            for (int qID = 0; qID < n_query_item; qID++) {
                query_heap_l[qID].resize(topk);
            }

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
                const double tmp_inner_product_time = inner_product_record_.get_elapsed_time_second();
                this->inner_product_time_ += tmp_inner_product_time;

                //coarse binary search
                memory_index_search_record_.reset();
                rank_bound_ins_.RankBound(queryIP_l_, rank_lb_l_, rank_ub_l_);
                PruneCandidateByBound(rank_lb_l_, rank_ub_l_,
                                      n_user_,
                                      prune_l_, topkLbHeap);
                const double tmp_memory_index_search_time = memory_index_search_record_.get_elapsed_time_second();
                memory_index_search_time_ += tmp_memory_index_search_time;
                int n_user_candidate = 0;
                for (int userID = 0; userID < n_user_; userID++) {
                    if (!prune_l_[userID]) {
                        n_user_candidate++;
                    }
                }
                assert(n_user_candidate >= topk);
                user_prune_ratio_ += 1.0 * (n_user_ - n_user_candidate) / n_user_;

                //read disk and fine binary search
                size_t io_cost = 0;
                size_t ip_cost = 0;
                double read_disk_time = 0;
                double rank_compute_time = 0;
                disk_ins_.GetRank(queryIP_l_, rank_lb_l_, rank_ub_l_, prune_l_, user_, data_item_,
                                  n_user_candidate, io_cost, ip_cost, read_disk_time, rank_compute_time);
                total_io_cost_ += io_cost;
                total_ip_cost_ += ip_cost;

                for (int candID = 0; candID < topk; candID++) {
                    query_heap_l[queryID][candID] = disk_ins_.user_topk_cache_l_[candID];
                }
                assert(query_heap_l[queryID].size() == topk);

                spdlog::info(
                        "finish compute rank queryID {}, n_user_candidate {}, io_cost {}, ip_cost {}, inner product time {:.3f}s, memory index search time {:.3f}s, read disk time {:.3f}s, rank compute time {:.3f}s",
                        queryID, n_user_candidate, io_cost, ip_cost,
                        tmp_inner_product_time, tmp_memory_index_search_time,
                        read_disk_time, rank_compute_time);
            }
            disk_ins_.FinishRetrieval();

            exact_rank_time_ = disk_ins_.exact_rank_time_;
            read_disk_time_ = disk_ins_.read_disk_time_;

            user_prune_ratio_ /= n_query_item;
            return query_heap_l;
        }

        std::string
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &ms_per_query) override {
            // int topk;
            //double total_time,
            //          inner_product_time, memory_index_search_time_
            //          read_disk_time_, exact_rank_time_,
            //          total_io_cost_, total_ip_cost_, user_prune_ratio_
            //double ms_per_query;
            //unit: second

            char buff[1024];
            sprintf(buff,
                    "top%d retrieval time:\n\ttotal %.3fs\n\tinner product %.3fs, memory index search %.3fs\n\tread disk time %.3f, exact rank time %.3fs\n\ttotal io cost %ld, total ip cost %ld, user prune ratio %.4f\n\tmillion second per query %.3fms",
                    topk, retrieval_time,
                    inner_product_time_, memory_index_search_time_,
                    read_disk_time_, exact_rank_time_,
                    total_io_cost_, total_ip_cost_, user_prune_ratio_,
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

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    std::unique_ptr<Index> BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const char *index_path,
                                      const int &n_interval, const uint64_t &index_size_gb) {
        const int n_data_item = data_item.n_vector_;
        const int vec_dim = data_item.vec_dim_;
        const int n_user = user.n_vector_;

        user.vectorNormalize();

        //disk index
        const uint64_t index_size_byte = (uint64_t) index_size_gb * 1024 * 1024 * 1024;
        const uint64_t predict_index_size_byte = (uint64_t) sizeof(double) * n_data_item * n_user;
        const uint64_t topt_big_size = index_size_byte / sizeof(double) / n_user;
        int topt = int(topt_big_size);
        if (index_size_byte >= predict_index_size_byte) {
            spdlog::info("index size larger than the whole score table, use whole table setting");
            topt = n_data_item;
        }
        TopTIP disk_ins(n_user, n_data_item, vec_dim, index_path, topt);
        disk_ins.BuildIndexPreprocess();
        disk_ins.PreprocessData(user, data_item);

        //hash search
        ScoreSearch rank_bound_ins(n_interval, n_user, n_data_item);

        //Compute Score Table
        ComputeScoreTable cst(user, data_item);

        TimeRecord record;
        record.reset();
        std::vector<double> distance_l(n_data_item);
        for (int userID = 0; userID < n_user; userID++) {
            cst.ComputeSortItems(userID, distance_l.data());

            rank_bound_ins.LoopPreprocess(distance_l.data(), userID);
            disk_ins.BuildIndexLoop(distance_l.data());

            if (userID % cst.report_every_ == 0) {
                std::cout << "preprocessed " << userID / (0.01 * n_user) << " %, "
                          << record.get_elapsed_time_second() << " s/iter" << " Mem: "
                          << get_current_RSS() / 1000000 << " Mb \n";
                record.reset();
            }
        }
        cst.FinishCompute();

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(
                //score search
                rank_bound_ins,
                //disk index
                disk_ins,
                //general retrieval
                user, data_item);
        return index_ptr;
    }

}

#endif //REVERSE_KRANKS_COMPRESSTOPTIPBRUTEFROCE_HPP
