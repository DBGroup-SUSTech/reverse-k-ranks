//
// Created by BianZheng on 2022/6/27.
//

#ifndef REVERSE_KRANKS_SSCOMPUTEALL_HPP
#define REVERSE_KRANKS_SSCOMPUTEALL_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "alg/TopkMaxHeap.hpp"
#include "alg/DiskIndex/ComputeAll.hpp"
#include "ScoreSearch.hpp"
#include "alg/RankBoundRefinement/PruneCandidateByBound.hpp"

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

namespace ReverseMIPS::SSComputeAll {

    class Index : public BaseIndex {
        void ResetTimer() {
            inner_product_time_ = 0;
            interval_search_time_ = 0;
            exact_rank_refinement_time_ = 0;
            interval_prune_ratio_ = 0;
        }

    public:
        //for interval search, store in memory
        ScoreSearch interval_ins_;

        //read all instance
        ComputeAll disk_ins_;

        VectorMatrix user_, data_item_;
        int vec_dim_, n_data_item_, n_user_;
        double inner_product_time_, interval_search_time_, exact_rank_refinement_time_;
        TimeRecord inner_product_record_, interval_search_record_;
        double interval_prune_ratio_;

        //temporary retrieval variable
        std::vector<bool> prune_l_;
        std::vector<double> queryIP_l_;
        std::vector<int> rank_lb_l_;
        std::vector<int> rank_ub_l_;
        std::unique_ptr<double[]> query_cache_;

        Index(
                //interval search
                ScoreSearch &interval_ins,
                //disk index
                ComputeAll &disk_ins,
                //general retrieval
                VectorMatrix &user, VectorMatrix &data_item) {
            //interval search
            this->interval_ins_ = std::move(interval_ins);
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
                this->inner_product_time_ += inner_product_record_.get_elapsed_time_second();

                interval_search_record_.reset();
                //count rank bound
                interval_ins_.RankBound(queryIP_l_, rank_lb_l_, rank_ub_l_);
                //prune the bound
                PruneCandidateByBound(rank_lb_l_, rank_ub_l_,
                                      n_user_,
                                      prune_l_, topkLbHeap);

                this->interval_search_time_ += interval_search_record_.get_elapsed_time_second();
                int n_candidate = 0;
                for (int userID = 0; userID < n_user_; userID++) {
                    if (!prune_l_[userID]) {
                        n_candidate++;
                    }
                }
                assert(n_candidate >= topk);
                interval_prune_ratio_ += 1.0 * (n_user_ - n_candidate) / n_user_;
                spdlog::info("finish memory index search n_candidate {} queryID {}", n_candidate, queryID);

                //read disk and fine binary search
                size_t n_compute = 0;
                disk_ins_.GetRank(user_, data_item_, queryIP_l_, prune_l_, n_compute);
                spdlog::info("finish compute rank n_compute {}, queryID {}", n_compute, queryID);

                for (int candID = 0; candID < topk; candID++) {
                    query_heap_l[queryID][candID] = disk_ins_.user_topk_cache_l_[candID];
                }
                assert(query_heap_l[queryID].size() == topk);
            }

            exact_rank_refinement_time_ = disk_ins_.exact_rank_refinement_time_;

            interval_prune_ratio_ /= n_query_item;
            return query_heap_l;
        }

        std::string VariancePerformanceMetricName() override {
            return "queryID, retrieval time, second per query, interval prune ratio";
        }

        std::string VariancePerformanceStatistics(
                const double &retrieval_time, const double &second_per_query, const int &queryID) override {
            char str[256];
            sprintf(str, "%d,%.3f,%.3f,%.3f", queryID, retrieval_time, second_per_query, interval_prune_ratio_);
            return str;
        };

        std::string
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &ms_per_query) override {
            // int topk;
            //double total_time,
            //          inner_product_time, interval_search_time_,
            //          exact_rank_refinement_time_,
            //          interval_prune_ratio_
            //double ms_per_query;
            //unit: second

            char buff[1024];
            sprintf(buff,
                    "top%d retrieval time:\n\ttotal %.3fs\n\tinner product %.3fs, interval search %.3fs, \n\texact rank refinement time %.3fs\n\tinterval prune ratio %.4f\n\tmillion second per query %.3fms",
                    topk, retrieval_time,
                    inner_product_time_, interval_search_time_,
                    exact_rank_refinement_time_,
                    interval_prune_ratio_,
                    ms_per_query);
            std::string str(buff);
            return str;
        }

    };

    const int write_every_ = 1000;
    const int report_batch_every_ = 100;

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    std::unique_ptr<Index> BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const char *index_path,
                                      const int &n_interval) {
        const int n_data_item = data_item.n_vector_;
        const int vec_dim = data_item.vec_dim_;
        const int n_user = user.n_vector_;

        user.vectorNormalize();

        //disk index
        ComputeAll disk_ins(n_user, n_data_item, vec_dim);
        disk_ins.PreprocessData(user, data_item);

        //interval search
        ScoreSearch interval_ins(n_interval, n_user, n_data_item);

        //Compute Score Table
        ComputeScoreTable cst(user, data_item);

        TimeRecord record;
        record.reset();
        std::vector<double> distance_l(n_data_item);
        for (int userID = 0; userID < n_user; userID++) {
            cst.ComputeSortItems(userID, distance_l.data());

            interval_ins.LoopPreprocess(distance_l.data(), userID);

            if (userID % cst.report_every_ == 0) {
                std::cout << "preprocessed " << userID / (0.01 * n_user) << " %, "
                          << record.get_elapsed_time_second() << " s/iter" << " Mem: "
                          << get_current_RSS() / 1000000 << " Mb \n";
                record.reset();
            }
        }
        cst.FinishCompute();

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(
                //interval search
                interval_ins,
                //disk index
                disk_ins,
                //general retrieval
                user, data_item);
        return index_ptr;
    }

}
#endif //REVERSE_KRANKS_SSCOMPUTEALL_HPP
