//
// Created by BianZheng on 2022/9/5.
//

#ifndef REVERSE_K_RANKS_QUERYRANKSAMPLESCOREDISTRIBUTION_HPP
#define REVERSE_K_RANKS_QUERYRANKSAMPLESCOREDISTRIBUTION_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "alg/TopkMaxHeap.hpp"
#include "alg/DiskIndex/ReadAll.hpp"
#include "alg/RankBoundRefinement/PruneCandidateByBound.hpp"
#include "QRSScoreDistribution.hpp"

#include "score_computation/ComputeScoreTable.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/MethodBase.hpp"
#include "util/TimeMemory.hpp"
#include "util/VectorIO.hpp"
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <set>
#include <cassert>
#include <spdlog/spdlog.h>

namespace ReverseMIPS::QueryRankSampleScoreDistribution {

    class Index : public BaseIndex {
        void ResetTimer() {
            inner_product_time_ = 0;
            coarse_binary_search_time_ = 0;
            read_disk_time_ = 0;
            fine_binary_search_time_ = 0;
            rank_prune_ratio_ = 0;
            total_io_cost_ = 0;
        }

        //rank search
        QRSScoreDistribution rank_ins_;
        //read disk
        ReadAll disk_ins_;

        VectorMatrix user_;
        int vec_dim_, n_data_item_, n_user_;
        double inner_product_time_, coarse_binary_search_time_, read_disk_time_, fine_binary_search_time_;
        TimeRecord inner_product_record_, coarse_binary_search_record_;
        TimeRecord query_record_;
        uint64_t total_io_cost_;
        double rank_prune_ratio_;

    public:

        //temporary retrieval variable
        std::vector<bool> prune_l_;
        std::vector<bool> result_l_;
        std::vector<int> refine_seq_l_;
        std::vector<double> queryIP_l_;
        std::vector<int> rank_lb_l_;
        std::vector<int> rank_ub_l_;
        std::unique_ptr<double[]> query_cache_;

        Index(//rank search
                QRSScoreDistribution &rank_ins,
                //disk index
                ReadAll &disk_ins,
                //general retrieval
                VectorMatrix &user, const int &n_data_item
        ) {
            //rank search
            this->rank_ins_ = std::move(rank_ins);
            //read disk
            this->disk_ins_ = std::move(disk_ins);

            this->user_ = std::move(user);
            this->vec_dim_ = this->user_.vec_dim_;
            this->n_user_ = this->user_.n_vector_;
            this->n_data_item_ = n_data_item;

            //retrieval variable
            this->prune_l_.resize(n_user_);
            this->result_l_.resize(n_user_);
            this->refine_seq_l_.resize(n_user_);
            this->queryIP_l_.resize(n_user_);
            this->rank_lb_l_.resize(n_user_);
            this->rank_ub_l_.resize(n_user_);
            this->query_cache_ = std::make_unique<double[]>(vec_dim_);
        }

        std::vector<std::vector<UserRankElement>>
        Retrieval(const VectorMatrix &query_item, const int &topk, const int &n_execute_query,
                  std::vector<SingleQueryPerformance> &query_performance_l) override {
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

            //coarse binary search
            spdlog::info("n_query_item {}", n_execute_query);
            const int n_query_item = n_execute_query;

            std::vector<std::vector<UserRankElement>> query_heap_l(n_query_item);
            for (int qID = 0; qID < n_query_item; qID++) {
                query_heap_l[qID].resize(topk);
            }

            // for binary search, check the number
            for (int queryID = 0; queryID < n_query_item; queryID++) {
                query_record_.reset();
                prune_l_.assign(n_user_, false);
                result_l_.assign(n_user_, false);
                rank_lb_l_.assign(n_user_, n_data_item_);
                rank_ub_l_.assign(n_user_, 0);

                const double *query_item_vec = query_item.getVector(queryID);
                double *query_vecs = query_cache_.get();
                disk_ins_.PreprocessQuery(query_item_vec, vec_dim_, query_vecs);

                //calculate IP
                inner_product_record_.reset();
                for (int userID = 0; userID < n_user_; userID++) {
                    queryIP_l_[userID] = InnerProduct(query_vecs, user_.getVector(userID), vec_dim_);
                }
                const double tmp_inner_product_time = inner_product_record_.get_elapsed_time_second();
                this->inner_product_time_ += tmp_inner_product_time;

                //rank search
                int refine_user_size = 0;
                int n_result_user = 0;
                int n_prune_user = 0;
                coarse_binary_search_record_.reset();
                rank_ins_.RankBound(queryIP_l_, rank_lb_l_, rank_ub_l_);
                PruneCandidateByBound(rank_lb_l_, rank_ub_l_,
                                      n_user_, topk,
                                      refine_seq_l_, refine_user_size,
                                      n_result_user, n_prune_user,
                                      prune_l_, result_l_);
                const double tmp_memory_index_time = coarse_binary_search_record_.get_elapsed_time_second();
                coarse_binary_search_time_ += tmp_memory_index_time;
                rank_prune_ratio_ += 1.0 * (n_user_ - refine_user_size) / n_user_;
                assert(n_result_user + n_prune_user + refine_user_size == n_user_);
                assert(0 <= n_result_user && n_result_user <= topk);

                //read disk and fine binary search
                size_t io_cost = 0;
                size_t ip_cost = 0;
                double read_disk_time = 0;
                double rank_compute_time = 0;
                disk_ins_.GetRank(queryIP_l_, rank_lb_l_, rank_ub_l_,
                                  refine_seq_l_, refine_user_size, topk - n_result_user,
                                  io_cost, ip_cost, read_disk_time, rank_compute_time);
                total_io_cost_ += io_cost;

                int n_cand = 0;
                for (int userID = 0; userID < n_user_; userID++) {
                    if (result_l_[userID]) {
                        query_heap_l[queryID][n_cand] = UserRankElement(userID, rank_lb_l_[userID], queryIP_l_[userID]);
                        n_cand++;
                    }
                }

                for (int candID = n_cand; candID < topk; candID++) {
                    query_heap_l[queryID][candID] = disk_ins_.user_topk_cache_l_[candID - n_cand];
                }
                assert(n_cand + disk_ins_.n_refine_user_ >= topk);
                assert(query_heap_l[queryID].size() == topk);

                const double total_time =
                        query_record_.get_elapsed_time_second();
                const double &memory_index_time = tmp_memory_index_time + tmp_inner_product_time;
                query_performance_l[queryID] = SingleQueryPerformance(queryID,
                                                                      n_prune_user, n_result_user,
                                                                      disk_ins_.n_refine_user_,
                                                                      io_cost, ip_cost,
                                                                      total_time,
                                                                      memory_index_time, read_disk_time,
                                                                      rank_compute_time);
            }
            disk_ins_.FinishRetrieval();

            read_disk_time_ = disk_ins_.read_disk_time_;
            fine_binary_search_time_ = disk_ins_.exact_rank_refinement_time_;

            rank_prune_ratio_ /= n_query_item;

            return query_heap_l;
        }

        std::string VariancePerformanceMetricName() override {
            return "queryID, retrieval time, second per query, rank prune ratio";
        }

        std::string VariancePerformanceStatistics(
                const double &retrieval_time, const double &second_per_query, const int &queryID) override {
            char str[256];
            sprintf(str, "%d,%.3f,%.3f,%.3f", queryID, retrieval_time, second_per_query, rank_prune_ratio_);
            return str;
        };

        std::string
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &ms_per_query) override {
            // int topk;
            //double total_time,
            //          inner_product_time, coarse_binary_search_time, read_disk_time
            //          fine_binary_search_time;
            //double rank_prune_ratio;
            //double ms_per_query;
            //unit: second

            char buff[1024];

            sprintf(buff,
                    "top%d retrieval time: total %.3fs\n\tinner product %.3fs, coarse binary search %.3fs, read disk %.3fs\n\tfine binary search %.3fs\n\ttotal io cost %ld, rank prune ratio %.4f\n\tmillion second per query %.3fms",
                    topk, retrieval_time,
                    inner_product_time_, coarse_binary_search_time_, read_disk_time_,
                    fine_binary_search_time_,
                    total_io_cost_, rank_prune_ratio_,
                    ms_per_query);
            std::string str(buff);
            return str;
        }

    };

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    std::unique_ptr<Index>
    BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const char *index_path, const char *dataset_name,
               const int &n_sample, const int n_sample_score_distribution, const int &n_sample_query,
               const int &sample_topk) {
        const int n_user = user.n_vector_;
        const int n_data_item = data_item.n_vector_;

        user.vectorNormalize();

        //rank search
        QRSScoreDistribution rank_ins(n_sample, n_sample_score_distribution,
                                      n_data_item, n_user,
                                      dataset_name, n_sample_query, sample_topk);

        //disk index
        ReadAll disk_ins(n_user, n_data_item, index_path, n_data_item);
        disk_ins.PreprocessData(user, data_item);

        //Compute Score Table
        ComputeScoreTable cst(user, data_item);

        TimeRecord record;
        record.reset();
        std::vector<DistancePair> distance_l(n_data_item);
        for (int userID = 0; userID < n_user; userID++) {
            cst.ComputeSortItems(userID, distance_l.data());

            rank_ins.LoopPreprocess(distance_l.data(), userID);
            disk_ins.BuildIndexLoop(distance_l.data());

            if (userID % cst.report_every_ == 0) {
                std::cout << "preprocessed " << userID / (0.01 * n_user) << " %, "
                          << record.get_elapsed_time_second() << " s/iter" << " Mem: "
                          << get_current_RSS() / 1000000 << " Mb \n";
                record.reset();
            }
        }
        cst.FinishCompute();
        disk_ins.FinishBuildIndex();

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(rank_ins, disk_ins, user, n_data_item);
        return index_ptr;
    }

}

#endif //REVERSE_K_RANKS_QUERYRANKSAMPLESCOREDISTRIBUTION_HPP
