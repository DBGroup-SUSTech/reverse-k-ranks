//
// Created by BianZheng on 2022/7/23.
//

#ifndef REVERSE_KRANKS_RANKSAMPLEMEASUREPRUNERATIO_HPP
#define REVERSE_KRANKS_RANKSAMPLEMEASUREPRUNERATIO_HPP

#include "FileIO.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "alg/TopkMaxHeap.hpp"
#include "alg/RankBoundRefinement/RankSearch.hpp"
#include "alg/RankBoundRefinement/PruneCandidateByBound.hpp"

#include "struct/VectorMatrix.hpp"
#include "struct/UserRankElement.hpp"
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

namespace ReverseMIPS::RankSamplePrintRankBound {

    class Index {
        void ResetTimer() {
            inner_product_time_ = 0;
            interval_search_time_ = 0;
            read_disk_time_ = 0;
            exact_rank_refinement_time_ = 0;
            interval_prune_ratio_ = 0;
        }

    public:
        //for interval search, store in memory
        RankSearch memory_index_ins_;

        VectorMatrix user_;
        int vec_dim_, n_data_item_, n_user_;
        double inner_product_time_, interval_search_time_, read_disk_time_, exact_rank_refinement_time_;
        TimeRecord inner_product_record_, interval_search_record_;
        double interval_prune_ratio_;

        //temporary retrieval variable
        std::vector<bool> prune_l_;
        std::vector<double> queryIP_l_;
        std::vector<int> rank_lb_l_;
        std::vector<int> rank_ub_l_;
        std::vector<std::pair<double, double>> queryIPbound_l_;

        Index(
                //interval search
                RankSearch &memory_index_ins,
                //general retrieval
                VectorMatrix &user, const int &n_data_item) {
            //interval search
            this->memory_index_ins_ = std::move(memory_index_ins);
            //general retrieval
            this->n_user_ = user.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->user_ = std::move(user);
            this->n_data_item_ = n_data_item;
            assert(0 < this->user_.vec_dim_);

            //retrieval variable
            this->prune_l_.resize(n_user_);
            this->queryIP_l_.resize(n_user_);
            this->rank_lb_l_.resize(n_user_);
            this->rank_ub_l_.resize(n_user_);
            this->queryIPbound_l_.resize(n_user_);

        }

        void Retrieval(const VectorMatrix &query_item, const int &topk, const int &queryID,
                       std::pair<int, int> *rank_bound_l) {
            ResetTimer();

            if (topk > user_.n_vector_) {
                spdlog::error("top-k is too large, program exit");
                exit(-1);
            }

            const int n_query_item = query_item.n_vector_;

            // store queryIP
            TopkLBHeap topkLbHeap(topk);
            {
                prune_l_.assign(n_user_, false);
                rank_lb_l_.assign(n_user_, n_data_item_);
                rank_ub_l_.assign(n_user_, 0);
                queryIPbound_l_.assign(n_user_, std::pair<double, double>(-std::numeric_limits<double>::max(),
                                                                          std::numeric_limits<double>::max()));
                topkLbHeap.Reset();

                const double *query_vecs = query_item.getVector(queryID);

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
                memory_index_ins_.RankBound(queryIP_l_, rank_lb_l_, rank_ub_l_, queryIPbound_l_);
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
                const double tmp_prune_ratio = 1.0 * (n_user_ - n_candidate) / n_user_;
                for (int userID = 0; userID < n_user_; userID++) {
                    rank_bound_l[userID] = std::make_pair(rank_lb_l_[userID], rank_ub_l_[userID]);
                }
                interval_prune_ratio_ += tmp_prune_ratio;

            }
            interval_prune_ratio_ /= n_query_item;
        }

        std::string
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &ms_per_query) {
            // int topk;
            //double total_time,
            //          inner_product_time, interval_search_time_,
            //          read_disk_time_, exact_rank_refinement_time_,
            //          interval_prune_ratio_
            //double ms_per_query;
            //unit: second

            char buff[1024];
            sprintf(buff,
                    "top%d retrieval time:\n\ttotal %.3fs\n\tinner product %.3fs, interval search %.3fs, \n\tread disk time %.3f, exact rank refinement time %.3fs\n\tinterval prune ratio %.4f\n\tmillion second per query %.3fms",
                    topk, retrieval_time,
                    inner_product_time_, interval_search_time_,
                    read_disk_time_, exact_rank_refinement_time_,
                    interval_prune_ratio_,
                    ms_per_query);
            std::string str(buff);
            return str;
        }

    };

    const int write_every_ = 10;
    const int report_batch_every_ = 100;

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    std::unique_ptr<Index>
    BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const char *rank_sample_index_path) {
        const int n_data_item = data_item.n_vector_;

        //rank search
        RankSearch memory_index_ins(rank_sample_index_path);

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(
                //interval search
                memory_index_ins,
                //general retrieval
                user, n_data_item);
        return index_ptr;
    }

    void MeasurePruneRatio(const char *dataset_name, const char *basic_dir, const char *memeory_index_path,
                           const int &n_sample,
                           const std::vector<int> &queryID_l) {
        //measure prune ratio
        int n_data_item, n_query_item, n_user, vec_dim;
        std::vector<VectorMatrix> data = readData(basic_dir, dataset_name,
                                                  n_data_item, n_query_item, n_user, vec_dim);
        VectorMatrix &user = data[0];
        VectorMatrix &data_item = data[1];
        VectorMatrix &query_item = data[2];
        user.vectorNormalize();

        std::unique_ptr<RankSamplePrintRankBound::Index> index =
                RankSamplePrintRankBound::BuildIndex(data_item, user, memeory_index_path);

        std::string method_name = "RankSamplePrintRankBound";

        PrintRankBound::RetrievalResultAttribution config;
        const int topk = 10;
        TimeRecord record;
        for (const int &queryID: queryID_l) {
            char parameter_name[256];
            sprintf(parameter_name, "n_sample_%d-queryID_%d", n_sample, queryID);

            record.reset();
            std::vector<std::pair<int, int>> rank_bound_l(n_user);
            index->Retrieval(query_item, topk, queryID,
                             rank_bound_l.data());
            const double retrieval_time = record.get_elapsed_time_second();
            const double ms_per_query = retrieval_time / query_item.n_vector_;
            std::string performance_str = index->PerformanceStatistics(topk, retrieval_time, ms_per_query);
            std::cout << performance_str << std::endl;
            config.AddRetrievalInfo(performance_str, topk, retrieval_time, ms_per_query);
            config.WriteRankBound(rank_bound_l, n_user, topk,
                                  dataset_name, method_name.c_str(), parameter_name);
        }

    }


}
#endif //REVERSE_KRANKS_RANKSAMPLEMEASUREPRUNERATIO_HPP
