//
// Created by bianzheng on 2023/5/26.
//

#ifndef REVERSE_KRANKS_LINEARSCAN_HPP
#define REVERSE_KRANKS_LINEARSCAN_HPP

#include "util/NameTranslation.hpp"

#include "alg/SpaceInnerProduct.hpp"
#include "alg/TopkMaxHeap.hpp"
#include "alg/LinearScanIndex.hpp"

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

namespace ReverseMIPS::LinearScan {

    class Index : public BaseIndex {
        void ResetTimer() {
            total_retrieval_time_ = 0;
            inner_product_time_ = 0;
            refine_user_time_ = 0;

            total_ip_cost_ = 0;
            total_refine_ip_cost_ = 0;
        }

        //read disk
        LinearScanIndex refine_ins_;

        VectorMatrix user_, data_item_;
        int vec_dim_, n_data_item_, n_user_;
        double total_retrieval_time_, inner_product_time_, refine_user_time_;
        TimeRecord total_retrieval_record_, inner_product_record_, refine_user_record_;
        uint64_t total_ip_cost_, total_refine_ip_cost_;
        size_t stop_time_;

    public:

        //temporary retrieval variable
        std::vector<float> queryIP_l_;
        std::vector<UserRankElement> topk_rank_l_;

        Index(
                //disk index
                LinearScanIndex &refine_ins,
                //general retrieval
                const VectorMatrix &user, const VectorMatrix &data_item, const size_t &stop_time
        ) {
            //read disk
            this->refine_ins_ = std::move(refine_ins);

            this->vec_dim_ = user.vec_dim_;
            this->n_user_ = user.n_vector_;
            this->n_data_item_ = data_item.n_vector_;

            std::unique_ptr<float[]> user_ptr = std::make_unique<float[]>((size_t) user.n_vector_ * user.vec_dim_);
            std::memcpy(user_ptr.get(), user.getRawData(), sizeof(float) * (size_t) user.n_vector_ * user.vec_dim_);
            this->user_.init(user_ptr, n_user_, vec_dim_);

            std::unique_ptr<float[]> data_item_ptr = std::make_unique<float[]>(
                    (size_t) data_item.n_vector_ * data_item.vec_dim_);
            std::memcpy(data_item_ptr.get(), data_item.getRawData(),
                        sizeof(float) * (size_t) data_item.n_vector_ * data_item.vec_dim_);
            this->data_item_.init(data_item_ptr, n_data_item_, vec_dim_);

            this->stop_time_ = stop_time;

            //retrieval variable
            this->queryIP_l_.resize(n_user_);
        }

        std::vector<std::vector<UserRankElement>>
        Retrieval(const VectorMatrix &query_item, const int &topk, const int &n_execute_query,
                  std::vector<SingleQueryPerformance> &query_performance_l) override {
            ResetTimer();
            this->topk_rank_l_.resize(topk);

            if (n_execute_query > query_item.n_vector_) {
                spdlog::error("n_execute_query larger than n_query_item, program exit");
                exit(-1);
            }

            if (topk > n_user_) {
                spdlog::error("top-k is too large, program exit");
                exit(-1);
            }

            //coarse binary search
            const int n_query_item = n_execute_query;

            std::vector<std::vector<UserRankElement>> query_heap_l(n_query_item, std::vector<UserRankElement>(topk));

            // for binary search, check the number
            for (int queryID = 0; queryID < n_query_item; queryID++) {
                total_retrieval_record_.reset();
                inner_product_record_.reset();

                const float *query_vecs = query_item.getVector(queryID);
#pragma omp parallel for default(none) shared(query_vecs, queryID)
                for (int userID = 0; userID < n_user_; userID++) {
                    queryIP_l_[userID] = InnerProduct(query_vecs, user_.getVector(userID), vec_dim_);
                }
                const double tmp_queryIP_time = inner_product_record_.get_elapsed_time_second();
                this->inner_product_time_ += tmp_queryIP_time;
                this->total_ip_cost_ += n_user_;

                //read disk and fine binary search
                size_t tmp_refine_ip_cost = 0;
                const double remain_time = stop_time_ - total_retrieval_time_;
                bool is_finish = false;
                refine_user_record_.reset();
                refine_ins_.GetRank(queryIP_l_, topk, remain_time,
                                    topk_rank_l_, tmp_refine_ip_cost, is_finish);
                const double tmp_refine_user_time = refine_user_record_.get_elapsed_time_second();
                total_refine_ip_cost_ += tmp_refine_ip_cost;
                total_ip_cost_ += tmp_refine_ip_cost;
                refine_user_time_ += tmp_refine_user_time;

                for (int candID = 0; candID < topk; candID++) {
                    query_heap_l[queryID][candID] = topk_rank_l_[candID];
                }
                assert(query_heap_l[queryID].size() == topk);

                const double total_time =
                        total_retrieval_record_.get_elapsed_time_second();

                total_retrieval_time_ += total_time;
                spdlog::info(
                        "queryID {}, single_retrieval_time {}s, ip_cost {}, accu_retrieval_time {}s, accu_ip_cost {}, is_finish {}",
                        queryID, total_time, tmp_refine_ip_cost + n_user_,
                        total_retrieval_time_, total_ip_cost_, is_finish);

                const double &memory_index_time = tmp_queryIP_time;
                query_performance_l[queryID] = SingleQueryPerformance(queryID,
                                                                      0, 0, 0,
                                                                      tmp_refine_ip_cost + n_user_, 0,
                                                                      total_time,
                                                                      memory_index_time, 0);
                if (total_retrieval_time_ > stop_time_) {
                    spdlog::info("terminate queryID {}, total retrieval time {}s, total IP cost {}",
                                 queryID, total_retrieval_time_, total_ip_cost_);
                    break;
                }
            }

            return query_heap_l;
        }

        void FinishCompute() override {
            refine_ins_.FinishCompute();
        }

        std::string
        PerformanceStatistics(const int &topk) override {
            // int topk;
            //double total_time,
            //          inner_product_time, coarse_binary_search_time, read_disk_time
            //          fine_binary_search_time;
            //double rank_prune_ratio;
            //unit: second

            char buff[1024];

            sprintf(buff,
                    "top%d retrieval time: total %.3fs\n\tinner product %.3fs, refine user %.3fs\n\ttotal ip cost %ld, total refine ip cost %ld",
                    topk, total_retrieval_time_,
                    inner_product_time_, refine_user_time_,
                    total_ip_cost_, total_refine_ip_cost_);
            std::string str(buff);
            return str;
        }

        uint64_t IndexSizeByte() override {
            return 0;
        }

    };

    std::unique_ptr<Index>
    BuildIndex(const VectorMatrix &data_item, const VectorMatrix &user, const size_t &stop_time) {

        LinearScanIndex refine_ins(user, data_item);

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(refine_ins,
                                                                   user, data_item, stop_time);
        return index_ptr;
    }

}
#endif //REVERSE_KRANKS_LINEARSCAN_HPP
