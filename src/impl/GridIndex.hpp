//
// Created by bianzheng on 2023/3/24.
//

#ifndef REVERSE_KRANKS_GRIDINDEXBATCH_HPP
#define REVERSE_KRANKS_GRIDINDEXBATCH_HPP

#include "alg/Grid.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "alg/SVD.hpp"
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
#include <cfloat>
#include <cassert>
#include <spdlog/spdlog.h>

namespace ReverseMIPS::GridIndex {

    class Index : public BaseIndex {
        void ResetTimer() {
            total_retrieval_time_ = 0;
            query_inner_product_time_ = 0;
            refinement_time_ = 0;
            early_prune_ratio_ = 0;
            total_ip_cost_ = 0;
            total_appr_ip_cost_ = 0;
            n_proc_query_ = 0;
        }

        std::unique_ptr<Grid> ip_bound_ins_;

        VectorMatrix user_, data_item_;
        int vec_dim_, n_data_item_, n_user_;
        size_t stop_time_;
        double total_retrieval_time_, query_inner_product_time_, refinement_time_;
        TimeRecord total_retrieval_record_;
        double n_proc_query_;
        size_t total_ip_cost_, total_appr_ip_cost_;
        double early_prune_ratio_;
    public:

        //temporary retrieval variable
        std::vector<int> item_cand_l_;

        Index(
                //ip_bound_ins
                std::unique_ptr<Grid> &ip_bound_ins,
                //general retrieval
                VectorMatrix &data_item,
                VectorMatrix &user,
                const size_t &stop_time
        ) {
            this->ip_bound_ins_ = std::move(ip_bound_ins);

            this->data_item_ = std::move(data_item);
            this->n_data_item_ = this->data_item_.n_vector_;
            this->user_ = std::move(user);
            this->vec_dim_ = this->user_.vec_dim_;
            this->n_user_ = this->user_.n_vector_;
            this->stop_time_ = stop_time;

            //retrieval variable
            item_cand_l_.resize(n_data_item_);
        }

        std::vector<std::vector<UserRankElement>>
        Retrieval(const VectorMatrix &query_item, const int &topk, const int &n_execute_query,
                  std::vector<SingleQueryPerformance> &query_performance_l) override {
            ResetTimer();

            if (n_execute_query > query_item.n_vector_) {
                spdlog::error("n_execute_query larger than n_query_item, program exit");
                exit(-1);
            }

            if (topk > user_.n_vector_) {
                spdlog::error("top-k is too large, program exit");
                exit(-1);
            }

            const int report_every = 1000000;
            //coarse binary search
            const int n_query_item = n_execute_query;

            std::vector<std::vector<UserRankElement>> query_heap_l(n_query_item, std::vector<UserRankElement>(topk));

            total_retrieval_record_.reset();
            for (int queryID = 0; queryID < n_query_item; queryID++) {
                TimeRecord query_record, query_tmp_record;
                query_record.reset();
                float *query_vecs = query_item.getVector(queryID);

                size_t this_ip_cost = 0;
                size_t this_appr_ip_cost = 0;
                //calculate IP
                query_tmp_record.reset();
                std::vector<float> queryIP_l(n_user_);
                for (int userID = 0; userID < n_user_; userID++) {
                    float *user_vec = user_.getVector(userID);
                    float queryIP = InnerProduct(query_vecs, user_vec, vec_dim_);
                    queryIP_l[userID] = queryIP;
                }
                const double tmp_query_inner_product_time = query_tmp_record.get_elapsed_time_second();
                query_inner_product_time_ += tmp_query_inner_product_time;
                this_ip_cost += n_user_;

                //rank search
                int early_prune_candidate = 0;
                std::vector<UserRankElement> rank_max_heap(topk);
//                std::vector<UserRankElement> &rank_max_heap = query_heap_l[queryID];
#pragma omp parallel for default(none) shared(topk, queryIP_l, this_ip_cost, this_appr_ip_cost, rank_max_heap)
                for (int userID = 0; userID < topk; userID++) {
                    const float *user_vecs = user_.getVector(userID);
                    int rank = GetRank(queryIP_l[userID], n_data_item_ + 1, userID, user_vecs, data_item_,
                                       this_ip_cost, this_appr_ip_cost);

                    assert(rank != -1);
                    rank_max_heap[userID] = UserRankElement(userID, rank, queryIP_l[userID]);
                }

                std::make_heap(rank_max_heap.begin(), rank_max_heap.end(), std::less());

#pragma omp parallel for default(none) shared(topk, rank_max_heap, queryIP_l, this_ip_cost, this_appr_ip_cost, queryID, early_prune_candidate, query_tmp_record)
                for (int userID = topk; userID < n_user_; userID++) {
                    UserRankElement heap_ele = rank_max_heap.front();
                    const float *user_vecs = user_.getVector(userID);
                    int rank = GetRank(queryIP_l[userID], heap_ele.rank_, userID, user_vecs, data_item_,
                                       this_ip_cost, this_appr_ip_cost);

                    if (rank == -1) {
                        early_prune_candidate++;
                        continue;
                    }

#pragma omp critical
                    {
                        UserRankElement element(userID, rank, queryIP_l[userID]);
                        heap_ele = rank_max_heap.front();
                        if (heap_ele > element) {
                            std::pop_heap(rank_max_heap.begin(), rank_max_heap.end(), std::less());
                            rank_max_heap.pop_back();
                            rank_max_heap.push_back(element);
                            std::push_heap(rank_max_heap.begin(), rank_max_heap.end(), std::less());
                        }
                    }

                }
                std::make_heap(rank_max_heap.begin(), rank_max_heap.end(), std::less());
                std::sort_heap(rank_max_heap.begin(), rank_max_heap.end(), std::less());

                for (int i = 0; i < topk; i++) {
                    query_heap_l[queryID][i] = rank_max_heap[i];
                }

                n_proc_query_ += 1;
                const double retrieval_time = query_record.get_elapsed_time_second();
                const double tmp_refinement_time = query_tmp_record.get_elapsed_time_second();
                refinement_time_ += tmp_refinement_time;
                const double current_total_retrieval_time = total_retrieval_record_.get_elapsed_time_second();
                const double prune_ratio = early_prune_candidate * 1.0 / n_user_;
                early_prune_ratio_ += prune_ratio;
                total_ip_cost_ += this_ip_cost;
                total_appr_ip_cost_ += this_appr_ip_cost;


                query_performance_l[queryID] = SingleQueryPerformance(queryID, 0, topk, 0,
                                                                      this_ip_cost, 0,
                                                                      retrieval_time / omp_get_num_procs(), 0, 0);

                spdlog::info(
                        "finish queryID {}, current_total_retrieval_time {:.2f}s, single_query_retrieval_time {:.2f}s, prune_ratio {:.2f}, ip_cost {}, appr_ip_cost {}",
                        queryID, current_total_retrieval_time, retrieval_time, prune_ratio,
                        this_ip_cost, this_appr_ip_cost);
                if (total_retrieval_record_.get_elapsed_time_second() > (double) stop_time_) {
                    spdlog::info("total retrieval time larger than stop time, retrieval exit");
                    total_retrieval_time_ = total_retrieval_record_.get_elapsed_time_second();
                    break;
                }
            }
            early_prune_ratio_ /= n_proc_query_;

            return query_heap_l;
        }

        int GetRank(const float &queryIP, const int &min_rank, const int &userID, const float *user_vecs,
                    const VectorMatrix &item, size_t &total_ip_cost, size_t &total_appr_ip_cost) {
            int rank = 1;
            int n_cand_ = 0;
            const int n_data_item = item.n_vector_;
            std::vector<int> item_cand_l(n_data_item);
            for (int itemID = 0; itemID < n_data_item; itemID++) {
                const float *item_vecs = item.getVector(itemID);
                const float IP_lb = ip_bound_ins_->IPLowerBound(userID, itemID);
                total_appr_ip_cost++;
                assert(IP_lb <= InnerProduct(user_vecs, item_vecs, vec_dim_));
                if (queryIP < IP_lb) {
                    rank++;
                    if (rank > min_rank) {
                        return -1;
                    }
                } else {
                    const float IP_ub = ip_bound_ins_->IPUpperBound(userID, itemID);
                    total_appr_ip_cost++;
                    assert(InnerProduct(user_vecs, item_vecs, vec_dim_) <= IP_ub);
                    if (IP_lb <= queryIP && queryIP <= IP_ub) {
                        item_cand_l[n_cand_] = itemID;
                        n_cand_++;
                    }
                }
            }

            for (int candID = 0; candID < n_cand_; candID++) {
                const int itemID = item_cand_l[candID];
                const float *item_vecs = item.getVector(itemID);
                const float IP = InnerProduct(user_vecs, item_vecs, vec_dim_);
                total_ip_cost++;
                if (IP > queryIP) {
                    rank++;
                    if (rank > min_rank) {
                        return -1;
                    }
                }
            }
            if (rank > min_rank) {
                return -1;
            }
            return rank;
        }

        std::string
        PerformanceStatistics(const int &topk) override {
            // int topk;
            //double total_time,
            //          inner_product_time, inner_product_bound_time
            //double early_prune_ratio_
            //unit: second

            char buff[1024];

            const size_t max_ip_cost = (n_data_item_ + 1) * n_user_ * n_proc_query_;
            const double proportion = (double) total_ip_cost_ / (double) max_ip_cost;

            sprintf(buff,
                    "top%d retrieval time: total %.3fs, n_proc_query %.2f\n\taccumulate query inner product time %.3fs, accumulate refinement time %.3fs\n\tearly prune ratio %.4f, total IP cost %ld, total appr IP cost %ld\n\tmaximum ip cost %ld, proportion %.3f",
                    topk, total_retrieval_time_, n_proc_query_,
                    query_inner_product_time_, refinement_time_,
                    early_prune_ratio_, total_ip_cost_, total_appr_ip_cost_,
                    max_ip_cost, proportion);
            std::string str(buff);
            return str;
        }

        uint64_t IndexSizeByte() override {
            return ip_bound_ins_->IndexSize();
        }


    };

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: float, the distance pair for each user
     */

    std::unique_ptr<Index>
    BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const size_t &stop_time) {
        user.vectorNormalize();
        assert(user.vec_dim_ == data_item.vec_dim_);

        std::unique_ptr<Grid> IPbound_ptr;

        int n_user = user.n_vector_;
        int n_data_item = data_item.n_vector_;
        int vec_dim = user.vec_dim_;

        const int min_codeword = std::floor(std::sqrt(1.0 * 80 * std::sqrt(3 * user.vec_dim_)));
        int n_codeword = 1;
        while (n_codeword < min_codeword) {
            n_codeword = n_codeword << 1;
        }
        spdlog::info("GridIndex min_codeword {}, codeword {}", min_codeword, n_codeword);

        IPbound_ptr = std::make_unique<Grid>(n_user, n_data_item, vec_dim, n_codeword);
        IPbound_ptr->Preprocess(user, data_item);

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(IPbound_ptr, data_item, user, stop_time);
        return index_ptr;
    }

}
#endif //REVERSE_KRANKS_GRIDINDEXBATCH_HPP
