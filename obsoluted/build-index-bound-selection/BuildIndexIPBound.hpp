//
// Created by BianZheng on 2022/5/20.
//

#ifndef REVERSE_KRANKS_COMPUTEALL_HPP
#define REVERSE_KRANKS_COMPUTEALL_HPP

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

namespace ReverseMIPS::BuildIndexIPBound {

    class Index : public BaseIndex {
        void ResetTimer() {
            total_retrieval_time_ = 0;
            inner_product_time_ = 0;
            inner_product_bound_time_ = 0;
            prune_ratio_ = 0;
        }

        std::unique_ptr<BaseIPBound> ip_bound_ins_;

        VectorMatrix user_, data_item_;
        int vec_dim_, n_data_item_, n_user_;
        double total_retrieval_time_, inner_product_time_, inner_product_bound_time_;
        TimeRecord total_retrieval_record_, inner_product_record_, inner_product_bound_record_;
        double prune_ratio_;
    public:

        //temporary retrieval variable
        std::vector<double> queryIP_l_;
        std::vector<int> item_cand_l_;
        std::unique_ptr<double[]> query_ptr_;
        std::unique_ptr<std::pair<double, double>[]> IPbound_l_; //n_data_item, store the IP bound of all item

        Index(
                //ip_bound_ins
                std::unique_ptr<BaseIPBound> &ip_bound_ins,
                //general retrieval
                VectorMatrix &data_item,
                VectorMatrix &user
        ) {
            this->ip_bound_ins_ = std::move(ip_bound_ins);

            this->data_item_ = std::move(data_item);
            this->n_data_item_ = this->data_item_.n_vector_;
            this->user_ = std::move(user);
            this->vec_dim_ = this->user_.vec_dim_;
            this->n_user_ = this->user_.n_vector_;

            //retrieval variable
            queryIP_l_.resize(n_user_);
            item_cand_l_.resize(n_data_item_);
            query_ptr_ = std::make_unique<double[]>(vec_dim_);
            IPbound_l_ = std::make_unique<std::pair<double, double>[]>(n_data_item_);
        }

        std::vector<std::vector<UserRankElement>>
        Retrieval(const VectorMatrix &query_item, const int &topk, const int &n_execute_query,
                  std::vector<SingleQueryPerformance> &query_performance_l) override {
            ResetTimer();

            if (topk > user_.n_vector_) {
                spdlog::error("top-k is too large, program exit");
                exit(-1);
            }

            //coarse binary search
            const int n_query_item = query_item.n_vector_;

            std::vector<std::vector<UserRankElement>> query_heap_l(n_query_item, std::vector<UserRankElement>(topk));

            for (int queryID = 0; queryID < n_query_item; queryID++) {
                total_retrieval_record_.reset();
                double *query_vecs = query_ptr_.get();
                ip_bound_ins_->PreprocessQuery(query_item.getVector(queryID), vec_dim_, query_vecs);

                //calculate IP
                inner_product_record_.reset();
                for (int userID = 0; userID < n_user_; userID++) {
                    double *user_vec = user_.getVector(userID);
                    double queryIP = InnerProduct(query_vecs, user_vec, vec_dim_);
                    queryIP_l_[userID] = queryIP;
                }
                inner_product_time_ += inner_product_record_.get_elapsed_time_second();

                //rank search
                inner_product_bound_record_.reset();
                std::vector<UserRankElement> &rank_max_heap = query_heap_l[queryID];
                for (int userID = 0; userID < topk; userID++) {
                    const double *user_vecs = user_.getVector(userID);
                    int total_n_prune = 0;
                    int rank = GetRank(queryIP_l_[userID], userID, user_vecs, data_item_, total_n_prune);
                    prune_ratio_ += total_n_prune * 1.0 / n_data_item_;

                    rank_max_heap[userID] = UserRankElement(userID, rank, queryIP_l_[userID]);
                }

                std::make_heap(rank_max_heap.begin(), rank_max_heap.end(), std::less());

                UserRankElement heap_ele = rank_max_heap.front();
                for (int userID = topk; userID < n_user_; userID++) {
                    const double *user_vecs = user_.getVector(userID);
                    int total_n_prune = 0;
                    int rank = GetRank(queryIP_l_[userID], userID, user_vecs, data_item_, total_n_prune);
                    prune_ratio_ += total_n_prune * 1.0 / n_data_item_;

                    UserRankElement element(userID, rank, queryIP_l_[userID]);
                    if (heap_ele > element) {
                        std::pop_heap(rank_max_heap.begin(), rank_max_heap.end(), std::less());
                        rank_max_heap.pop_back();
                        rank_max_heap.push_back(element);
                        std::push_heap(rank_max_heap.begin(), rank_max_heap.end(), std::less());
                        heap_ele = rank_max_heap.front();
                    }

                }

                std::sort(rank_max_heap.begin(), rank_max_heap.end(), std::less());
//                std::make_heap(rank_max_heap.begin(), rank_max_heap.end(), std::less());
//                std::sort_heap(rank_max_heap.begin(), rank_max_heap.end(), std::less());
                total_retrieval_time_ += total_retrieval_record_.get_elapsed_time_second();
                inner_product_bound_time_ += inner_product_bound_record_.get_elapsed_time_second();
            }
            prune_ratio_ = prune_ratio_ / (n_user_ * n_query_item);

            return query_heap_l;
        }

        int GetRank(const double &queryIP, const int &userID, const double *user_vecs,
                    const VectorMatrix &item, int &n_prune) {
            int rank = 1;

            std::iota(item_cand_l_.begin(), item_cand_l_.end(), 0);

            ip_bound_ins_->IPBound(user_vecs, userID, item_cand_l_, item, IPbound_l_.get());

            for (const int &itemID: item_cand_l_) {
                assert(0 <= itemID && itemID < n_data_item_);
                const double *item_vecs = item.getVector(itemID);
                const std::pair<double, double> IP_pair = IPbound_l_[itemID];

                assert(IP_pair.first <= InnerProduct(user_vecs, item_vecs, vec_dim_) &&
                       InnerProduct(user_vecs, item_vecs, vec_dim_) <= IP_pair.second);
                const double IP_lb = IP_pair.first;
                const double IP_ub = IP_pair.second;
                if (queryIP <= IP_lb) {
                    rank++;
                    n_prune++;
                } else if (IP_lb <= queryIP && queryIP <= IP_ub) {
                    double IP = InnerProduct(user_vecs, item_vecs, vec_dim_);
                    if (queryIP <= IP) {
                        rank++;
                    }
                } else { // IP_ub <= queryIP
                    n_prune++;
                }
            }
            return rank;
        }

        std::string
        PerformanceStatistics(const int &topk) override {
            // int topk;
            //double total_time,
            //          inner_product_time, inner_product_bound_time
            //double early_prune_ratio_
            //double ms_per_query;
            //unit: second

            char buff[1024];

            sprintf(buff,
                    "top%d retrieval time: total %.3fs\n\tinner product time %.3fs, inner product bound time %.3fs\n\tprune ratio %.4f",
                    topk, total_retrieval_time_,
                    inner_product_time_, inner_product_bound_time_,
                    prune_ratio_);
            std::string str(buff);
            return str;
        }

    };

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    std::unique_ptr<Index>
    BuildIndex(VectorMatrix &data_item, VectorMatrix &user, std::unique_ptr<BaseIPBound> &IPbound_ptr,
               std::uint64_t &bucket_size_var) {
        user.vectorNormalize();
        IPbound_ptr->Preprocess(user, data_item);

        const int n_user = user.n_vector_;
        const int n_data_item = data_item.n_vector_;
        const int vec_dim = user.vec_dim_;

//        const int n_bucket = 20;
//
//        std::vector<std::pair<double, double>> ip_bound_l(n_data_item);
//        std::vector<int> bucketID_l(n_data_item);
//        std::vector<int> bucket_n_item_l(n_bucket);
//
//        uint64_t bucket_size_var_sum = 0;
//
//        double avg_bucket_size = 1.0 * n_data_item / n_bucket;
//
//        for (int userID = 0; userID < n_user; userID++) {
//            const double *user_vecs = user.getVector(userID);
//
//            double max_val = -DBL_MAX;
//            double min_val = DBL_MAX;
//#pragma omp parallel for default(none) shared(n_data_item, data_item, IPbound_ptr, user_vecs, userID, ip_bound_l, min_val, max_val)
//            for (int itemID = 0; itemID < n_data_item; itemID++) {
//                const double *item_vecs = data_item.getVector(itemID);
//                std::pair<double, double> ip_bound_pair = IPbound_ptr->IPBound(user_vecs, userID, item_vecs, itemID);
//                ip_bound_l[itemID] = ip_bound_pair;
//#pragma omp critical
//                {
//                    min_val = std::min(min_val, ip_bound_pair.first);
//                    max_val = std::max(max_val, ip_bound_pair.second);
//                }
//            }
//            min_val -= 0.01;
//            max_val += 0.01;
//
//            bucket_n_item_l.assign(n_bucket, 0);
//
//            const double itv_distance = (max_val - min_val) / n_bucket;
//#pragma omp parallel for default(none) shared(n_data_item, ip_bound_l, min_val, itv_distance, bucketID_l, bucket_n_item_l, data_item, user_vecs, vec_dim)
//            for (int itemID = 0; itemID < n_data_item; itemID++) {
//                std::pair<double, double> ip_bound_pair = ip_bound_l[itemID];
//                const int lb_bktID = std::floor((ip_bound_pair.first - min_val) / itv_distance);
//                const int ub_bktID = std::floor((ip_bound_pair.second - min_val) / itv_distance);
//                assert(0 <= lb_bktID && lb_bktID <= ub_bktID && ub_bktID < n_bucket);
//                if (lb_bktID == ub_bktID) {
//                    bucketID_l[itemID] = lb_bktID;
//#pragma omp critical
//                    bucket_n_item_l[lb_bktID]++;
//                } else { //lb_bktID != ub_bktID
//                    const double *item_vecs = data_item.getVector(itemID);
//                    const double queryIP = InnerProduct(user_vecs, item_vecs, vec_dim);
//                    const int bktID = std::floor((queryIP - min_val) / itv_distance);
//                    assert(0 <= bktID && bktID < n_bucket);
//                    bucketID_l[itemID] = bktID;
//#pragma omp critical
//                    bucket_n_item_l[bktID]++;
//                }
//            }
//
//            for (int bucketID = 0; bucketID < n_bucket; bucketID++) {
//                bucket_size_var_sum +=
//                        (bucket_n_item_l[bucketID] - avg_bucket_size) * (bucket_n_item_l[bucketID] - avg_bucket_size);
//            }
//
//        }
//
//        bucket_size_var = bucket_size_var_sum / n_user / n_bucket;
        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(IPbound_ptr, data_item, user);
        return index_ptr;
    }

}
#endif //REVERSE_KRANKS_COMPUTEALL_HPP
