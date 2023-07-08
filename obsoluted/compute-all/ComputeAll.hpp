//
// Created by BianZheng on 2022/5/20.
//

#ifndef REVERSE_KRANKS_COMPUTEALL_HPP
#define REVERSE_KRANKS_COMPUTEALL_HPP

#include "ComputeAllIPBound/BaseIPBound.hpp"
#include "ComputeAllIPBound/FullDim.hpp"
#include "ComputeAllIPBound/FullInt.hpp"
#include "ComputeAllIPBound/FullNorm.hpp"
#include "ComputeAllIPBound/Grid.hpp"
#include "ComputeAllIPBound/ItemPQ.hpp"
#include "ComputeAllIPBound/PartDimPartInt.hpp"
#include "ComputeAllIPBound/PartDimPartNorm.hpp"
#include "ComputeAllIPBound/PartIntPartNorm.hpp"
#include "ComputeAllIPBound/UserItemPQ.hpp"

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

namespace ReverseMIPS::ComputeAll {

    class Index : public BaseIndex {
        void ResetTimer() {
            inner_product_time_ = 0;
            inner_product_bound_time_ = 0;
            prune_ratio_ = 0;
        }

        std::unique_ptr<BaseIPBound> ip_bound_ins_;

        VectorMatrix user_, data_item_;
        int vec_dim_, n_data_item_, n_user_;
        double inner_product_time_, inner_product_bound_time_;
        TimeRecord inner_product_record_, inner_product_bound_record_;
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

        std::vector<std::vector<UserRankElement>> Retrieval(const VectorMatrix &query_item, const int &topk) override {
            ResetTimer();

            if (topk > user_.n_vector_) {
                spdlog::error("top-k is too large, program exit");
                exit(-1);
            }

            //coarse binary search
            const int n_query_item = query_item.n_vector_;

            std::vector<std::vector<UserRankElement>> query_heap_l(n_query_item, std::vector<UserRankElement>(topk));

            for (int queryID = 0; queryID < n_query_item; queryID++) {
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

                std::make_heap(rank_max_heap.begin(), rank_max_heap.end(), std::less());
                std::sort_heap(rank_max_heap.begin(), rank_max_heap.end(), std::less());
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
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &ms_per_query) override {
            // int topk;
            //double total_time,
            //          inner_product_time, inner_product_bound_time
            //double early_prune_ratio_
            //double ms_per_query;
            //unit: second

            char buff[1024];

            sprintf(buff,
                    "top%d retrieval time: total %.3fs\n\tinner product time %.3fs, inner product bound time %.3fs\n\tprune ratio %.4f\n\tmillion second per query %.3fms",
                    topk, retrieval_time,
                    inner_product_time_, inner_product_bound_time_,
                    prune_ratio_,
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
    BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const std::string &bound_name,
               const int &scale, const int &n_codebook, const int &n_codeword, char *parameter_name) {
        user.vectorNormalize();
        assert(user.vec_dim_ == data_item.vec_dim_);

        std::unique_ptr<BaseIPBound> IPbound_ptr;

        int n_user = user.n_vector_;
        int n_data_item = data_item.n_vector_;
        int vec_dim = user.vec_dim_;

        if (bound_name == "CAGrid") {
            const int min_codeword = std::floor(std::sqrt(1.0 * 80 * std::sqrt(3 * user.vec_dim_)));
            int n_cur_codeword = 1;
            while (n_cur_codeword < min_codeword) {
                n_cur_codeword = n_cur_codeword << 1;
            }
            spdlog::info("CAGrid min_codeword {}, codeword {}", min_codeword, n_cur_codeword);
            IPbound_ptr = std::make_unique<Grid>(n_user, n_data_item, vec_dim, n_cur_codeword);
            sprintf(parameter_name, "codeword_%d", n_cur_codeword);

        } else if (bound_name == "CAFullDim") {
            IPbound_ptr = std::make_unique<FullDim>(n_user, n_data_item, vec_dim);

        } else if (bound_name == "CAFullNorm") {
            IPbound_ptr = std::make_unique<FullNorm>(n_user, n_data_item, vec_dim);

        } else if (bound_name == "CAFullInt") {
            spdlog::info("CAFullInt scale {}", scale);
            IPbound_ptr = std::make_unique<FullInt>(n_user, n_data_item, vec_dim, scale);
            sprintf(parameter_name, "scale_%d", scale);

        } else if (bound_name == "CAPartDimPartInt") {
            spdlog::info("CAPartDimPartInt scale {}", scale);
            IPbound_ptr = std::make_unique<PartDimPartInt>(n_user, n_data_item, vec_dim, scale);
            sprintf(parameter_name, "scale_%d", scale);

        } else if (bound_name == "CAPartDimPartNorm") {
            IPbound_ptr = std::make_unique<PartDimPartNorm>(n_user, n_data_item, vec_dim);

        } else if (bound_name == "CAPartIntPartNorm") {
            spdlog::info("CAPartIntPartNorm scale {}", scale);
            IPbound_ptr = std::make_unique<PartIntPartNorm>(n_user, n_data_item, vec_dim, scale);
            sprintf(parameter_name, "scale_%d", scale);

        } else if (bound_name == "CAUserItemPQ") {
            spdlog::info("CAUserItemPQ n_codebook {}, n_codeword {}", n_codebook, n_codeword);
            IPbound_ptr = std::make_unique<CAUserItemPQ>(n_user, n_data_item, vec_dim, n_codebook, n_codeword);
            sprintf(parameter_name, "n_codebook_%d-n_codeword_%d", n_codebook, n_codeword);

        } else if (bound_name == "CAItemPQ") {
            spdlog::info("CAItemPQ n_codebook {}, n_codeword {}", n_codebook, n_codeword);
            IPbound_ptr = std::make_unique<CAItemPQ>(n_user, n_data_item, vec_dim, n_codebook, n_codeword);
            sprintf(parameter_name, "n_codebook_%d-n_codeword_%d", n_codebook, n_codeword);

        } else {
            spdlog::error("not found IPBound name, program exit");
            exit(-1);
        }

        IPbound_ptr->Preprocess(user, data_item);

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(IPbound_ptr, data_item, user);
        return index_ptr;
    }

}
#endif //REVERSE_KRANKS_COMPUTEALL_HPP
