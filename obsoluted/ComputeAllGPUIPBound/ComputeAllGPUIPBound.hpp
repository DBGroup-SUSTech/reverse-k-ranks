//
// Created by bianzheng on 2023/3/28.
//

#ifndef REVERSE_KRANKS_COMPUTEALLGPUIPBOUND_HPP
#define REVERSE_KRANKS_COMPUTEALLGPUIPBOUND_HPP

#include "../../src/include/alg/SpaceInnerProduct.hpp"
#include "../../src/include/struct/UserRankElement.hpp"

#ifdef USE_RETRIEVAL_GPU

#include "RefineGPUIPBound.hpp"

#else

#include "alg/DiskIndex/RefineByComputation/RefineCPUIPBound.hpp"

#endif

#include "../../../../../usr/include/c++/9/memory"
#include "../../../../../usr/lib/gcc/x86_64-linux-gnu/9/include/omp.h"
#include "../../../../../usr/local/include/spdlog/spdlog.h"

namespace ReverseMIPS {

    class ComputeAllGPUIPBound {
        //DO two things, first is, given query, compute the IP between the query and user
        //second thing is given query IP and a user, compute the rank of among those item
        int n_data_item_, n_user_, vec_dim_;

        TimeRecord refine_user_record_;

#ifdef USE_RETRIEVAL_GPU
        RefineGPUIPBound refine_gpu_;
#else
        RefineCPUIPBound refine_cpu_;
#endif

    public:
        std::vector<UserRankElement> user_topk_cache_l_;

        inline ComputeAllGPUIPBound() = default;

        inline ComputeAllGPUIPBound(const VectorMatrix &user, const VectorMatrix &data_item) {
            const float *user_ptr = user.getRawData();
            const float *data_item_ptr = data_item.getRawData();
            this->n_user_ = user.n_vector_;
            this->n_data_item_ = data_item.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->user_topk_cache_l_.resize(n_user_);

#ifdef USE_RETRIEVAL_GPU
            refine_gpu_ = RefineGPUIPBound(user_ptr, data_item_ptr, n_user_, n_data_item_, vec_dim_);
#else
            refine_cpu_ = RefineCPUIPBound(user_ptr, data_item_ptr, n_user_, n_data_item_, vec_dim_);
#endif
        }

        void ComputeQueryIP(const float *query_vecs, float *queryIP_l_) {
#ifdef USE_RETRIEVAL_GPU
            refine_gpu_.ComputeQueryIP(query_vecs, queryIP_l_);
#else
            refine_cpu_.ComputeQueryIP(query_vecs, queryIP_l_);
#endif
        }

        void GetRank(const VectorMatrix &user, const VectorMatrix &data_item,
                     const std::vector<float> &queryIP_l,
                     const std::vector<char> &prune_l, const std::vector<char> &result_l,
                     const int &n_remain_result, size_t &refine_ip_cost, int &n_refine_user, int64_t &n_compute_item,
                     double &refine_user_time) {

            //read disk and fine binary search
            refine_ip_cost = 0;
            n_refine_user = 0;
            n_compute_item = 0;
            if (n_remain_result == 0) {
                return;
            }

            int n_candidate = 0;
            refine_user_record_.reset();
#ifdef USE_RETRIEVAL_GPU
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID] || result_l[userID]) {
                    continue;
                }
                const float queryIP = queryIP_l[userID];

                int64_t tmp_n_compute_item = 0;
                int64_t tmp_refine_ip_cost = 0;
                const int rank = refine_gpu_.RefineRank(queryIP, userID, tmp_n_compute_item, tmp_refine_ip_cost) + 1;

                n_compute_item += tmp_n_compute_item;
                refine_ip_cost += tmp_refine_ip_cost;
                user_topk_cache_l_[n_candidate] = UserRankElement(userID, rank, queryIP);
                n_candidate++;

            }
#else
#pragma omp parallel for default(none) shared(prune_l, result_l, queryIP_l, n_candidate, n_compute_item, refine_ip_cost)
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID] || result_l[userID]) {
                    continue;
                }
                const float queryIP = queryIP_l[userID];
                int64_t tmp_n_compute_item = 0;
                int64_t tmp_ip_cost = 0;
                const int rank = refine_cpu_.RefineRank(queryIP, userID, tmp_n_compute_item, tmp_ip_cost) + 1;

#pragma omp critical
                {
                    refine_ip_cost += tmp_ip_cost;
                    n_compute_item += tmp_n_compute_item;
                    user_topk_cache_l_[n_candidate] = UserRankElement(userID, rank, queryIP);
                    n_candidate++;
                };

            }
#endif
            refine_user_time += refine_user_record_.get_elapsed_time_second();

            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_candidate,
                      std::less());
            n_refine_user = n_candidate;
        }

        void FinishCompute() {

#ifdef USE_RETRIEVAL_GPU
            refine_gpu_.FinishCompute();
#else
            refine_cpu_.FinishCompute();
#endif
        }

    };
}
#endif //REVERSE_KRANKS_COMPUTEALLGPUIPBOUND_HPP
