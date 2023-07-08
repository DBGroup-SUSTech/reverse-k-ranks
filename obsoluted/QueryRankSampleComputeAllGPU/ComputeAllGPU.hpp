//
// Created by bianzheng on 2023/3/24.
//

#ifndef REVERSE_KRANKS_COMPUTEALLGPU_HPP
#define REVERSE_KRANKS_COMPUTEALLGPU_HPP

#include "../../src/include/alg/SpaceInnerProduct.hpp"
#include "../../src/include/struct/UserRankElement.hpp"

#ifdef USE_RETRIEVAL_GPU

#include "RefineGPU.hpp"

#else

#include "alg/DiskIndex/RefineByComputation/RefineCPU.hpp"

#endif

#include "../../../../../usr/include/c++/9/memory"
#include "../../../../../usr/lib/gcc/x86_64-linux-gnu/9/include/omp.h"
#include "../../../../../usr/local/include/spdlog/spdlog.h"

namespace ReverseMIPS {

    class ComputeAllGPU {
        //DO two things, first is, given query, compute the IP between the query and user
        //second thing is given query IP and a user, compute the rank of among those item
        int n_data_item_, n_user_, vec_dim_;

        TimeRecord refine_user_record_;

#ifdef USE_RETRIEVAL_GPU
        RefineGPU refine_gpu_;
#else
        RefineCPU refine_cpu_;
#endif

    public:
        std::vector<UserRankElement> user_topk_cache_l_;

        inline ComputeAllGPU() = default;

        inline ComputeAllGPU(const VectorMatrix &user, const VectorMatrix &data_item) {
            const float *user_ptr = user.getRawData();
            const float *data_item_ptr = data_item.getRawData();
            this->n_user_ = user.n_vector_;
            this->n_data_item_ = data_item.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->user_topk_cache_l_.resize(n_user_);

#ifdef USE_RETRIEVAL_GPU
            refine_gpu_ = RefineGPU(user_ptr, data_item_ptr, n_user_, n_data_item_, vec_dim_);
#else
            refine_cpu_ = RefineCPU(user_ptr, data_item_ptr, n_user_, n_data_item_, vec_dim_);
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
                     const std::vector<bool> &prune_l, const std::vector<bool> &result_l,
                     const int &n_remain_result, size_t &n_compute, int &n_refine_user, double &refine_user_time) {

            //read disk and fine binary search
            n_compute = 0;
            n_refine_user = 0;
            if (n_remain_result == 0) {
                return;
            }

            int n_candidate = 0;
            refine_user_record_.reset();
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID] || result_l[userID]) {
                    continue;
                }
                const float queryIP = queryIP_l[userID];
#ifdef USE_RETRIEVAL_GPU
                const int rank = refine_gpu_.RefineRank(queryIP, userID) + 1;
#else
                const int rank = refine_cpu_.RefineRank(queryIP, userID) + 1;
#endif

                n_compute += n_data_item_;
                user_topk_cache_l_[n_candidate] = UserRankElement(userID, rank, queryIP);
                n_candidate++;

            }
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
#endif //REVERSE_KRANKS_COMPUTEALLGPU_HPP
