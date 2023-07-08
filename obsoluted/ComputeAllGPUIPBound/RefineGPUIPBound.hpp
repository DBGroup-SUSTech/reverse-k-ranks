//
// Created by bianzheng on 2023/3/28.
//

#ifndef REVERSE_KRANKS_REFINEGPUIPBOUND_HPP
#define REVERSE_KRANKS_REFINEGPUIPBOUND_HPP

#include "../../src/include/alg/SpaceInnerProduct.hpp"
#include "../../src/include/alg/QueryIPBound/FullIntGPU.hpp"

#include "../../../software/anaconda3/include/cublas_v2.h"
#include "../../../../../usr/include/c++/9/algorithm"
#include "../../../../../usr/include/c++/9/vector"
#include "../../../../../usr/include/c++/9/iostream"
#include "../../../software/anaconda3/include/thrust/count.h"
#include "../../../software/anaconda3/include/thrust/inner_product.h"
#include "../../../software/anaconda3/include/thrust/execution_policy.h"

namespace ReverseMIPS {

    struct ReduceRank {

        const thrust::pair<float, float> *itemIP_bound_l_;
        const float *user_vecs_device_;
        const float *data_item_device_ptr_;

        const float queryIP_;
        const int64_t vec_dim_;

        ReduceRank(const thrust::pair<float, float> *itemIP_bound_l, const float *user_vecs_device,
                   const float *data_item_device_ptr, const float queryIP,
                   const int64_t vec_dim)
                : itemIP_bound_l_(itemIP_bound_l), user_vecs_device_(user_vecs_device), data_item_device_ptr_(
                data_item_device_ptr), queryIP_(queryIP), vec_dim_(vec_dim) {}

        __host__ __device__
        float operator()(const int &itemID) const {
            const float IP_lb = itemIP_bound_l_[itemID].first;
            const float IP_ub = itemIP_bound_l_[itemID].second;
            if (queryIP_ < IP_lb) {
                return 1;
            } else if (queryIP_ > IP_ub) {
                return 0;
            } else {
                const float *data_item_ptr = data_item_device_ptr_ + itemID * vec_dim_;
                const float itemIP = thrust::inner_product(thrust::device, data_item_ptr, data_item_ptr + vec_dim_,
                                                           user_vecs_device_, 0.0f);
                return queryIP_ < itemIP ? 1 : 0;
            }
        }
    };

    class RefineGPUIPBound {

        uint64_t n_user_, n_data_item_, vec_dim_;
        const float *user_ptr_;

        float *user_device_ptr_;
        float *item_device_ptr_;
        float *query_vecs_device_ptr_;

        float *queryIP_device_ptr_;
        float *itemIP_device_ptr_;
        thrust::pair<float, float> *itemIP_bound_device_ptr_;
        int *eval_seq_device_ptr_;
        cublasHandle_t handle_;

        FullIntGPU ip_bound_ins_;
        std::vector<float> item_norm_l_;

    public:
        RefineGPUIPBound() = default;

        inline RefineGPUIPBound(const float *user_ptr, const float *data_item_ptr,
                                const int n_user, const int n_data_item, const int vec_dim) {
            this->user_ptr_ = user_ptr;
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->vec_dim_ = vec_dim;

            std::unique_ptr<float[]> data_item_proc_unique_ptr = std::make_unique<float[]>(n_data_item_ * vec_dim_);
            TransformNorm(data_item_ptr, data_item_proc_unique_ptr.get());

            GPUInit(user_ptr, data_item_proc_unique_ptr.get());
        }

        void TransformNorm(const float *data_item_ptr, float *data_item_proc_ptr) {
            item_norm_l_.resize(n_data_item_);

            //compute norm
            std::vector<float> item_norm_l(n_data_item_);
            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                const float *item_vecs = data_item_ptr + itemID * vec_dim_;
                const float norm = std::sqrt(InnerProduct(item_vecs, item_vecs, (int) vec_dim_));
                item_norm_l[itemID] = norm;
            }

            //arg sort by norm size, descending
            std::vector<int> item_idx_l(n_data_item_);
            std::iota(item_idx_l.begin(), item_idx_l.end(), 0);
            std::sort(item_idx_l.begin(), item_idx_l.end(),
                      [&item_norm_l](const int i1, const int i2) { return item_norm_l[i1] > item_norm_l[i2]; });

            //assign the input
            for (int origin_itemID = 0; origin_itemID < n_data_item_; origin_itemID++) {
                const int now_itemID = item_idx_l[origin_itemID];
                assert(0 <= now_itemID && now_itemID < n_data_item_);
                item_norm_l_[origin_itemID] = item_norm_l[now_itemID];
                memcpy(data_item_proc_ptr + origin_itemID * vec_dim_, data_item_ptr + now_itemID * vec_dim_,
                       vec_dim_ * sizeof(float));
            }
        }

        void GPUInit(const float *user_ptr, const float *data_item_ptr) {
            cudaMalloc((void **) &user_device_ptr_, n_user_ * vec_dim_ * sizeof(float));
            cudaMalloc((void **) &item_device_ptr_, n_data_item_ * vec_dim_ * sizeof(float));
            cudaMalloc((void **) &query_vecs_device_ptr_, vec_dim_ * sizeof(float));

            cudaMalloc((void **) &queryIP_device_ptr_, n_user_ * sizeof(float));
            cudaMalloc((void **) &itemIP_device_ptr_, n_data_item_ * sizeof(float));
            cudaMalloc((void **) &itemIP_bound_device_ptr_, n_data_item_ * sizeof(thrust::pair<float, float>));
            cudaMalloc((void **) &eval_seq_device_ptr_, std::max(n_data_item_, n_user_) * sizeof(int));
            cudaCheckErrors("cuda malloc fail");

            cublasCheckErrors(
                    cublasSetMatrix(vec_dim_, n_user_, sizeof(float), (void *) user_ptr, vec_dim_,
                                    (void *) user_device_ptr_,
                                    vec_dim_));
            cublasCheckErrors(
                    cublasSetMatrix(vec_dim_, n_data_item_, sizeof(float), (void *) data_item_ptr, vec_dim_,
                                    (void *) item_device_ptr_,
                                    vec_dim_));

            ip_bound_ins_.init(data_item_ptr, (int64_t) n_data_item_, (int64_t) vec_dim_, 1000);

            const uint64_t n_eval = std::max(n_data_item_, n_user_);
            std::vector<int> eval_seq_l(n_eval);
            std::iota(eval_seq_l.begin(), eval_seq_l.end(), 0);
            CHECK(cudaMemcpy(eval_seq_device_ptr_, eval_seq_l.data(), n_eval * sizeof(int),
                             cudaMemcpyHostToDevice))

            cublasCheckErrors(cublasCreate(&handle_));
        }

        void ComputeQueryIP(const float *query_vecs, float *queryIP_l_) {
            CHECK(cudaMemcpy(query_vecs_device_ptr_, query_vecs, vec_dim_ * sizeof(float), cudaMemcpyHostToDevice));
            float alpha = 1.0;
            float beta = 0.0;
            cublasCheckErrors(
                    cublasSgemv(handle_, CUBLAS_OP_T, vec_dim_, n_user_, &alpha, user_device_ptr_, vec_dim_,
                                query_vecs_device_ptr_, 1, &beta,
                                queryIP_device_ptr_, 1));
            CHECK(cudaMemcpy(queryIP_l_, queryIP_device_ptr_, n_user_ * sizeof(float), cudaMemcpyDeviceToHost));
        }

        [[nodiscard]] int RefineRank(const float queryIP, const int userID, int64_t &n_compute_item, int64_t& refine_ip_cost) const {

            //cauchy inequality, perform in CPU
            const float *rank_ptr = std::lower_bound(item_norm_l_.data(), item_norm_l_.data() + n_data_item_, queryIP,
                                                     [](const float &arrIP, float queryIP) {
                                                         return arrIP >= queryIP;
                                                     });
            const int n_proc_item = rank_ptr - item_norm_l_.data();
            assert(0 <= n_proc_item && n_proc_item <= n_data_item_);
            n_compute_item = n_proc_item;
            refine_ip_cost = n_proc_item;

            //compute the ip bound in GPU
            const float *user_vecs = user_ptr_ + userID * vec_dim_;
            ip_bound_ins_.IPBound(user_vecs, n_proc_item, itemIP_bound_device_ptr_);

            //compute and reduce
            const float *user_device_vecs = user_device_ptr_ + userID * vec_dim_;
            ReduceRank f(itemIP_bound_device_ptr_, user_device_vecs, item_device_ptr_, queryIP, (int64_t) vec_dim_);
            const int base_rank =
                    1 + thrust::count_if(thrust::device, eval_seq_device_ptr_, eval_seq_device_ptr_ + n_proc_item, f);

            return base_rank;
        }

        void FinishCompute() {

            if (user_device_ptr_ != nullptr) {
                cudaFree(user_device_ptr_);
            }
            if (item_device_ptr_ != nullptr) {
                cudaFree(item_device_ptr_);
            }
            if (query_vecs_device_ptr_ != nullptr) {
                cudaFree(query_vecs_device_ptr_);
            }
            if (queryIP_device_ptr_ != nullptr) {
                cudaFree(queryIP_device_ptr_);
            }
            if (itemIP_device_ptr_ != nullptr) {
                cudaFree(itemIP_device_ptr_);
            }
            if (eval_seq_device_ptr_ != nullptr) {
                cudaFree(eval_seq_device_ptr_);
            }
        }
    };

}
#endif //REVERSE_KRANKS_REFINEGPUIPBOUND_HPP
