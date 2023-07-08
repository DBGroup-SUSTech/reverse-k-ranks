//
// Created by bianzheng on 2023/3/24.
//

#ifndef REVERSE_KRANKS_REFINEGPU_HPP
#define REVERSE_KRANKS_REFINEGPU_HPP

#include <cublas_v2.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace ReverseMIPS {

// error check macros
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// for CUBLAS V2 API
#define cublasCheckErrors(fn) \
    do { \
        cublasStatus_t __err = fn; \
        if (__err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "Fatal cublas error: %d (at %s:%d)\n", \
                (int)(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

    struct count_bigger {
        float queryIP_;

        count_bigger(const float queryIP) { queryIP_ = queryIP; }

        __host__ __device__
        bool operator()(const float x) const {
            return x > queryIP_;
        }
    };

    class RefineGPU {

        uint64_t n_user_, n_data_item_, vec_dim_;
        float *user_device_ptr_;
        float *item_device_ptr_;
        float *query_vecs_device_ptr_;

        float *queryIP_device_ptr_;
        float *itemIP_device_ptr_;
        cublasHandle_t handle_;
    public:
        RefineGPU() = default;

        inline RefineGPU(const float *user_ptr, const float *data_item_ptr,
                         const uint64_t n_user, const uint64_t n_data_item, const uint64_t vec_dim) {
            n_user_ = n_user;
            n_data_item_ = n_data_item;
            vec_dim_ = vec_dim;

            cudaMalloc((void **) &user_device_ptr_, n_user_ * vec_dim_ * sizeof(float));
            cudaMalloc((void **) &item_device_ptr_, n_data_item_ * vec_dim_ * sizeof(float));
            cudaMalloc((void **) &query_vecs_device_ptr_, vec_dim_ * sizeof(float));

            cudaMalloc((void **) &queryIP_device_ptr_, n_user_ * sizeof(float));
            cudaMalloc((void **) &itemIP_device_ptr_, n_data_item_ * sizeof(float));
            cudaCheckErrors("cuda malloc fail");

            cublasCheckErrors(
                    cublasSetMatrix(vec_dim_, n_user_, sizeof(float), (void *) user_ptr, vec_dim,
                                    (void *) user_device_ptr_,
                                    vec_dim_));
            cublasCheckErrors(
                    cublasSetMatrix(vec_dim, n_data_item_, sizeof(float), (void *) data_item_ptr, vec_dim,
                                    (void *) item_device_ptr_,
                                    vec_dim));


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

        [[nodiscard]] int RefineRank(const float queryIP, const int userID) const {

            const float *user_vecs_device_ptr = user_device_ptr_ + userID * vec_dim_;
            float alpha = 1.0;
            float beta = 0.0;
            cublasCheckErrors(
                    cublasSgemv(handle_, CUBLAS_OP_T, vec_dim_, n_data_item_, &alpha, item_device_ptr_, vec_dim_,
                                user_vecs_device_ptr, 1, &beta,
                                itemIP_device_ptr_, 1));

            count_bigger f(queryIP);
            int count = thrust::count_if(thrust::device, itemIP_device_ptr_, itemIP_device_ptr_ + n_data_item_, f);
            return count;
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
        }
    };

}
#endif //REVERSE_KRANKS_REFINEGPU_HPP
