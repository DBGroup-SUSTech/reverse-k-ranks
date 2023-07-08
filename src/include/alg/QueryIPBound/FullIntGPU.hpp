//
// Created by bianzheng on 2023/3/28.
//

#ifndef REVERSE_KRANKS_FULLINTGPU_HPP
#define REVERSE_KRANKS_FULLINTGPU_HPP

#include <cassert>
#include <memory>
#include <thrust/inner_product.h>

#include "alg/QueryIPBound/BaseQueryIPBound.hpp"
#include "alg/SpaceInnerProduct.hpp"

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

    struct IntIPBound {
        const int *query_int_device_ptr_;
        const int *user_int_device_ptr_;
        const int *user_int_sum_ptr_;
        float convert_coe_;
        int query_int_sum_;
        int vec_dim_;

        IntIPBound(const int *query_int_device_ptr, const int *user_int_device_ptr,
                   const int *user_int_sum_ptr,
                   const float convert_coe, const int query_int_sum, const int vec_dim) {
            query_int_device_ptr_ = query_int_device_ptr;
            user_int_device_ptr_ = user_int_device_ptr;
            user_int_sum_ptr_ = user_int_sum_ptr;
            convert_coe_ = convert_coe;
            query_int_sum_ = query_int_sum;
            vec_dim_ = vec_dim;
        }

        __host__ __device__
        thrust::pair<float, float> operator()(const int userID) const {
            const int *user_int_vecs = user_int_device_ptr_ + userID * vec_dim_;

            const int intIP = thrust::inner_product(thrust::device, user_int_vecs, user_int_vecs + vec_dim_,
                                                    query_int_device_ptr_, 0.0f);

//            int intIP = ReverseMIPS::IntIPBound::InnerProduct();
            int int_otherIP = user_int_sum_ptr_[userID] + query_int_sum_;
            int lb_part = intIP - int_otherIP;
            int ub_part = intIP + int_otherIP;

            float lower_bound = convert_coe_ * (float) lb_part;
            float upper_bound = convert_coe_ * (float) ub_part;

            return thrust::make_pair(lower_bound, upper_bound);
        }
    };

    class FullIntGPU {

        int64_t n_user_, vec_dim_;
        float scale_;
        float user_convert_coe_;

        int *user_int_device_ptr_;
        int *user_int_sum_device_ptr_;
        int *query_int_device_ptr_;
        int *eval_seq_device_ptr_;

        std::unique_ptr<int[]> query_int_host_l_;
    public:

        FullIntGPU() {};

        void init(const float *user_ptr, const int64_t &n_user, const int64_t &vec_dim, const float &scale) {
            this->n_user_ = n_user;
            this->vec_dim_ = vec_dim;

            std::unique_ptr<int[]> user_int_ptr = std::make_unique<int[]>(n_user_ * vec_dim_);
            std::unique_ptr<int[]> user_int_sum_ptr = std::make_unique<int[]>(n_user_);
            std::unique_ptr<int[]> query_int_ptr = std::make_unique<int[]>(vec_dim_);
            query_int_host_l_ = std::make_unique<int[]>(vec_dim_);

            Preprocess(user_ptr, scale, user_int_ptr.get(), user_int_sum_ptr.get());
            GPUPreprocess(user_int_ptr.get(), user_int_sum_ptr.get());

        }

        void Preprocess(const float *user_ptr, const float &scale,
                        int *user_int_ptr, int *user_int_sum_ptr) {
            this->scale_ = scale;

            float user_max_dim_ = user_ptr[0];
            //compute the integer bound for the first part
            for (int userID = 0; userID < n_user_; userID++) {
                const float *user_vecs = user_ptr + userID * vec_dim_;
                for (int dim = 0; dim < vec_dim_; dim++) {
                    user_max_dim_ = std::max(user_max_dim_, user_vecs[dim]);
                }
            }
            user_max_dim_ = std::abs(user_max_dim_);

#pragma omp parallel for default(none) shared(user_max_dim_,user_int_ptr, user_ptr, user_int_sum_ptr )
            for (int userID = 0; userID < n_user_; userID++) {
                int *user_int_vecs = user_int_ptr + userID * vec_dim_;
                const float *user_float_vecs = user_ptr + userID * vec_dim_;
                user_int_sum_ptr[userID] = 0;
                for (int dim = 0; dim < vec_dim_; dim++) {
                    user_int_vecs[dim] = std::floor(user_float_vecs[dim] * scale_ / user_max_dim_);
                    user_int_sum_ptr[userID] += std::abs(user_int_vecs[dim]) + 1;
                }
            }

            user_convert_coe_ = user_max_dim_ / (scale_ * scale_);
        }

        void GPUPreprocess(const int *user_int_ptr, const int *user_int_sum_ptr) {
            cudaMalloc((void **) &user_int_device_ptr_, n_user_ * vec_dim_ * sizeof(int));
            cudaMalloc((void **) &user_int_sum_device_ptr_, n_user_ * sizeof(int));
            cudaMalloc((void **) &query_int_device_ptr_, vec_dim_ * sizeof(int));
            cudaMalloc((void **) &eval_seq_device_ptr_, n_user_ * sizeof(int));
            cudaCheckErrors("cuda malloc fail");

            CHECK(cudaMemcpy(user_int_device_ptr_, user_int_ptr, n_user_ * vec_dim_ * sizeof(int),
                             cudaMemcpyHostToDevice))
            CHECK(cudaMemcpy(user_int_sum_device_ptr_, user_int_sum_ptr, n_user_ * sizeof(int),
                             cudaMemcpyHostToDevice))

            std::vector<int> eval_seq_l(n_user_);
            std::iota(eval_seq_l.begin(), eval_seq_l.end(), 0);
            CHECK(cudaMemcpy(eval_seq_device_ptr_, eval_seq_l.data(), n_user_ * sizeof(int),
                             cudaMemcpyHostToDevice))

        }

        void
        IPBound(const float *query_vecs, const int n_eval,
                thrust::pair<float, float> *ip_bound_device_l) const {

            float query_max_dim = query_vecs[0];
            for (int dim = 1; dim < vec_dim_; dim++) {
                query_max_dim = std::max(query_max_dim, query_vecs[dim]);
            }
            query_max_dim = std::abs(query_max_dim);
            const float convert_coe = user_convert_coe_ * query_max_dim;

            const float qratio = scale_ / query_max_dim;

            int query_int_sum = 0;
            int *query_int_vecs = query_int_host_l_.get();
            for (int dim = 0; dim < vec_dim_; dim++) {
                query_int_vecs[dim] = std::floor(query_vecs[dim] * qratio);
                query_int_sum += std::abs(query_int_vecs[dim]);
            }

            CHECK(cudaMemcpy(query_int_device_ptr_, query_int_vecs, vec_dim_ * sizeof(int),
                             cudaMemcpyHostToDevice));

            IntIPBound f(query_int_device_ptr_, user_int_device_ptr_,
                         user_int_sum_device_ptr_,
                         convert_coe, query_int_sum, vec_dim_);
            thrust::transform(thrust::device, eval_seq_device_ptr_, eval_seq_device_ptr_ + n_eval, ip_bound_device_l, f);
        }

        void FinishCompute() {
            if (user_int_device_ptr_ != nullptr) {
                cudaFree(user_int_device_ptr_);
            }
            if (user_int_sum_device_ptr_ != nullptr) {
                cudaFree(user_int_sum_device_ptr_);
            }
            if (query_int_device_ptr_ != nullptr) {
                cudaFree(query_int_device_ptr_);
            }
        }

    };
}
#endif //REVERSE_KRANKS_FULLINTGPU_HPP
