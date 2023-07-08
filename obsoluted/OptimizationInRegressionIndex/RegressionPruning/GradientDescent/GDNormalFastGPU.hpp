//
// Created by bianzheng on 2023/4/4.
//

#ifndef REVERSE_KRANKS_GDNORMALFASTGPU_HPP
#define REVERSE_KRANKS_GDNORMALFASTGPU_HPP

#include <cublas_v2.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <vector>
#include <iostream>

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

    struct minus_square {
        float average_;

        __host__ __device__
        float operator()(const float x) const {
            const float minus = x - average_;
            return minus * minus;
        }
    };

    struct cdf_transform {
        cdf_transform(const float d, const float d1) {
            mu_ = d;
            sigma_ = d1;
        }

        float mu_, sigma_;
        float sqrt_2_ = std::sqrt(2.0);

        float CDFPhi(float x) const {
            // constants
            constexpr float a1 = 0.254829592;
            constexpr float a2 = -0.284496736;
            constexpr float a3 = 1.421413741;
            constexpr float a4 = -1.453152027;
            constexpr float a5 = 1.061405429;
            constexpr float p = 0.3275911;

            // Save the sign of x
            int sign = 1;
            if (x < 0)
                sign = -1;
            x = std::fabs(x) / sqrt_2_;

            // A&S formula 7.1.26
            float t = 1.0 / (1.0 + p * x);
            float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

            return 0.5 * (1.0 + sign * y);
        }

        __host__ __device__
        float operator()(const float x) const {
            const float normal_num = (x - mu_) / sigma_;
            return CDFPhi(normal_num);
        }
    };

    struct forward_transform {
        forward_transform(const float *sampleIP_device_ptr,
                          const float para_a, const float para_b,
                          const int n_sample_rank,
                          float *error_device_ptr, float *grad_a_device_ptr, float *grad_b_device_ptr) {
            sampleIP_device_ptr_ = sampleIP_device_ptr;
            para_a_ = para_a;
            para_b_ = para_b;
            n_sample_rank_ = n_sample_rank;

            error_device_ptr_ = error_device_ptr;
            grad_a_device_ptr_ = grad_a_device_ptr;
            grad_b_device_ptr_ = grad_b_device_ptr;
        }

        const float *sampleIP_device_ptr_;
        float para_a_, para_b_;
        int n_sample_rank_;
        float *error_device_ptr_;
        float *grad_a_device_ptr_;
        float *grad_b_device_ptr_;

        __host__ __device__
        void operator()(const int sampleID) const {
            const float cdf = sampleIP_device_ptr_[sampleID];
            const float pred_rank = cdf * para_a_ + para_b_;
            float tmp_error = (float) (sampleID + 1) - pred_rank;
            const bool is_error_negative2 = tmp_error < 0;
            tmp_error = std::abs(tmp_error);
            error_device_ptr_[sampleID * 2] = tmp_error;
            grad_a_device_ptr_[sampleID * 2] = is_error_negative2 ? cdf : -cdf;
            grad_b_device_ptr_[sampleID * 2] = is_error_negative2 ? 1 : -1;

            float tmp_error2 = (float) (sampleID) - pred_rank;
            const bool is_error_negative = tmp_error2 < 0;
            tmp_error2 = std::abs(tmp_error2);
            error_device_ptr_[sampleID * 2 + 1] = tmp_error2;
            grad_a_device_ptr_[sampleID * 2 + 1] = is_error_negative ? cdf : -cdf;
            grad_b_device_ptr_[sampleID * 2 + 1] = is_error_negative ? 1 : -1;
        }
    };

    class GDNormalFastGPU {
        int n_sample_rank_, batch_n_user_;

        float *transform_array_device_ptr_; //batch_n_user_ * n_sample_rank_
        float *error_arr_device_ptr_; //batch_n_user_ * n_sample_rank_ * 2
        float *grad_a_arr_device_ptr_; //batch_n_user_ * n_sample_rank_ * 2
        float *grad_b_arr_device_ptr_; //batch_n_user_ * n_sample_rank_ * 2
        int *eval_seq_ptr_;

        float ComputeAverage(const float *sampleIP_device_l) const {
            float sum = thrust::reduce(thrust::device, sampleIP_device_l, sampleIP_device_l + n_sample_rank_,
                                       (float) 0, thrust::plus<float>());
            return sum / (float) n_sample_rank_;
        }

        float ComputeStd(const float *sampleIP_device_l, const float average) const {
            minus_square op = {average};
            float sigma = thrust::transform_reduce(thrust::device, sampleIP_device_l,
                                                   sampleIP_device_l + n_sample_rank_,
                                                   op, (float) 0, thrust::plus<float>());
            sigma /= (float) n_sample_rank_;
            return std::sqrt(sigma);
        }

    public:
        inline GDNormalFastGPU() {}

        inline GDNormalFastGPU(const int n_sample_rank, const int batch_n_user) {
            this->n_sample_rank_ = n_sample_rank;
            this->batch_n_user_ = batch_n_user;

            cudaMalloc((void **) &transform_array_device_ptr_, batch_n_user_ * n_sample_rank_ * sizeof(float));
            cudaMalloc((void **) &error_arr_device_ptr_, batch_n_user_ * n_sample_rank_ * 2 * sizeof(float));
            cudaMalloc((void **) &grad_a_arr_device_ptr_, batch_n_user_ * n_sample_rank_ * 2 * sizeof(float));
            cudaMalloc((void **) &grad_b_arr_device_ptr_, batch_n_user_ * n_sample_rank_ * 2 * sizeof(float));
            cudaMalloc((void **) &eval_seq_ptr_, n_sample_rank_ * sizeof(int));
            std::vector<int> eval_seq_l(n_sample_rank_);
            std::iota(eval_seq_l.begin(), eval_seq_l.end(), 0);
            CHECK(cudaMemcpy(eval_seq_ptr_, eval_seq_l.data(), n_sample_rank_ * sizeof(int), cudaMemcpyHostToDevice));
            cudaCheckErrors("cuda malloc fail");

        }

        void Load(std::vector<const float *> sampleIP_l_l, const int start_userID, const int n_proc_user) {

            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                CHECK(cudaMemcpy(transform_array_device_ptr_ + proc_userID * n_sample_rank_, sampleIP_l_l[proc_userID],
                                 sizeof(float) * n_sample_rank_,
                                 cudaMemcpyHostToDevice));
            }

        }

        void CalcDistributionPara(const int start_userID, const int n_proc_user,
                                  const int n_distribution_parameter, float *distribution_para_l) {
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const int userID = proc_userID + start_userID;
                //compute average, std
                const float mu = ComputeAverage(transform_array_device_ptr_ + proc_userID * n_sample_rank_);
                const float sigma = ComputeStd(transform_array_device_ptr_ + proc_userID * n_sample_rank_, mu);
                distribution_para_l[userID * n_distribution_parameter] = mu;
                distribution_para_l[userID * n_distribution_parameter + 1] = sigma;

            }
        }

        void Precompute(const float *distribution_para_l, const int n_distribution_parameter,
                        const int start_userID, const int n_proc_user) {

            assert(n_proc_user <= batch_n_user_);

            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const int userID = proc_userID + start_userID;
                const float mu = distribution_para_l[userID * n_distribution_parameter];
                const float sigma = distribution_para_l[userID * n_distribution_parameter + 1];

                float *transform_array_device_ptr = transform_array_device_ptr_ + proc_userID * n_sample_rank_;
                thrust::transform(thrust::device, transform_array_device_ptr,
                                  transform_array_device_ptr + n_sample_rank_,
                                  transform_array_device_ptr,
                                  cdf_transform(mu, sigma));

            }
        }

        int cudaArgmax(const float *data_device, int size) {

            // Find the maximum element and its index
            auto max = thrust::max_element(thrust::device, data_device, data_device + size, thrust::less<float>());

            // Return the index of the maximum element
            return max - data_device;
        }

        void ForwardBatch(const int start_userID, const int n_proc_user,
                          const float *para_a_value_ptr, const float *para_b_value_ptr,
                          float *para_a_gradient_ptr, float *para_b_gradient_ptr) {

            assert(n_proc_user < batch_n_user_);

            //compute all tmp_error
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {

                const float para_a = para_a_value_ptr[proc_userID];
                const float para_b = para_b_value_ptr[proc_userID];

                float *transform_array_device_ptr = transform_array_device_ptr_ + proc_userID * n_sample_rank_;
                forward_transform f(transform_array_device_ptr, para_a, para_b,
                                    n_sample_rank_,
                                    error_arr_device_ptr_ + proc_userID * n_sample_rank_ * 2,
                                    grad_a_arr_device_ptr_ + proc_userID * n_sample_rank_ * 2,
                                    grad_b_arr_device_ptr_ + proc_userID * n_sample_rank_ * 2);
                thrust::for_each(thrust::device, eval_seq_ptr_, eval_seq_ptr_ + n_sample_rank_, f);

            }

            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {

                const int max_error_index = cudaArgmax(error_arr_device_ptr_ + proc_userID * n_sample_rank_ * 2,
                                                       n_sample_rank_ * 2);
                CHECK(cudaMemcpy(para_a_gradient_ptr + proc_userID,
                                 grad_a_arr_device_ptr_ + proc_userID * n_sample_rank_ * 2 + max_error_index,
                                 sizeof(float),
                                 cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(para_b_gradient_ptr + proc_userID,
                                 grad_b_arr_device_ptr_ + proc_userID * n_sample_rank_ * 2 + max_error_index,
                                 sizeof(float),
                                 cudaMemcpyDeviceToHost));
                if (proc_userID == 10 && start_userID == 0) {
                    printf("para_a_gradient_ptr %.3f, para_b_gradient_ptr %.3f\n", para_a_gradient_ptr[proc_userID],
                           para_b_gradient_ptr[proc_userID]);
                }
            }
        }

        void ForwardCalcError(const int start_userID, const int n_proc_user,
                              const float *predict_para_ptr, const int n_predict_parameter,
                              float *error_ptr) {
            assert(n_proc_user < batch_n_user_);

            //compute all tmp_error
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const int userID = proc_userID + start_userID;

                const float para_a = predict_para_ptr[userID * n_predict_parameter];
                const float para_b = predict_para_ptr[userID * n_predict_parameter + 1];

                float *transform_array_device_ptr = transform_array_device_ptr_ + proc_userID * n_sample_rank_;
                forward_transform f(transform_array_device_ptr, para_a, para_b,
                                    n_sample_rank_,
                                    error_arr_device_ptr_ + proc_userID * n_sample_rank_ * 2,
                                    grad_a_arr_device_ptr_ + proc_userID * n_sample_rank_ * 2,
                                    grad_b_arr_device_ptr_ + proc_userID * n_sample_rank_ * 2);
                thrust::for_each(thrust::device, eval_seq_ptr_, eval_seq_ptr_ + n_sample_rank_, f);

            }

            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                float max_error =
                        thrust::reduce(thrust::device, error_arr_device_ptr_ + proc_userID * n_sample_rank_ * 2,
                                       error_arr_device_ptr_ + proc_userID * n_sample_rank_ * 2 + n_sample_rank_ * 2,
                                       0.0f, thrust::maximum<float>()) + 0.01f;
                if (max_error > n_sample_rank_) {
                    max_error = n_sample_rank_ + 0.01f;
                }
                const int userID = proc_userID + start_userID;
                error_ptr[userID] = max_error;
            }

            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const int userID = proc_userID + start_userID;

                const float para_a = predict_para_ptr[userID * n_predict_parameter];
                const float para_b = predict_para_ptr[userID * n_predict_parameter + 1];
                if (proc_userID == 10 && start_userID == 0) {
                    printf("proc_userID %d, start_userID %d, para_a %.3f, para_b %.3f, error %.3f\n",
                           proc_userID, start_userID,
                           para_a, para_b, error_ptr[userID]);
                }
            }

        }

        void FinishPreprocess() {
            if (transform_array_device_ptr_ != nullptr) {
                cudaFree(transform_array_device_ptr_);
            }
            if (error_arr_device_ptr_ != nullptr) {
                cudaFree(error_arr_device_ptr_);
            }
            if (grad_a_arr_device_ptr_ != nullptr) {
                cudaFree(grad_a_arr_device_ptr_);
            }
            if (grad_b_arr_device_ptr_ != nullptr) {
                cudaFree(grad_b_arr_device_ptr_);
            }
            if (eval_seq_ptr_ != nullptr) {
                cudaFree(eval_seq_ptr_);
            }
        }

    };
}
#endif //REVERSE_KRANKS_GDNORMALFASTGPU_HPP
