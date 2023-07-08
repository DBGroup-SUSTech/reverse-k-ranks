//
// Created by bianzheng on 2023/3/23.
//

#ifndef REVERSE_KRANKS_QUERYRANKGPU_HPP
#define REVERSE_KRANKS_QUERYRANKGPU_HPP

#include <cublas_v2.h>
#include <algorithm>
#include <vector>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
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

    struct query_rank_functor {

        thrust::device_ptr<float> score_table_ptr_; // batch_n_user * n_data_item
        size_t n_data_item_;
        thrust::device_ptr<float> sample_item_IP_ptr_; // batch_n_user * n_sample_item
        thrust::device_ptr<int> sample_item_rank_ptr_; // batch_n_user * n_sample_item
        size_t n_sample_item_;

        query_rank_functor(thrust::device_ptr<float> score_table_ptr, size_t n_data_item,
                           thrust::device_ptr<float> sample_item_IP_ptr,
                           thrust::device_ptr<int> sample_item_rank_ptr,
                           size_t n_sample_item) {
            score_table_ptr_ = score_table_ptr;
            n_data_item_ = n_data_item;
            sample_item_IP_ptr_ = sample_item_IP_ptr;
            sample_item_rank_ptr_ = sample_item_rank_ptr;
            n_sample_item_ = n_sample_item;
        }

        __host__ __device__
        void operator()(const int start_idx) {
            thrust::lower_bound(thrust::device,
                                score_table_ptr_ + start_idx * n_data_item_,
                                score_table_ptr_ + (start_idx + 1) * n_data_item_,
                                sample_item_IP_ptr_ + start_idx * n_sample_item_,
                                sample_item_IP_ptr_ + (start_idx + 1) * n_sample_item_,
                                sample_item_rank_ptr_ + start_idx * n_sample_item_,
                                thrust::greater<float>());
        }
    };

    class QueryRankGPU {

        uint64_t n_user_, n_data_item_, n_sample_item_, vec_dim_;

        const float *user_ptr_;
        const float *data_item_ptr_;
        const float *sample_item_ptr_;

        float *user_device_ptr_; // n_user * vec_dim
        float *item_device_ptr_; // n_data_item * vec_dim
        float *sample_item_device_ptr_; // n_sample_item * vec_dim

        float *score_table_device_ptr_; // batch_n_user * n_data_item
        float *sample_item_IP_device_ptr_; // batch_n_user * n_sample_item
        int *sample_item_rank_device_ptr_; // batch_n_user * n_sample_item
        int *sort_idx_device_ptr_; // batch_n_user

        cublasHandle_t handle_;

    public:
        int64_t batch_n_user_;

        QueryRankGPU() = default;

        QueryRankGPU(const float *user_ptr, const float *data_item_ptr, const float *sample_item_ptr,
                     const int64_t &n_user, const int64_t &n_data_item, const int64_t &n_sample_item,
                     const int64_t &vec_dim) {
            this->user_ptr_ = user_ptr;
            this->data_item_ptr_ = data_item_ptr;
            this->sample_item_ptr_ = sample_item_ptr;

            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->n_sample_item_ = n_sample_item;
            this->vec_dim_ = (int) vec_dim;

            cudaMalloc((void **) &user_device_ptr_, n_user_ * vec_dim_ * sizeof(float));
            cudaMalloc((void **) &item_device_ptr_, n_data_item_ * vec_dim_ * sizeof(float));
            cudaMalloc((void **) &sample_item_device_ptr_, n_sample_item_ * vec_dim_ * sizeof(float));
            cudaCheckErrors("cuda malloc fail");

            cublasCheckErrors(
                    cublasSetMatrix(vec_dim_, n_user_, sizeof(float), (void *) user_ptr, vec_dim_,
                                    (void *) user_device_ptr_,
                                    vec_dim_));
            cublasCheckErrors(
                    cublasSetMatrix(vec_dim_, n_data_item_, sizeof(float), (void *) data_item_ptr, vec_dim_,
                                    (void *) item_device_ptr_,
                                    vec_dim_));
            cublasCheckErrors(
                    cublasSetMatrix(vec_dim_, n_sample_item_, sizeof(float), (void *) sample_item_ptr, vec_dim_,
                                    (void *) sample_item_device_ptr_,
                                    vec_dim_));

            cublasCheckErrors(cublasCreate(&handle_));

        }

        [[nodiscard]] int ComputeBatchUser() const {
            // things in memory,
            // user_ptr, sizeof(float) * n_user * vec_dim
            // item_ptr, sizeof(float) * n_data_item * vec_dim
            // sample_item_ptr, sizeof(float) * n_sample_item * vec_dim
            // score_table_ptr, sizeof(float) * n_batch_user * n_data_item
            // sample_item_ip_ptr, sizeof(float) * n_batch_user * n_sample_item
            // sample_item_rank_ptr, sizeof(float) * n_batch_user * n_sample_item
            int num_gpus;
            CHECK(cudaGetDeviceCount(&num_gpus));
            if (num_gpus < 1) {
                spdlog::error("do not have gpu, program exit");
                exit(-1);
            }
            const int gpu_id = 0;
            cudaSetDevice(gpu_id);
            int id;
            size_t free, total;
            cudaGetDevice(&id);
            cudaMemGetInfo(&free, &total);

            const uint64_t remain_gpu_size_gb = 300;
            const uint64_t batch_n_user =
                    (free - (remain_gpu_size_gb * 1024 * 1024)) /
                    (sizeof(float) * (n_data_item_ + n_sample_item_) + sizeof(int) * n_sample_item_ + sizeof(int));
            if (batch_n_user > n_user_) {
                return 64;
            }
//    return std::min((int) batch_n_user, 128);
            return (int) batch_n_user;
        }

        void init() {
            cudaMalloc((void **) &score_table_device_ptr_, batch_n_user_ * n_data_item_ * sizeof(float));
            cudaMalloc((void **) &sample_item_IP_device_ptr_, batch_n_user_ * n_sample_item_ * sizeof(float));
            cudaMalloc((void **) &sample_item_rank_device_ptr_, batch_n_user_ * n_sample_item_ * sizeof(int));
            cudaCheckErrors("cuda malloc fail");

            cudaMalloc((void **) &sort_idx_device_ptr_, batch_n_user_ * sizeof(int));
            cudaCheckErrors("cuda malloc fail");
            std::vector<int> seq_l(batch_n_user_);
            std::iota(seq_l.begin(), seq_l.end(), 0);
            CHECK(cudaMemcpy(sort_idx_device_ptr_,
                             seq_l.data(),
                             batch_n_user_ * sizeof(int),
                             cudaMemcpyHostToDevice));
        }

        void ComputeList(const int &start_userID, const int &n_compute_user, const int &n_item,
                         float *item_ptr, float *score_table_ptr) {
            const float *tmp_user_device_ptr = user_device_ptr_ + start_userID * vec_dim_;
            const float alpha = 1.0;
            const float beta = 0.0;
            cublasCheckErrors(
                    cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                n_item, n_compute_user, vec_dim_,
                                &alpha, item_ptr, vec_dim_, tmp_user_device_ptr, vec_dim_, &beta,
                                score_table_ptr, n_item
                    ));
            CHECK(cudaDeviceSynchronize());

        }

        void SortList(const int n_compute_user) {

            for (int comp_userID = 0; comp_userID < n_compute_user; comp_userID++) {
                thrust::device_ptr<float> sort_device_ptr = thrust::device_pointer_cast<float>(
                        score_table_device_ptr_ + n_data_item_ * comp_userID);
                thrust::stable_sort(thrust::device, sort_device_ptr, sort_device_ptr + n_data_item_, thrust::greater<float>());
            }
            CHECK(cudaDeviceSynchronize());
        }

        void ComputeRankOfQuery(const int n_compute_user) {
            thrust::device_ptr<float> score_table_ptr = thrust::device_pointer_cast<float>(
                    score_table_device_ptr_);
            thrust::device_ptr<float> sample_item_IP_ptr = thrust::device_pointer_cast<float>(
                    sample_item_IP_device_ptr_);
            thrust::device_ptr<int> sample_item_rank_ptr = thrust::device_pointer_cast<int>(
                    sample_item_rank_device_ptr_);

            query_rank_functor f(score_table_ptr, n_data_item_, sample_item_IP_ptr, sample_item_rank_ptr,
                                 n_sample_item_);

            thrust::for_each(thrust::device, sort_idx_device_ptr_,
                             sort_idx_device_ptr_ + n_compute_user,
                             f);
            CHECK(cudaDeviceSynchronize());
        }

        void ComputeQueryRank(const int &start_userID, const int &n_compute_user, int *sample_item_rank_ptr,
                              double& compute_table_time, double& sort_table_time, double& compute_query_IP_time,
                              double& compute_query_rank_time, double& transfer_time) {
            assert(n_compute_user <= batch_n_user_);
            TimeRecord record;
            record.reset();
            ComputeList(start_userID, n_compute_user, (int) n_data_item_, item_device_ptr_,
                        score_table_device_ptr_);
            const double tmp_compute_table_time = record.get_elapsed_time_second();
            compute_table_time = tmp_compute_table_time;

            record.reset();
            SortList(n_compute_user);
            const double tmp_sort_table_time = record.get_elapsed_time_second();
            sort_table_time = tmp_sort_table_time;

            record.reset();
            ComputeList(start_userID, n_compute_user, (int) n_sample_item_, sample_item_device_ptr_,
                        sample_item_IP_device_ptr_);
            const double tmp_compute_query_IP_time = record.get_elapsed_time_second();
            compute_query_IP_time = tmp_compute_query_IP_time;

            // sort the query rank
            record.reset();
            ComputeRankOfQuery(n_compute_user);
            const double tmp_compute_query_rank_time = record.get_elapsed_time_second();
            compute_query_rank_time = tmp_compute_query_rank_time;

            //transfer the item rank to cpu
            record.reset();
            CHECK(cudaMemcpy(sample_item_rank_ptr,
                             sample_item_rank_device_ptr_,
                             n_compute_user * n_sample_item_ * sizeof(int),
                             cudaMemcpyDeviceToHost));
            const double tmp_transfer_time = record.get_elapsed_time_second();
            transfer_time = tmp_transfer_time;

//            printf("sample_item_rank_ptr [0] %d, [1] %d, [2] %d\n",
//                   sample_item_rank_ptr[0], sample_item_rank_ptr[1], sample_item_rank_ptr[2]);

//            spdlog::info(
//                    "compute table time {:.5f}s, sort table time {:.5f}s, compute queryIP time {:.5f}s, compute query rank time {:.5f}s, transfer time {:.5f}s",
//                    compute_table_time, sort_table_time, compute_query_IP_time, compute_query_rank_time, transfer_time);

        }

        void FinishCompute() {
            if (user_device_ptr_ != nullptr) {
                cudaFree(user_device_ptr_);
            }
            if (item_device_ptr_ != nullptr) {
                cudaFree(item_device_ptr_);
            }
            if (sample_item_device_ptr_ != nullptr) {
                cudaFree(sample_item_device_ptr_);
            }
            if (score_table_device_ptr_ != nullptr) {
                cudaFree(score_table_device_ptr_);
            }
            if (sample_item_IP_device_ptr_ != nullptr) {
                cudaFree(sample_item_IP_device_ptr_);
            }
            if (sample_item_rank_device_ptr_ != nullptr) {
                cudaFree(sample_item_rank_device_ptr_);
            }
            if (sort_idx_device_ptr_ != nullptr) {
                cudaFree(sort_idx_device_ptr_);
            }
            cublasCheckErrors(cublasDestroy(handle_));
            cudaDeviceReset();
        }


    };
}
#endif //REVERSE_KRANKS_QUERYRANKGPU_HPP
