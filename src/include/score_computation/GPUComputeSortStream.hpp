//
// Created by bianzheng on 2023/3/15.
//

#ifndef REVERSE_KRANKS_GPUCOMPUTESORTSTREAM_HPP
#define REVERSE_KRANKS_GPUCOMPUTESORTSTREAM_HPP

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

    struct sort_functor {
        thrust::device_ptr<float> data;
        size_t dsize;

        __host__ __device__
        void operator()(const int start_idx) {
            thrust::sort(thrust::device, data + (dsize * start_idx), data + (dsize * (start_idx + 1)),
                         thrust::greater<float>());
        }
    };

    class GPUComputeSort {

        uint64_t n_user_, n_data_item_, vec_dim_;

        float *user_device_ptr_;
        float *item_device_ptr_;
        float *resIP_device_ptr_;

        cublasHandle_t handle_;

        int *sort_idx_ptr_;

        const static int NUM_STREAMS_ = 100;
        std::vector<cudaStream_t> streams_;
    public:
        int batch_n_user_;

        GPUComputeSort() = default;

        inline GPUComputeSort(const double *user_ptr, const double *data_item_ptr,
                              const uint64_t n_user, const uint64_t n_data_item, const uint64_t vec_dim) {
            n_user_ = n_user;
            n_data_item_ = n_data_item;
            vec_dim_ = vec_dim;

            cudaMalloc((void **) &item_device_ptr_, n_data_item_ * vec_dim_ * sizeof(float));
            cudaMalloc((void **) &user_device_ptr_, n_user_ * vec_dim_ * sizeof(float));

            cudaCheckErrors("cuda malloc fail");

            std::vector<float> user_float_l(n_user * vec_dim);
            const uint64_t user_size = n_user * vec_dim;
            for (uint64_t i = 0; i < user_size; i++) {
                user_float_l[i] = (float) user_ptr[i];
            }
            std::vector<float> item_float_l(n_data_item * vec_dim);
            const uint64_t item_size = n_data_item * vec_dim;
            for (uint64_t i = 0; i < item_size; i++) {
                item_float_l[i] = (float) data_item_ptr[i];
            }

            cublasCheckErrors(
                    cublasSetMatrix(vec_dim_, n_user_, sizeof(float), (void *) user_float_l.data(), vec_dim,
                                    (void *) user_device_ptr_,
                                    vec_dim_));
            cublasCheckErrors(
                    cublasSetMatrix(vec_dim, n_data_item_, sizeof(float), (void *) item_float_l.data(), vec_dim,
                                    (void *) item_device_ptr_,
                                    vec_dim));

//            std::vector<double> user_cpu_tran_vecs(n_user * vec_dim);
//            CHECK(cudaMemcpy(user_cpu_tran_vecs.data(), user_device_ptr_, n_user * vec_dim * sizeof(double),
//                             cudaMemcpyDeviceToHost));
//            printf("after memcpy in the user cpu tran vecs\n");
//            for (int i = 0; i < n_user * vec_dim; i++) {
//                assert(abs(user_cpu_tran_vecs[i] - user_ptr[i]) <= 0.01);
//            }

            streams_.resize(NUM_STREAMS_);
            // --- Create CUDA streams
            spdlog::info("use GPU stream");
            for (int i = 0; i < NUM_STREAMS_; i++) {
                CHECK(cudaStreamCreate(&streams_[i]));
            }

            cublasCheckErrors(cublasCreate(&handle_));
//            cublasCheckErrors(cublasSetVector(vec_dim, sizeof(double), &(vector[0]), 1, vector_gpu, 1));

        }

        int ComputeBatchUser() const {
            int num_gpus;
            cudaGetDeviceCount(&num_gpus);
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

            const uint64_t remain_gpu_size_gb = 1;
            const uint64_t batch_n_user =
                    (free - (remain_gpu_size_gb * 1024 * 1024 * 1024)) / sizeof(float) / 2 / n_data_item_;
            if (batch_n_user > n_user_) {
                return 120;
            }
//    return std::min((int) batch_n_user, 128);
            return (int) batch_n_user;
        }

        void Init(float *distance_host_l) {
            cudaMalloc((void **) &resIP_device_ptr_, batch_n_user_ * n_data_item_ * sizeof(float));

//            cudaDeviceSetLimit(cudaLimitMallocHeapSize,
//                               (batch_n_user_ * n_data_item_ * sizeof(float) + 8 * 1024 * 1024 * 1024));

            cudaCheckErrors("cuda malloc fail");

            CHECK(cudaHostRegister(distance_host_l, batch_n_user_ * n_data_item_ * sizeof(float),
                                   cudaHostRegisterPortable));


            cudaMalloc((void **) &sort_idx_ptr_, batch_n_user_ * sizeof(int));
            std::vector<int> seq_l(batch_n_user_);
            std::iota(seq_l.begin(), seq_l.end(), 0);
            CHECK(cudaMemcpy(sort_idx_ptr_,
                             seq_l.data(),
                             batch_n_user_ * sizeof(int),
                             cudaMemcpyHostToDevice));

        }

        void ComputeList(const int &start_userID, const int &n_compute_user) {
            const float *tmp_user_device_ptr = user_device_ptr_ + start_userID * vec_dim_;
            const float alpha = 1.0;
            const float beta = 0.0;
            cublasCheckErrors(
                    cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                n_data_item_, n_compute_user, vec_dim_,
                                &alpha, item_device_ptr_, vec_dim_, tmp_user_device_ptr, vec_dim_, &beta,
                                resIP_device_ptr_, n_data_item_
                    ));
            CHECK(cudaDeviceSynchronize());
//            cublasCheckErrors(
//                    cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
//                                n_compute_user, n_data_item_, vec_dim_,
//                                &alpha, tmp_user_device_ptr, vec_dim_, item_device_ptr_, vec_dim_, &beta,
//                                resIP_device_ptr_, n_data_item_
//                    ));


//            std::vector<double> vecs(n_compute_user * n_data_item_);
//            for (int userID = start_userID; userID < start_userID + n_compute_user; userID++) {
//                const double *tmp_user_host_ptr = tmp_user_ptr_ + userID * vec_dim_;
//                for (int itemID = 0; itemID < n_data_item_; itemID++) {
//                    const double *tmp_item_ptr = tmp_item_ptr_ + itemID * vec_dim_;
//                    vecs[itemID + (userID - start_userID) * n_data_item_] = InnerProduct(tmp_user_host_ptr,
//                                                                                         tmp_item_ptr, (int) vec_dim_);
//                }
//            }
//
//            std::vector<double> gpu_vecs(n_compute_user * n_data_item_);
//            CHECK(cudaMemcpy(gpu_vecs.data(), resIP_device_ptr_, n_compute_user * n_data_item_ * sizeof(double),
//                             cudaMemcpyDeviceToHost));
//            for (int cpt_userID = 0; cpt_userID < n_compute_user; cpt_userID++) {
//                for (int itemID = 0; itemID < n_data_item_; itemID++) {
//                    const int i = cpt_userID * n_data_item_ + itemID;
//                    if (abs(vecs[i] - gpu_vecs[i]) > 0.01) {
//                        printf("i %d, cpt_userID %d, itemID %d, vecs[i] %.3f, gpu_vecs[i] %.3f\n",
//                               i, cpt_userID, itemID, vecs[i], gpu_vecs[i]);
//                    }
//                    assert(abs(vecs[i] - gpu_vecs[i]) <= 0.01);
//                }
//            }


        }

        void ComputeSortListBatch(const int &start_userID, const int &n_compute_user, float *distance_l) {
            TimeRecord record;
            record.reset();
            ComputeList(start_userID, n_compute_user);
            const double compute_time = record.get_elapsed_time_second();

            record.reset();
            const int n_process_user_per_stream = n_compute_user / NUM_STREAMS_;
            const int n_remain_user_stream = n_compute_user / NUM_STREAMS_ + n_compute_user % NUM_STREAMS_;
            for (int streamID = 0; streamID < NUM_STREAMS_; streamID++) {
                const int start_userID_stream = n_process_user_per_stream * streamID;
                const int end_userID_stream = start_userID_stream + (streamID == NUM_STREAMS_ - 1 ? n_remain_user_stream
                                                                                                  : n_process_user_per_stream);

                const int n_stream_user = (streamID == NUM_STREAMS_ - 1 ? n_remain_user_stream
                                                                        : n_process_user_per_stream);

//                const int n_user_stream = end_userID_stream - start_userID_stream;

                thrust::device_ptr<float> sort_device_ptr = thrust::device_pointer_cast<float>(
                        resIP_device_ptr_ + start_userID_stream * n_data_item_);

                sort_functor f = {sort_device_ptr, n_data_item_};

                thrust::for_each(thrust::cuda::par.on(streams_[streamID]),
                                 sort_idx_ptr_,
                                 sort_idx_ptr_ + n_stream_user,
                                 f);

//                CHECK(cudaDeviceSynchronize());
//                thrust::host_vector<int> host_l = idxs;
//                printf("host_l val stream id: %d, n_stream_user: %d\n", streamID, n_stream_user);
//                for (int i = 0; i < host_l.size(); i++) {
//                    printf("%d ", host_l[i]);
//                }
//                printf("\n");

//                for (int comp_userID = start_userID_stream; comp_userID < end_userID_stream; comp_userID++) {
//                    thrust::device_ptr<float> sort_device_ptr = thrust::device_pointer_cast<float>(
//                            resIP_device_ptr_ + comp_userID * n_data_item_);
//                    thrust::sort(thrust::cuda::par.on(streams_[streamID]),
//                                 sort_device_ptr,
//                                 sort_device_ptr + n_data_item_,
//                                 thrust::greater<float>());
//                }

            }

            for (int streamID = 0; streamID < NUM_STREAMS_; streamID++) {
                const int start_userID_stream = n_process_user_per_stream * streamID;
                const int n_compute_user_stream = streamID == NUM_STREAMS_ - 1 ? n_remain_user_stream
                                                                               : n_process_user_per_stream;
//                CHECK(cudaMemcpy(distance_l + start_userID_stream * n_data_item_,
//                                 resIP_device_ptr_ + start_userID_stream * n_data_item_,
//                                 n_compute_user_stream * n_data_item_ * sizeof(float),
//                                 cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpyAsync(distance_l + start_userID_stream * n_data_item_,
                                      resIP_device_ptr_ + start_userID_stream * n_data_item_,
                                      n_compute_user_stream * n_data_item_ * sizeof(float),
                                      cudaMemcpyDeviceToHost, streams_[streamID]));
            }

//            for (int comp_userID = 0; comp_userID < n_compute_user; comp_userID++) {
//                CHECK(cudaMemcpyAsync(distance_l + comp_userID * n_data_item_,
//                                      resIP_device_ptr_ + comp_userID * n_data_item_,
//                                      n_data_item_ * sizeof(float),
//                                      cudaMemcpyDeviceToHost, streams_[comp_userID]));
//            }

            for (int i = 0; i < NUM_STREAMS_; i++) {
                CHECK(cudaStreamSynchronize(streams_[i]));
            }
            CHECK(cudaDeviceSynchronize());
            const double sort_memcpy_time = record.get_elapsed_time_second();

            spdlog::info("compute batch time {}s, sort memcpy time {}s", compute_time, sort_memcpy_time);


        }

        void FinishCompute(float *distance_host_l) {
            if (user_device_ptr_ != nullptr) {
                cudaFree(user_device_ptr_);
            }
            if (item_device_ptr_ != nullptr) {
                cudaFree(item_device_ptr_);
            }
            for (int i = 0; i < NUM_STREAMS_; i++) {
                CHECK(cudaStreamDestroy(streams_[i]));
            }
            CHECK(cudaHostUnregister(distance_host_l));
            cublasCheckErrors(cublasDestroy(handle_));
            cudaDeviceReset();
        }
    };

}
#endif //REVERSE_KRANKS_GPUCOMPUTESORTSTREAM_HPP
