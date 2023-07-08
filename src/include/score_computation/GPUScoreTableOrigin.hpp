//
// Created by BianZheng on 2022/7/18.
//

#ifndef REVERSE_KRANKS_GPUSCORETABLEORIGIN_HPP
#define REVERSE_KRANKS_GPUSCORETABLEORIGIN_HPP

#include <algorithm>
#include <vector>
#include <iostream>

namespace ReverseMIPS {

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

    __global__ void ComputeInnerProductGPU(const double *user_ptr, const double *data_item_ptr,
                                           const int n_data_item, const int vec_dim, const int userID,
                                           double *IP_ptr) {

        int itemID = blockIdx.x * blockDim.x + threadIdx.x;
        if (itemID < n_data_item) {
            double ip = 0;
            const double *tmp_data_item_ptr = data_item_ptr + itemID * vec_dim;

            for (int dim = 0; dim < vec_dim; dim++) {
                ip += user_ptr[dim] * tmp_data_item_ptr[dim];
            }
            IP_ptr[itemID] = ip;
        }
    }

    class GPUScoreTableOrigin {

        int n_user_, n_data_item_, vec_dim_;
        double *user_vecs_gpu_ptr_;
        double *data_item_gpu_ptr_;
        double *ip_cache_gpu_ptr_;
        const double *user_cpu_ptr_;
        std::vector<double> ip_cache_;
        int n_thread_, n_block_;
    public:
        GPUScoreTableOrigin() = default;

        inline GPUScoreTableOrigin(const double *user, const double *data_item,
                                   const int n_user, const int n_data_item, const int vec_dim) {
            n_user_ = n_user;
            n_data_item_ = n_data_item;
            vec_dim_ = vec_dim;
            user_cpu_ptr_ = user;
            ip_cache_.resize(n_data_item_);

            CHECK(cudaMalloc((void **) &user_vecs_gpu_ptr_, vec_dim_ * sizeof(double)));
            CHECK(cudaMalloc((void **) &data_item_gpu_ptr_, n_data_item_ * vec_dim_ * sizeof(double)));
            CHECK(cudaMalloc((void **) &ip_cache_gpu_ptr_, n_data_item_ * sizeof(double)));

            CHECK(cudaMemcpy(data_item_gpu_ptr_, data_item, n_data_item_ * vec_dim_ * sizeof(double),
                             cudaMemcpyHostToDevice));
            CHECK(cudaMemset(ip_cache_gpu_ptr_, 0, n_data_item_ * sizeof(double)););

            const int n_thread = 1024;
            const int n_block = n_data_item_ / n_thread + 1;

            this->n_thread_ = n_thread;
            this->n_block_ = n_block;

        }

        void ComputeList(const int &userID, double *distance_l) {

            const double *tmp_user_cpu_ptr = user_cpu_ptr_ + userID * vec_dim_;
            CHECK(cudaMemcpy(user_vecs_gpu_ptr_, tmp_user_cpu_ptr, vec_dim_ * sizeof(double),
                             cudaMemcpyHostToDevice));

            ComputeInnerProductGPU<<<n_thread_, n_block_>>>(user_vecs_gpu_ptr_, data_item_gpu_ptr_,
                    n_data_item_, vec_dim_, userID,
                    ip_cache_gpu_ptr_);
//            ComputeInnerProductGPU<<<n_block_, n_thread_>>>(user_vecs_gpu_ptr_, data_item_gpu_ptr_,
//                    n_data_item_, vec_dim_, userID,
//                    ip_cache_gpu_ptr_);
//            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(distance_l, ip_cache_gpu_ptr_, n_data_item_ * sizeof(double), cudaMemcpyDeviceToHost));
        }

        void FinishCompute() {
            if (user_vecs_gpu_ptr_ != nullptr) {
                CHECK(cudaFree(user_vecs_gpu_ptr_));
            }
            if (data_item_gpu_ptr_ != nullptr) {
                CHECK(cudaFree(data_item_gpu_ptr_));
            }
            if (ip_cache_gpu_ptr_ != nullptr) {
                CHECK(cudaFree(ip_cache_gpu_ptr_));
            }
        }
    };

}

#endif //REVERSE_KRANKS_GPUSCORETABLEORIGIN_HPP
