//
// Created by bianzheng on 2022/7/20.
//

#ifndef REVERSE_KRANKS_GPUSORT_HPP
#define REVERSE_KRANKS_GPUSORT_HPP

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

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

    class GPUSort {

        int n_data_item_;
        std::vector<int> index_vecs_;
        thrust::host_vector<int> index_host_vecs_;
        thrust::device_vector<int> index_device_vecs_;
        thrust::host_vector<float> ip_host_vecs_;
        thrust::device_vector<float> ip_device_vecs_;
    public:
        GPUSort() = default;

        inline GPUSort(const int n_data_item) {
            n_data_item_ = n_data_item;

            index_vecs_.resize(n_data_item);

            index_host_vecs_.resize(n_data_item);
            for (int itemID = 0; itemID < n_data_item; itemID++) {
                index_host_vecs_[itemID] = itemID;
            }

            index_device_vecs_.resize(n_data_item);
            ip_host_vecs_.resize(n_data_item);
            ip_device_vecs_.resize(n_data_item);
        }

        void SortList(const float *distance_l, DistancePair *distance_pair_l) {
            thrust::copy(distance_l, distance_l + n_data_item_, ip_device_vecs_.begin());
            thrust::copy(index_host_vecs_.begin(), index_host_vecs_.end(), index_device_vecs_.begin());
            thrust::sort_by_key(ip_device_vecs_.begin(), ip_device_vecs_.end(), index_device_vecs_.begin(), thrust::greater());
            thrust::copy(index_device_vecs_.begin(), index_device_vecs_.end(), index_vecs_.begin());
            thrust::copy(ip_device_vecs_.begin(), ip_device_vecs_.end(), ip_host_vecs_.begin());
            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                distance_pair_l[itemID] = DistancePair(ip_host_vecs_[itemID], index_vecs_[itemID]);
            }

        }

        void SortList(float *distance_l) {
            thrust::copy(distance_l, distance_l + n_data_item_, ip_device_vecs_.begin());
            thrust::sort(ip_device_vecs_.begin(), ip_device_vecs_.end(), thrust::greater());
            thrust::copy(ip_device_vecs_.begin(), ip_device_vecs_.end(), distance_l);

        }

    };

}
#endif //REVERSE_KRANKS_GPUSORT_HPP
