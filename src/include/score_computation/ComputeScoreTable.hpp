//
// Created by BianZheng on 2022/7/13.
//

#ifndef REVERSE_KRANKS_COMPUTESCORETABLE_HPP
#define REVERSE_KRANKS_COMPUTESCORETABLE_HPP

#include "struct/DistancePair.hpp"
#include "struct/VectorMatrix.hpp"

#include <boost/sort/sort.hpp>
#include <vector>
#include <parallel/algorithm>
#include <thread>
#include <spdlog/spdlog.h>

#ifdef USE_GPU

#include "score_computation/GPUCompute.hpp"
#include "score_computation/GPUSort.hpp"
//#include "score_computation/GPUScoreTableOrigin.hpp"

#else

#include "score_computation/CPUScoreTable.hpp"

#endif


namespace ReverseMIPS {
    class ComputeScoreTable {
        int n_data_item_;
        std::vector<float> ip_cache_l_;

#ifdef USE_GPU
        GPUCompute gpu_compute;
        GPUSort gpu_sort;
#else
        CPUScoreTable cpu;
#endif
    public:

        ComputeScoreTable() = default;

        inline ComputeScoreTable(const VectorMatrix &user, const VectorMatrix &data_item) {
            const float *user_vecs = user.getRawData();
            const float *item_vecs = data_item.getRawData();
            const int n_user = user.n_vector_;
            const int n_data_item = data_item.n_vector_;
            const int vec_dim = user.vec_dim_;
            assert(user.vec_dim_ == data_item.vec_dim_);
            this->n_data_item_ = n_data_item;
            this->ip_cache_l_.resize(n_data_item);
#ifdef USE_GPU
            gpu_compute = GPUCompute(user_vecs, item_vecs, n_user, n_data_item, vec_dim);
            gpu_sort = GPUSort(n_data_item);
            spdlog::info("use GPU");
#else
            cpu = CPUScoreTable(user_vecs, item_vecs, n_user, n_data_item, vec_dim);
             spdlog::info("use CPU");
#endif


        }

        void ComputeSortItems(const int &userID, float *distance_l) {
#ifdef USE_GPU
            gpu_compute.ComputeList(userID, distance_l);
            gpu_sort.SortList(distance_l);
#else
            cpu.ComputeList(userID, distance_l);
            //                        __gnu_parallel::sort(distance_l, distance_l + n_data_item_, std::greater());
            boost::sort::block_indirect_sort(distance_l, distance_l + n_data_item_, std::greater(),
                                                         std::thread::hardware_concurrency());
#endif

        }

        void ComputeSortItems(const int &userID, DistancePair *distance_l) {
#ifdef USE_GPU
            gpu_compute.ComputeList(userID, ip_cache_l_.data());
            gpu_sort.SortList(ip_cache_l_.data(), distance_l);
#else
            cpu.ComputeList(userID, ip_cache_l_.data());
#pragma omp parallel for default(none) shared(distance_l)
            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                distance_l[itemID] = DistancePair(ip_cache_l_[itemID], itemID);
            }
            boost::sort::block_indirect_sort(distance_l, distance_l + n_data_item_, std::greater(),
                                             std::thread::hardware_concurrency());
#endif

        }

        void FinishCompute() {
#ifdef USE_GPU
            gpu_compute.FinishCompute();
#else
            cpu.FinishCompute();
#endif
        }
    };

}
#endif //REVERSE_KRANKS_COMPUTESCORETABLE_HPP
