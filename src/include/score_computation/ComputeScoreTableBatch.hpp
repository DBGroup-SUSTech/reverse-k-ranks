//
// Created by bianzheng on 2023/2/28.
//

#ifndef REVERSE_KRANKS_COMPUTESCORETABLEBATCH_HPP
#define REVERSE_KRANKS_COMPUTESCORETABLEBATCH_HPP

//#define USE_GPU

#include "struct/DistancePair.hpp"
#include "struct/VectorMatrix.hpp"

#include <boost/sort/sort.hpp>
#include <vector>
#include <parallel/algorithm>
#include <thread>
#include <spdlog/spdlog.h>

#ifdef USE_GPU

#include "score_computation/GPUComputeSort.hpp"
//#include "score_computation/GPUComputeSortStream.hpp"

#else

#include "score_computation/CPUScoreTable.hpp"

#endif


namespace ReverseMIPS {
    class ComputeScoreTableBatch {
        int n_data_item_;

#ifdef USE_GPU
        GPUComputeSort gpu_compute_sort;
#else
        CPUScoreTable cpu;
#endif
    public:

        ComputeScoreTableBatch() = default;

        inline ComputeScoreTableBatch(const VectorMatrix &user, const VectorMatrix &data_item) {
            const float *user_vecs = user.getRawData();
            const float *item_vecs = data_item.getRawData();
            const int n_user = user.n_vector_;
            const int n_data_item = data_item.n_vector_;
            const int vec_dim = user.vec_dim_;
            assert(user.vec_dim_ == data_item.vec_dim_);
            this->n_data_item_ = n_data_item;
#ifdef USE_GPU
            gpu_compute_sort = GPUComputeSort(user_vecs, item_vecs, n_user, n_data_item, vec_dim);
            spdlog::info("use GPU");
#else
            cpu = CPUScoreTable(user_vecs, item_vecs, n_user, n_data_item, vec_dim);
            spdlog::info("use CPU");
#endif
        }

        int ComputeBatchUser() const {

            int batch_n_user;
#ifdef USE_GPU
            batch_n_user = gpu_compute_sort.ComputeBatchUser();
#else
            batch_n_user = cpu.ComputeBatchUser(40);
#endif
            return (int) batch_n_user;
        }

        void init(const int &batch_n_user, float *distance_l) {
#ifdef USE_GPU
            gpu_compute_sort.batch_n_user_ = batch_n_user;
            gpu_compute_sort.Init(distance_l);
#endif
        }

        void ComputeSortItemsBatch(const int &start_userID, const int &batch_n_user, float *distance_l) {
#ifdef USE_GPU
            gpu_compute_sort.ComputeSortListBatch(start_userID, batch_n_user, distance_l);
#else
            cpu.ComputeList(start_userID, batch_n_user, distance_l);
            //                        __gnu_parallel::sort(distance_l, distance_l + n_data_item_, std::greater());
            for (int batch_userID = 0; batch_userID < batch_n_user; batch_userID++) {
                float *tmp_distance_l = distance_l + batch_userID * n_data_item_;
                boost::sort::block_indirect_sort(tmp_distance_l, tmp_distance_l + n_data_item_, std::greater(),
                                                 std::thread::hardware_concurrency());
            }
#endif

        }

        void FinishCompute(float *distance_host_l) {
#ifdef USE_GPU
            gpu_compute_sort.FinishCompute(distance_host_l);
#else
            cpu.FinishCompute();
#endif
        }
    };

}
#endif //REVERSE_KRANKS_COMPUTESCORETABLEBATCH_HPP
