//
// Created by BianZheng on 2022/7/13.
//

#ifndef REVERSE_KRANKS_COMPUTESCORETABLESINGLETHREAD_HPP
#define REVERSE_KRANKS_COMPUTESCORETABLESINGLETHREAD_HPP

#include "struct/DistancePair.hpp"
#include "struct/VectorMatrix.hpp"
#include "util/TimeMemory.hpp"

#include <vector>
#include <parallel/algorithm>
#include "CPUScoreTableSingleThread.hpp"


namespace ReverseMIPS {
    class ComputeScoreTableSingleThread {
        TimeRecord record_;
        int n_data_item_;
        std::vector<double> ip_cache_l_;

        CPUScoreTableSingleThread cpu;
    public:
        const int report_every_ = 30000;

        double compute_time_, sort_time_;

        ComputeScoreTableSingleThread() = default;

        inline ComputeScoreTableSingleThread(const VectorMatrix &user, const VectorMatrix &data_item) {
            const double *user_vecs = user.getRawData();
            const double *item_vecs = data_item.getRawData();
            const int n_user = user.n_vector_;
            const int n_data_item = data_item.n_vector_;
            const int vec_dim = user.vec_dim_;
            assert(user.vec_dim_ == data_item.vec_dim_);
            this->n_data_item_ = n_data_item;
            this->ip_cache_l_.resize(n_data_item);
            cpu = CPUScoreTableSingleThread(user_vecs, item_vecs, n_user, n_data_item, vec_dim);
        }

        void ComputeSortItems(const int &userID, double *distance_l, double &compute_time, double &sort_time) {
            record_.reset();
            cpu.ComputeList(userID, distance_l);
            const double tmp_compute_time = record_.get_elapsed_time_second();
            compute_time += tmp_compute_time;
            compute_time_ += tmp_compute_time;

            record_.reset();
            cpu.SortList(distance_l);
            const double tmp_sort_time = record_.get_elapsed_time_second();
            sort_time += tmp_sort_time;
            sort_time_ += tmp_sort_time;
        }

        void ComputeSortItems(const int &userID, DistancePair *distance_l, double &computeIP_time, double &sort_time) {
            record_.reset();
            cpu.ComputeList(userID, ip_cache_l_.data());
            const double tmp_computeIP_time = record_.get_elapsed_time_second();
            computeIP_time += tmp_computeIP_time;
            compute_time_ += tmp_computeIP_time;

            record_.reset();
            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                distance_l[itemID] = DistancePair(ip_cache_l_[itemID], itemID);
            }
            cpu.SortList(distance_l);
            const double tmp_sort_time = record_.get_elapsed_time_second();
            sort_time += tmp_sort_time;
            sort_time_ += tmp_sort_time;
        }

        void FinishCompute() {
            cpu.FinishCompute();
        }
    };

}
#endif //REVERSE_KRANKS_COMPUTESCORETABLESINGLETHREAD_HPP
