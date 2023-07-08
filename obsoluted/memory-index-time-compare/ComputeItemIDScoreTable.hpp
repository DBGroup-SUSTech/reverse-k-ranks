//
// Created by BianZheng on 2022/7/27.
//

#ifndef REVERSE_KRANKS_COMPUTEITEMSCORE_HPP
#define REVERSE_KRANKS_COMPUTEITEMIDSCORETABLE_HPP

#include "struct/DistancePair.hpp"
#include "struct/VectorMatrix.hpp"
#include "util/TimeMemory.hpp"

#include <boost/sort/sort.hpp>
#include <vector>
#include <parallel/algorithm>

#include "score_computation/CPUScoreTable.hpp"


namespace ReverseMIPS {
    class ComputeItemIDScoreTable {
        TimeRecord record_;
        uint64_t n_data_item_;
        std::vector<double> ip_cache_l_;

        CPUScoreTable cpu;
    public:
        const int report_every_ = 100000;

        double compute_time_, sort_time_;

        ComputeItemIDScoreTable() = default;

        inline ComputeItemIDScoreTable(const VectorMatrix &user, const VectorMatrix &data_item) {
            const double *user_vecs = user.getRawData();
            const double *item_vecs = data_item.getRawData();
            const uint64_t n_user = user.n_vector_;
            const uint64_t n_data_item = data_item.n_vector_;
            const uint64_t vec_dim = user.vec_dim_;
            assert(user.vec_dim_ == data_item.vec_dim_);
            this->n_data_item_ = n_data_item;
            this->ip_cache_l_.resize(n_data_item);
            cpu = CPUScoreTable(user_vecs, item_vecs, n_user, n_data_item, vec_dim);

        }

        void ComputeItems(const int &userID, double *distance_l) {
            record_.reset();
            cpu.ComputeList(userID, distance_l);
            compute_time_ += record_.get_elapsed_time_second();
        }

        void SortItems(const int &userID, double *distance_l) {
            record_.reset();
//                        __gnu_parallel::sort(distance_l, distance_l + n_data_item_, std::greater());
            boost::sort::block_indirect_sort(distance_l, distance_l + n_data_item_, std::greater(),
                                             std::thread::hardware_concurrency());


            sort_time_ += record_.get_elapsed_time_second();
        }

        void FinishCompute() {
            cpu.FinishCompute();
        }
    };

}
#endif //REVERSE_KRANKS_COMPUTEITEMSCORE_HPP
