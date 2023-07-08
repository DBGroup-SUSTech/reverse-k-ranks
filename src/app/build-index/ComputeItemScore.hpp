//
// Created by BianZheng on 2022/7/27.
//

#ifndef REVERSE_KRANKS_COMPUTEITEMSCORE_HPP
#define REVERSE_KRANKS_COMPUTEITEMSCORE_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "struct/DistancePair.hpp"
#include "struct/VectorMatrix.hpp"
#include "util/TimeMemory.hpp"

#include <boost/sort/sort.hpp>
#include <vector>
#include <parallel/algorithm>

namespace ReverseMIPS {
    class ComputeItemScore {
        TimeRecord record_;

        uint64_t n_user_, n_data_item_, vec_dim_;
        const float *user_vecs_;
        const float *data_item_vecs_;
    public:
        const int report_every_ = 100000;

        double compute_time_;

        ComputeItemScore() = default;

        inline ComputeItemScore(const VectorMatrix &user, const VectorMatrix &data_item) {
            user_vecs_ = user.getRawData();
            data_item_vecs_ = data_item.getRawData();
            n_user_ = user.n_vector_;
            n_data_item_ = data_item.n_vector_;
            vec_dim_ = user.vec_dim_;
            assert(user.vec_dim_ == data_item.vec_dim_);
        }

        void ComputeItems(const int *itemID_l, const int &n_candidate, const int &userID, float *cand_score_l) {
            record_.reset();

            assert(0 <= userID && userID < n_user_);
            const float *tmp_user_vecs = user_vecs_ + userID * vec_dim_;
#pragma omp parallel for default(none) shared(n_candidate, itemID_l, tmp_user_vecs, cand_score_l)
            for (int candID = 0; candID < n_candidate; candID++) {
                const int itemID = itemID_l[candID];
                assert(0 <= itemID && itemID < n_data_item_);
                const float *tmp_data_item_vecs = data_item_vecs_ + itemID * vec_dim_;
                const float ip = InnerProduct(tmp_user_vecs, tmp_data_item_vecs, (int) vec_dim_);
                cand_score_l[candID] = ip;
            }
            compute_time_ += record_.get_elapsed_time_second();
        }
    };

}
#endif //REVERSE_KRANKS_COMPUTEITEMSCORE_HPP
