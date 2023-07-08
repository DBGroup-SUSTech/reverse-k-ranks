//
// Created by bianzheng on 2023/3/24.
//

#ifndef REVERSE_KRANKS_REFINECPU_HPP
#define REVERSE_KRANKS_REFINECPU_HPP

#include "alg/SpaceInnerProduct.hpp"

#include <algorithm>
#include <vector>
#include <iostream>

namespace ReverseMIPS {

    class RefineCPU {

        uint64_t n_user_, n_data_item_, vec_dim_;
        const float *user_ptr_;
        const float *data_item_ptr_;
    public:
        RefineCPU() = default;

        inline RefineCPU(const float *user_ptr, const float *data_item_ptr,
                         const uint64_t n_user, const uint64_t n_data_item, const uint64_t vec_dim) {
            this->user_ptr_ = user_ptr;
            this->data_item_ptr_ = data_item_ptr;
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->vec_dim_ = vec_dim;
        }

        [[nodiscard]] int RefineRank(const float queryIP, const int userID, int &n_compute_item) const {
            n_compute_item = (int) n_data_item_;
            const float *user_vecs = user_ptr_ + userID * vec_dim_;

            std::vector<char> is_queryIP_smaller_l(n_data_item_, false);

            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                const float *item_vecs = data_item_ptr_ + vec_dim_ * itemID;
                const float ip = InnerProduct(user_vecs, item_vecs, (int) vec_dim_);
                if (ip > queryIP) {
                    is_queryIP_smaller_l[itemID] = true;
                } else {
                    is_queryIP_smaller_l[itemID] = false;
                }
            }

            int base_rank = 0;
            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                base_rank += is_queryIP_smaller_l[itemID] ? 1 : 0;
            }

            return base_rank;
        }

        void FinishCompute() {
        }
    };

}
#endif //REVERSE_KRANKS_REFINECPU_HPP
