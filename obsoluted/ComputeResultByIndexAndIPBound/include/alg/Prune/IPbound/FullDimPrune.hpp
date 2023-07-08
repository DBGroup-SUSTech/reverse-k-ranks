//
// Created by BianZheng on 2022/3/21.
//

#ifndef REVERSE_KRANKS_FULLDIMPRUNE_HPP
#define REVERSE_KRANKS_FULLDIMPRUNE_HPP

#include <cassert>
#include <memory>
#include <vector>
#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/RankBoundElement.hpp"

namespace ReverseMIPS {

    class FullDimPrune {
        int n_user_, vec_dim_;
    public:
        FullDimPrune() = default;

        //make bound from offset_dim to vec_dim
        void Preprocess(const VectorMatrix &user) {
            this->n_user_ = user.n_vector_;
            this->vec_dim_ = user.vec_dim_;
        }

        void
        IPBound(const double *query_vecs, const VectorMatrix &user, const std::vector<bool> &prune_l,
                std::vector<std::pair<double, double>> &ip_bound_l) {
            assert(ip_bound_l.size() == n_user_);
            assert(prune_l.size() == n_user_);

            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID]) {
                    continue;
                }

                double ip = InnerProduct(query_vecs, user.getVector(userID), vec_dim_);
                ip_bound_l[userID] = std::make_pair(ip, ip);
            }

        }

    };

}
#endif //REVERSE_KRANKS_FULLDIMPRUNE_HPP
