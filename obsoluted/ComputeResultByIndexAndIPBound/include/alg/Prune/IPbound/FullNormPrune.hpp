//
// Created by BianZheng on 2022/3/21.
//

#ifndef REVERSE_KRANKS_FULLNORMPRUNE_HPP
#define REVERSE_KRANKS_FULLNORMPRUNE_HPP

#include <cassert>
#include <memory>
#include <vector>
#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/RankBoundElement.hpp"

namespace ReverseMIPS {

    class FullNormPrune {
        int n_user_, vec_dim_, check_dim_, remain_dim_;
        std::vector<std::pair<double, double>> user_norm_l_;
    public:
        FullNormPrune() = default;

        //make bound from offset_dim to vec_dim
        void Preprocess(const VectorMatrix &user, const int &check_dim) {
            this->n_user_ = user.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->check_dim_ = check_dim;
            this->remain_dim_ = vec_dim_ - check_dim_;
            assert(check_dim_ < vec_dim_);
            user_norm_l_.resize(n_user_);

            //compute the integer bound for the first part
            for (int userID = 0; userID < n_user_; userID++) {
                double left_norm = InnerProduct(user.getVector(userID), user.getVector(userID), check_dim);
                left_norm = std::sqrt(left_norm);

                double right_norm = InnerProduct(user.getVector(userID, check_dim_), user.getVector(userID, check_dim_),
                                                 remain_dim_);
                right_norm = std::sqrt(right_norm);
                user_norm_l_[userID] = std::make_pair(left_norm, right_norm);
            }
        }

        void
        IPBound(const double *query_vecs, const VectorMatrix &user, const std::vector<bool> &prune_l,
                std::vector<std::pair<double, double>> &ip_bound_l) {
            assert(ip_bound_l.size() == n_user_);
            assert(prune_l.size() == n_user_);

            double qleft_norm = InnerProduct(query_vecs, query_vecs, check_dim_);
            qleft_norm = std::sqrt(qleft_norm);

            const double *query_remain_vecs = query_vecs + check_dim_;
            double qright_norm = InnerProduct(query_remain_vecs, query_remain_vecs, remain_dim_);
            qright_norm = std::sqrt(qright_norm);

            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID]) {
                    continue;
                }
                std::pair<double, double> pair = user_norm_l_[userID];
                double times_norm = pair.first * qleft_norm + pair.second * qright_norm;
                ip_bound_l[userID] = std::make_pair(-times_norm, times_norm);
                assert(-times_norm <= times_norm);
            }

        }

    };

}
#endif //REVERSE_KRANKS_FULLNORMPRUNE_HPP
