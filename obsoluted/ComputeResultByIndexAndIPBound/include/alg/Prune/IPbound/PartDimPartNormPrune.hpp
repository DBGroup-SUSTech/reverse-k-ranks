//
// Created by BianZheng on 2022/3/21.
//

#ifndef REVERSE_KRANKS_PARTDIMPARTNORM_HPP
#define REVERSE_KRANKS_PARTDIMPARTNORM_HPP

#include <cassert>
#include <memory>
#include <vector>
#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/RankBoundElement.hpp"

namespace ReverseMIPS {

    class PartDimPartNormPrune {
        int n_user_, vec_dim_, check_dim_, remain_dim_;
        std::vector<double> user_norm_l_;
    public:
        PartDimPartNormPrune() = default;

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
                double right_norm = InnerProduct(user.getVector(userID, check_dim_), user.getVector(userID, check_dim_),
                                                 remain_dim_);
                right_norm = std::sqrt(right_norm);
                user_norm_l_[userID] = right_norm;
            }
        }

        void
        IPBound(const double *query_vecs, const VectorMatrix &user, const std::vector<bool> &prune_l,
                std::vector<std::pair<double, double>> &ip_bound_l) {
            assert(ip_bound_l.size() == n_user_);
            assert(prune_l.size() == n_user_);

            const double *query_remain_vecs = query_vecs + check_dim_;
            double qright_norm = InnerProduct(query_remain_vecs, query_remain_vecs, remain_dim_);
            qright_norm = std::sqrt(qright_norm);

            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID]) {
                    continue;
                }
                double leftIP = InnerProduct(user.getVector(userID), query_vecs, check_dim_);
                double times_norm = user_norm_l_[userID] * qright_norm;
                double lb = leftIP - times_norm;
                double ub = leftIP + times_norm;
                ip_bound_l[userID] = std::make_pair(lb, ub);
                assert(lb <= ub);
            }

        }

    };

}
#endif //REVERSE_KRANKS_PARTDIMPARTNORM_HPP
