//
// Created by BianZheng on 2022/3/21.
//

#ifndef REVERSE_KRANKS_PARTINTPARTNORMPRUNE_HPP
#define REVERSE_KRANKS_PARTINTPARTNORMPRUNE_HPP

#include <cassert>
#include <memory>
#include <vector>
#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/RankBoundElement.hpp"

namespace ReverseMIPS {

    class PartIntPartNormPrune {
        int n_user_, vec_dim_, check_dim_, remain_dim_;
        //int prune
        double scale_;
        std::unique_ptr<int[]> user_int_ptr_;
        std::unique_ptr<int[]> user_int_sum_ptr_;
        std::unique_ptr<int[]> query_int_ptr_;
        double user_convert_coe_, convert_coe_;
        //norm prune
        std::vector<double> user_norm_l_;
    public:
        PartIntPartNormPrune() = default;

        //make bound from offset_dim to vec_dim
        void Preprocess(const VectorMatrix &user, const int &check_dim, const double &scale) {
            this->n_user_ = user.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->check_dim_ = check_dim;
            this->remain_dim_ = vec_dim_ - check_dim_;
            this->scale_ = scale;
            assert(check_dim_ < vec_dim_);
            user_norm_l_.resize(n_user_);

            user_int_ptr_ = std::make_unique<int[]>(n_user_ * check_dim_);
            user_int_sum_ptr_ = std::make_unique<int[]>(n_user_);
            double user_max_dim_ = user.getVector(0)[0];
            query_int_ptr_ = std::make_unique<int[]>(check_dim_);

            //compute the integer bound for the first part
            for (int userID = 0; userID < n_user_; userID++) {
                double *user_vecs = user.getVector(userID);
                for (int dim = 0; dim < check_dim; dim++) {
                    user_max_dim_ = std::max(user_max_dim_, user_vecs[dim]);
                }
            }
            user_max_dim_ = std::abs(user_max_dim_);

            for (int userID = 0; userID < n_user_; userID++) {
                int *user_int_vecs = user_int_ptr_.get() + userID * check_dim_;
                double *user_double_vecs = user.getVector(userID);
                user_int_sum_ptr_[userID] = 0;
                for (int dim = 0; dim < check_dim; dim++) {
                    user_int_vecs[dim] = std::floor(user_double_vecs[dim] * scale_ / user_max_dim_);
                    user_int_sum_ptr_[userID] += std::abs(user_int_vecs[dim]) + 1;
                }
            }

            user_convert_coe_ = user_max_dim_ / (scale_ * scale_);

            for (int userID = 0; userID < n_user_; userID++) {
                double *user_remain_vecs = user.getVector(userID, check_dim_);
                double user_norm = InnerProduct(user_remain_vecs, user_remain_vecs, remain_dim_);
                user_norm = std::sqrt(user_norm);
                user_norm_l_[userID] = user_norm;
            }

        }

        void
        IPBound(const double *query_vecs, const VectorMatrix &user, const std::vector<bool> &prune_l,
                std::vector<std::pair<double, double>> &ip_bound_l) {
            assert(ip_bound_l.size() == n_user_);
            assert(prune_l.size() == n_user_);

            double query_max_dim = query_vecs[0];
            for (int dim = 1; dim < check_dim_; dim++) {
                query_max_dim = std::max(query_max_dim, query_vecs[dim]);
            }
            query_max_dim = std::abs(query_max_dim);
            convert_coe_ = user_convert_coe_ * query_max_dim;

            double qratio = scale_ / query_max_dim;

            int query_int_sum = 0;
            int *query_int_vecs = query_int_ptr_.get();
            for (int dim = 0; dim < check_dim_; dim++) {
                query_int_vecs[dim] = std::floor(query_vecs[dim] * qratio);
                query_int_sum += std::abs(query_int_vecs[dim]);
            }


            const double *query_remain_vecs = query_vecs + check_dim_;
            double qright_norm = InnerProduct(query_remain_vecs, query_remain_vecs, remain_dim_);
            qright_norm = std::sqrt(qright_norm);

            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID]) {
                    continue;
                }
                int *user_int_vecs = user_int_ptr_.get() + userID * check_dim_;
                int leftIP = InnerProduct(user_int_vecs, query_int_vecs, check_dim_);
                int left_otherIP = user_int_sum_ptr_[userID] + query_int_sum;
                int lb_left_part = leftIP - left_otherIP;
                int ub_left_part = leftIP + left_otherIP;

                double right_norm = qright_norm * user_norm_l_[userID];

                double lower_bound = convert_coe_ * lb_left_part - right_norm;
                double upper_bound = convert_coe_ * ub_left_part + right_norm;

                ip_bound_l[userID] = std::make_pair(lower_bound, upper_bound);
                assert(lower_bound <= upper_bound);
            }

        }

    };

}
#endif //REVERSE_KRANKS_PARTINTPARTNORMPRUNE_HPP
