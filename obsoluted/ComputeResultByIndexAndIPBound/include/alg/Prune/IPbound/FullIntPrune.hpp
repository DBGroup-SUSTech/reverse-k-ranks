//
// Created by BianZheng on 2022/3/17.
//

#ifndef REVERSE_KRANKS_FULLINTPRUNE_HPP
#define REVERSE_KRANKS_FULLINTPRUNE_HPP

#include <cassert>
#include <memory>
#include <vector>
#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/RankBoundElement.hpp"

namespace ReverseMIPS {

    class FullIntPrune {
        int n_user_, vec_dim_, check_dim_, remain_dim_;
        double scale_;
        std::unique_ptr<int[]> user_int_ptr_;
        std::unique_ptr<std::pair<int, int>[]> user_int_sum_ptr_;

        std::unique_ptr<int[]> query_int_ptr_;
        std::pair<double, double> user_convert_coe_;

        std::pair<double, double> convert_coe_;
    public:
        FullIntPrune() = default;

        //make bound from offset_dim to vec_dim
        void Preprocess(const VectorMatrix &user, const int &check_dim, const double &scale) {
            this->n_user_ = user.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->check_dim_ = check_dim;
            this->remain_dim_ = vec_dim_ - check_dim_;
            assert(vec_dim_ > check_dim_);
            this->scale_ = scale;

            user_int_ptr_ = std::make_unique<int[]>(n_user_ * vec_dim_);
            user_int_sum_ptr_ = std::make_unique<std::pair<int, int>[]>(n_user_);
            query_int_ptr_ = std::make_unique<int[]>(vec_dim_);

            std::pair<double, double> user_max_dim_;
            user_max_dim_.first = user.getVector(0)[0];
            user_max_dim_.second = user.getVector(0)[check_dim];

            //compute the integer bound for the first part
            for (int userID = 0; userID < n_user_; userID++) {
                double *user_vecs = user.getVector(userID);
                for (int dim = 0; dim < check_dim; dim++) {
                    user_max_dim_.first = std::max(user_max_dim_.first, user_vecs[dim]);
                }
                for (int dim = check_dim; dim < vec_dim_; dim++) {
                    user_max_dim_.second = std::max(user_max_dim_.second, user_vecs[dim]);
                }
            }
            user_max_dim_.first = std::abs(user_max_dim_.first);
            user_max_dim_.second = std::abs(user_max_dim_.second);

            for (int userID = 0; userID < n_user_; userID++) {
                int *user_int_vecs = user_int_ptr_.get() + userID * vec_dim_;
                double *user_double_vecs = user.getVector(userID);
                user_int_sum_ptr_[userID].first = 0;
                for (int dim = 0; dim < check_dim; dim++) {
                    user_int_vecs[dim] = std::floor(user_double_vecs[dim] * scale_ / user_max_dim_.first);
                    user_int_sum_ptr_[userID].first += std::abs(user_int_vecs[dim]) + 1;
                }

                user_int_sum_ptr_[userID].second = 0;
                for (int dim = check_dim; dim < vec_dim_; dim++) {
                    user_int_vecs[dim] = std::floor(user_double_vecs[dim] * scale_ / user_max_dim_.second);
                    user_int_sum_ptr_[userID].second += std::abs(user_int_vecs[dim]) + 1;
                }

            }

            user_convert_coe_.first = user_max_dim_.first / (scale_ * scale_);
            user_convert_coe_.second = user_max_dim_.second / (scale_ * scale_);
        }

        void
        IPBound(const double *query_vecs, const VectorMatrix &user, const std::vector<bool> &prune_l,
                std::vector<std::pair<double, double>> &ip_bound_l) {
            assert(ip_bound_l.size() == n_user_);
            assert(prune_l.size() == n_user_);

            std::pair<double, double> query_max_dim(query_vecs[0], query_vecs[check_dim_]);
            for (int dim = 1; dim < check_dim_; dim++) {
                query_max_dim.first = std::max(query_max_dim.first, query_vecs[dim]);
            }
            for (int dim = check_dim_ + 1; dim < vec_dim_; dim++) {
                query_max_dim.second = std::max(query_max_dim.second, query_vecs[dim]);
            }
            query_max_dim.first = std::abs(query_max_dim.first);
            query_max_dim.second = std::abs(query_max_dim.second);
            convert_coe_ = std::make_pair(user_convert_coe_.first * query_max_dim.first,
                                          user_convert_coe_.second * query_max_dim.second);

            std::pair<double, double> qratio(scale_ / query_max_dim.first,
                                             scale_ / query_max_dim.second);

            std::pair<int, int> query_int_sum(0, 0);
            int *query_int_vecs = query_int_ptr_.get();
            for (int dim = 0; dim < check_dim_; dim++) {
                query_int_vecs[dim] = std::floor(query_vecs[dim] * qratio.first);
                query_int_sum.first += std::abs(query_int_vecs[dim]);
            }

            for (int dim = check_dim_; dim < vec_dim_; dim++) {
                query_int_vecs[dim] = std::floor(query_vecs[dim] * qratio.second);
                query_int_sum.second += std::abs(query_int_vecs[dim]);
            }

            int *query_remain_int_vecs = query_int_vecs + check_dim_;
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID]) {
                    continue;
                }
                int *user_int_vecs = user_int_ptr_.get() + userID * vec_dim_;
                int leftIP = InnerProduct(user_int_vecs, query_int_vecs, check_dim_);
                int left_otherIP = user_int_sum_ptr_[userID].first + query_int_sum.first;
                int lb_left_part = leftIP - left_otherIP;
                int ub_left_part = leftIP + left_otherIP;

                int *user_remain_int_vecs = user_int_vecs + check_dim_;
                int rightIP = InnerProduct(user_remain_int_vecs, query_remain_int_vecs, remain_dim_);
                int right_otherIP = user_int_sum_ptr_[userID].second + query_int_sum.second;
                int lb_right_part = rightIP - right_otherIP;
                int ub_right_part = rightIP + right_otherIP;

                double lower_bound = convert_coe_.first * lb_left_part + convert_coe_.second * lb_right_part;
                double upper_bound = convert_coe_.first * ub_left_part + convert_coe_.second * ub_right_part;

                ip_bound_l[userID] = std::make_pair(lower_bound, upper_bound);
                assert(lower_bound <= upper_bound);
            }

        }

    };

}
#endif //REVERSE_KRANKS_FULLINTPRUNE_HPP
