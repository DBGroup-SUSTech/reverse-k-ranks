//
// Created by BianZheng on 2022/3/17.
//

#ifndef REVERSE_KRANKS_FULLINT_HPP
#define REVERSE_KRANKS_FULLINT_HPP

#include <cassert>
#include <memory>
#include <vector>
#include <omp.h>

#include "alg/QueryIPBound/BaseQueryIPBound.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/RankBoundElement.hpp"

namespace ReverseMIPS {

    class FullInt : public BaseQueryIPBound {
        int n_user_, vec_dim_;
        float scale_;
        std::unique_ptr<int[]> user_int_ptr_;
        std::unique_ptr<int[]> user_int_sum_ptr_;

        float user_convert_coe_;

    public:
        FullInt() = default;

        FullInt(const int &n_user, const int &vec_dim, const float &scale) {
            this->n_user_ = n_user;
            this->vec_dim_ = vec_dim;
            this->scale_ = scale;
            user_int_ptr_ = std::make_unique<int[]>(n_user_ * vec_dim_);
            user_int_sum_ptr_ = std::make_unique<int[]>(n_user_);
        }

        //make bound from offset_dim to vec_dim
        void Preprocess(const VectorMatrix &user) override {

            float user_max_dim_ = user.getVector(0)[0];

            //compute the integer bound for the first part
            for (int userID = 0; userID < n_user_; userID++) {
                float *user_vecs = user.getVector(userID);
                for (int dim = 0; dim < vec_dim_; dim++) {
                    user_max_dim_ = std::max(user_max_dim_, user_vecs[dim]);
                }
            }
            user_max_dim_ = std::abs(user_max_dim_);

#pragma omp parallel for default(none) shared(user, user_max_dim_)
            for (int userID = 0; userID < n_user_; userID++) {
                int *user_int_vecs = user_int_ptr_.get() + userID * vec_dim_;
                float *user_double_vecs = user.getVector(userID);
                user_int_sum_ptr_[userID] = 0;
                for (int dim = 0; dim < vec_dim_; dim++) {
                    user_int_vecs[dim] = std::floor(user_double_vecs[dim] * scale_ / user_max_dim_);
                    user_int_sum_ptr_[userID] += std::abs(user_int_vecs[dim]) + 1;
                }

            }

            user_convert_coe_ = user_max_dim_ / (scale_ * scale_);
        }

        void
        IPBound(const float *query_vecs, const VectorMatrix &user,
                std::vector<std::pair<float, float>> &ip_bound_l, const int &n_proc_user) const override {
            assert(ip_bound_l.size() == n_user_);
            assert(n_proc_user <= n_user_);

            float query_max_dim = query_vecs[0];
//#pragma omp parallel for default(none) shared(query_vecs) reduction(max:query_max_dim)
            for (int dim = 1; dim < vec_dim_; dim++) {
                query_max_dim = std::max(query_max_dim, query_vecs[dim]);
            }
            query_max_dim = std::abs(query_max_dim);
            const float convert_coe = user_convert_coe_ * query_max_dim;

            const float qratio = scale_ / query_max_dim;

            int query_int_sum = 0;
            std::unique_ptr<int[]> query_int_ptr_ = std::make_unique<int[]>(vec_dim_);
            int *query_int_vecs = query_int_ptr_.get();
//#pragma omp parallel for default(none) shared(query_vecs, qratio, query_int_vecs) reduction(+:query_int_sum)
            for (int dim = 0; dim < vec_dim_; dim++) {
                query_int_vecs[dim] = std::floor(query_vecs[dim] * qratio);
                query_int_sum += std::abs(query_int_vecs[dim]);
            }

#pragma omp parallel for default(none) shared(n_proc_user, convert_coe, ip_bound_l, query_int_vecs, query_int_sum) num_threads(omp_get_num_procs())
            for (int userID = 0; userID < n_proc_user; userID++) {
                int *user_int_vecs = user_int_ptr_.get() + userID * vec_dim_;
                int intIP = InnerProduct(user_int_vecs, query_int_vecs, vec_dim_);
                int int_otherIP = user_int_sum_ptr_[userID] + query_int_sum;
                int lb_part = intIP - int_otherIP;
                int ub_part = intIP + int_otherIP;

                float lower_bound = convert_coe * (float) lb_part;
                float upper_bound = convert_coe * (float) ub_part;

                ip_bound_l[userID] = std::make_pair(lower_bound, upper_bound);
                assert(lower_bound <= upper_bound);
            }

        }

        uint64_t IndexSizeByte() const override {
            const uint64_t user_int_size = sizeof(int) * n_user_ * vec_dim_;
            const uint64_t user_int_sum_size = sizeof(int) * n_user_;
            return user_int_size + user_int_sum_size;
        }

    };

}
#endif //REVERSE_KRANKS_FULLINT_HPP
