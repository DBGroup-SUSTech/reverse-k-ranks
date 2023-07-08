//
// Created by BianZheng on 2022/5/19.
//

#ifndef REVERSE_KRANKS_PARTDIMPARTNORM_HPP
#define REVERSE_KRANKS_PARTDIMPARTNORM_HPP

#include <cfloat>

#include "BaseQueryIPBound.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "alg/SVD.hpp"

namespace ReverseMIPS {

    class PartDimPartNorm {
        int n_user_, vec_dim_;
        //SVD variable
        int check_dim_, remain_dim_;
        int n_thread_;
        //IPBound prune
        std::unique_ptr<float[]> user_norm_l_;
        std::unique_ptr<float[]> checkdim_IP_l_;

    public:

        inline PartDimPartNorm() = default;

        inline PartDimPartNorm(const int &n_user, const int &vec_dim, const int &check_dim,
                               const int n_thread = omp_get_num_procs()) {
            this->n_user_ = n_user;
            this->vec_dim_ = vec_dim;
            this->check_dim_ = check_dim;
            this->remain_dim_ = vec_dim - check_dim;
            this->n_thread_ = n_thread;

            user_norm_l_ = std::make_unique<float[]>(n_user_);
            checkdim_IP_l_ = std::make_unique<float[]>(n_user_);
        }

        void Preprocess(const VectorMatrix &user) {
#pragma omp parallel for default(none) shared(user)
            for (int userID = 0; userID < n_user_; userID++) {
                float right_norm = InnerProduct(user.getVector(userID, check_dim_), user.getVector(userID, check_dim_),
                                                remain_dim_);
                right_norm = std::sqrt(right_norm);
                user_norm_l_[userID] = right_norm;
            }

        }

        void
        IPBound(const float *query_vecs, const VectorMatrix &user,
                std::vector<std::pair<float, float>> &queryIP_l, const int &n_proc_user) const {

            const float query_norm = std::sqrt(
                    InnerProduct(query_vecs + check_dim_, query_vecs + check_dim_, remain_dim_));

#pragma omp parallel for default(none) shared(n_proc_user, user, query_vecs, query_norm, queryIP_l) num_threads(n_thread_)
            for (int userID = 0; userID < n_proc_user; userID++) {
                const float *user_vecs = user.getVector(userID);
                const float left_IP = InnerProduct(user_vecs, query_vecs, check_dim_);

                checkdim_IP_l_[userID] = left_IP;

                const float user_norm = user_norm_l_[userID];
                const float right_IP = user_norm * query_norm;

                const float lower_bound = left_IP - right_IP;
                const float upper_bound = left_IP + right_IP;
                queryIP_l[userID] = std::make_pair(lower_bound, upper_bound);
            }

        }

        void ComputeRemainDim(const float *query_vecs, const VectorMatrix &user,
                              const std::vector<char> &prune_l, const std::vector<char> &result_l,
                              std::vector<float> &queryIP_l, const int &n_proc_user) {

            const float *query_check_vecs = query_vecs + check_dim_;

#pragma omp parallel for default(none) shared(n_proc_user, prune_l, result_l, user, query_check_vecs, queryIP_l) num_threads(n_thread_)
            for (int userID = 0; userID < n_proc_user; userID++) {
                if (prune_l[userID] || result_l[userID]) {
                    continue;
                }
                const float *user_check_vecs = user.getVector(userID, check_dim_);

                const float left_IP = checkdim_IP_l_[userID];

                const float right_IP = InnerProduct(user_check_vecs, query_check_vecs, remain_dim_);
                queryIP_l[userID] = left_IP + right_IP;
            }

        }

        uint64_t IndexSizeByte() {
            uint64_t index_size = sizeof(float) * n_user_;
            return index_size;
        }

    };
}
#endif //REVERSE_KRANKS_PARTDIMPARTNORM_HPP
