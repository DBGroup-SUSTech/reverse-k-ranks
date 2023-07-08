//
// Created by bianzheng on 2023/6/16.
//

#ifndef REVERSE_KRANKS_PARTDIMPARTNORMUPDATE_HPP
#define REVERSE_KRANKS_PARTDIMPARTNORMUPDATE_HPP

#include <cfloat>

#include "BaseQueryIPBound.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "alg/SVD.hpp"

namespace ReverseMIPS {

    class PartDimPartNormUpdate {
        int n_user_, vec_dim_;
        //SVD variable
        int check_dim_, remain_dim_;
        int n_thread_;
        //IPBound prune
        std::unique_ptr<char[]> use_transfer_query_l_; // n_user
        std::unique_ptr<float[]> user_norm_l_; // n_user
        std::unique_ptr<float[]> checkdim_IP_l_; // n_user

    public:

        inline PartDimPartNormUpdate() = default;

        inline PartDimPartNormUpdate(const int &n_user, const int &vec_dim, const int &check_dim,
                                     const int n_thread = omp_get_num_procs()) {
            this->n_user_ = n_user;
            this->vec_dim_ = vec_dim;
            this->check_dim_ = check_dim;
            this->remain_dim_ = vec_dim - check_dim;
            this->n_thread_ = n_thread;

            user_norm_l_ = std::make_unique<float[]>(n_user_);
            checkdim_IP_l_ = std::make_unique<float[]>(n_user_);
            use_transfer_query_l_ = std::make_unique<char[]>(n_user_);
            for (int userID = 0; userID < n_user_; userID++) {
                use_transfer_query_l_[userID] = true;
            }

        }

        void Preprocess(const VectorMatrixUpdate &user) {
#pragma omp parallel for default(none) shared(user)
            for (int userID = 0; userID < n_user_; userID++) {
                float right_norm = InnerProduct(user.getVector(userID, check_dim_), user.getVector(userID, check_dim_),
                                                remain_dim_);
                right_norm = std::sqrt(right_norm);
                user_norm_l_[userID] = right_norm;
            }
        }

        void InsertUser(const VectorMatrixUpdate &insert_user) {
            std::unique_ptr<float[]> new_user_norm_l = std::make_unique<float[]>(n_user_ + insert_user.n_vector_);
            std::unique_ptr<float[]> new_checkdim_IP_l = std::make_unique<float[]>(n_user_ + insert_user.n_vector_);
            std::unique_ptr<char[]> new_use_transfer_query_l = std::make_unique<char[]>(
                    n_user_ + insert_user.n_vector_);

            std::memcpy(new_user_norm_l.get(), user_norm_l_.get(), n_user_ * sizeof(float));
            std::memcpy(new_use_transfer_query_l.get(), use_transfer_query_l_.get(), n_user_ * sizeof(char));

            for (int insert_userID = 0; insert_userID < insert_user.n_vector_; insert_userID++) {
                float right_norm = InnerProduct(insert_user.getVector(insert_userID, check_dim_),
                                                insert_user.getVector(insert_userID, check_dim_), remain_dim_);
                right_norm = std::sqrt(right_norm);
                new_user_norm_l[n_user_ + insert_userID] = right_norm;
                new_use_transfer_query_l[n_user_ + insert_userID] = false;
            }

            user_norm_l_ = std::move(new_user_norm_l);
            use_transfer_query_l_ = std::move(new_use_transfer_query_l);
            checkdim_IP_l_ = std::move(new_checkdim_IP_l);

            n_user_ += insert_user.n_vector_;

        }

        void DeleteUser(const std::vector<int> &del_userID_l) {
            std::unique_ptr<float[]> new_user_norm_l = std::make_unique<float[]>(n_user_ + del_userID_l.size());
            std::unique_ptr<float[]> new_checkdim_IP_l = std::make_unique<float[]>(n_user_ + del_userID_l.size());
            std::unique_ptr<char[]> new_use_transfer_query_l = std::make_unique<char[]>(
                    n_user_ + del_userID_l.size());

            std::unordered_set<int> del_userID_set(del_userID_l.begin(), del_userID_l.end());
            std::vector<int> remain_userID_l;
            for (int userID = 0; userID < n_user_; userID++) {
                if (del_userID_set.find(userID) == del_userID_set.end()) {
                    remain_userID_l.push_back(userID);
                }
            }
            assert(remain_userID_l.size() + del_userID_l.size() == n_user_);

            for (int candID = 0; candID < remain_userID_l.size(); candID++) {
                const int userID = remain_userID_l[candID];
                new_user_norm_l[candID] = user_norm_l_[userID];
                new_use_transfer_query_l[candID] = use_transfer_query_l_[userID];
            }

            user_norm_l_ = std::move(new_user_norm_l);
            checkdim_IP_l_ = std::move(new_checkdim_IP_l);
            use_transfer_query_l_ = std::move(new_use_transfer_query_l);

            n_user_ -= del_userID_l.size();
        }

        void
        IPBound(const float *query_transfer_vecs,
                const float *query_no_transfer_vecs,
                const VectorMatrixUpdate &user,
                std::vector<std::pair<float, float>> &queryIP_l, const int &n_proc_user, const int& queryID) const {

            const float query_transfer_norm = std::sqrt(
                    InnerProduct(query_transfer_vecs + check_dim_,
                                 query_transfer_vecs + check_dim_, remain_dim_));
            const float query_no_transfer_norm = std::sqrt(
                    InnerProduct(query_no_transfer_vecs + check_dim_,
                                 query_no_transfer_vecs + check_dim_, remain_dim_));

#pragma omp parallel for default(none) shared(n_proc_user, user, query_transfer_vecs, query_no_transfer_vecs, query_transfer_norm, query_no_transfer_norm, queryIP_l) num_threads(n_thread_)
            for (int userID = 0; userID < n_proc_user; userID++) {
                const float *user_vecs = user.getVector(userID);
                float left_IP, right_IP;
                if (use_transfer_query_l_[userID]) {
                    left_IP = InnerProduct(user_vecs, query_transfer_vecs, check_dim_);
                    right_IP = user_norm_l_[userID] * query_transfer_norm;
                } else {
                    left_IP = InnerProduct(user_vecs, query_no_transfer_vecs, check_dim_);
                    right_IP = user_norm_l_[userID] * query_no_transfer_norm;
                }

                checkdim_IP_l_[userID] = left_IP;

                const float lower_bound = left_IP - right_IP;
                const float upper_bound = left_IP + right_IP;
                queryIP_l[userID] = std::make_pair(lower_bound, upper_bound);
            }

        }

        void ComputeRemainDim(const float *query_transfer_vecs,
                              const float *query_no_transfer_vecs,
                              const VectorMatrixUpdate &user,
                              const std::vector<char> &prune_l, const std::vector<char> &result_l,
                              std::vector<float> &queryIP_l, const int &n_proc_user) {

#pragma omp parallel for default(none) shared(n_proc_user, prune_l, result_l, user, query_transfer_vecs, query_no_transfer_vecs, queryIP_l) num_threads(n_thread_)
            for (int userID = 0; userID < n_proc_user; userID++) {
                if (prune_l[userID] || result_l[userID]) {
                    continue;
                }
                const float *user_check_vecs = user.getVector(userID, check_dim_);

                const float *query_check_vecs = use_transfer_query_l_[userID] ? query_transfer_vecs + check_dim_ :
                                                query_no_transfer_vecs + check_dim_;

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
#endif //REVERSE_KRANKS_PARTDIMPARTNORMUPDATE_HPP
