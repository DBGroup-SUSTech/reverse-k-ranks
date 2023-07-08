//
// Created by bianzheng on 2023/4/12.
//

#ifndef REVERSE_KRANKS_PARTDIMPARTNORMREFINE_HPP
#define REVERSE_KRANKS_PARTDIMPARTNORMREFINE_HPP

#include <cfloat>

#include "alg/SpaceInnerProduct.hpp"
#include "alg/SVD.hpp"

namespace ReverseMIPS {

    class RefineCPUPartDimPartNorm {
        int n_user_, n_data_item_, vec_dim_;
        //SVD variable
        int check_dim_, remain_dim_;
        //IPBound prune
        std::unique_ptr<float[]> user_norm_l_;
        std::unique_ptr<float[]> item_norm_l_;

    public:

        inline RefineCPUPartDimPartNorm() = default;

        inline RefineCPUPartDimPartNorm(const int &n_user, const int &n_data_item, const int &vec_dim,
                                        const int &check_dim) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->vec_dim_ = vec_dim;
            this->check_dim_ = check_dim;
            this->remain_dim_ = vec_dim - check_dim;

            user_norm_l_ = std::make_unique<float[]>(n_user_);
            item_norm_l_ = std::make_unique<float[]>(n_data_item_);
        }

        void Preprocess(const VectorMatrix &user, const VectorMatrix &data_item) {
            for (int userID = 0; userID < n_user_; userID++) {
                float right_norm = InnerProduct(user.getVector(userID, check_dim_), user.getVector(userID, check_dim_),
                                                remain_dim_);
                right_norm = std::sqrt(right_norm);
                user_norm_l_[userID] = right_norm;
            }

            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                float right_norm = InnerProduct(data_item.getVector(itemID, check_dim_),
                                                data_item.getVector(itemID, check_dim_), remain_dim_);
                right_norm = std::sqrt(right_norm);
                item_norm_l_[itemID] = right_norm;
            }

        }

        void
        IPBound(const VectorMatrix &data_item, const int &n_proc_item, const float *user_vecs, const int &userID,
                std::vector<std::pair<float, float>> &itemIP_l, std::vector<float> &checkdimIP_l) const {

            assert(itemIP_l.size() == n_data_item_);
            assert(checkdimIP_l.size() == n_data_item_);

            const float user_norm = user_norm_l_[userID];

#pragma omp parallel for default(none) shared(n_proc_item, data_item, user_vecs, itemIP_l, checkdimIP_l, user_norm)
            for (int itemID = 0; itemID < n_proc_item; itemID++) {
                const float *item_vecs = data_item.getVector(itemID);
                const float left_IP = InnerProduct(user_vecs, item_vecs, check_dim_);

                checkdimIP_l[itemID] = left_IP;

                const float item_norm = item_norm_l_[itemID];
                const float right_IP = item_norm * user_norm;

                const float lower_bound = left_IP - right_IP;
                const float upper_bound = left_IP + right_IP;
                itemIP_l[itemID] = std::make_pair(lower_bound, upper_bound);
            }

        }

        void
        IPBoundNoParallel(const VectorMatrix &data_item, const int &n_proc_item, const float *user_vecs,
                          const int &userID,
                          std::vector<std::pair<float, float>> &itemIP_l, std::vector<float> &checkdimIP_l) const {

            assert(itemIP_l.size() == n_data_item_);
            assert(checkdimIP_l.size() == n_data_item_);

            const float user_norm = user_norm_l_[userID];

//#pragma omp parallel for default(none) shared(n_proc_item, data_item, user_vecs, itemIP_l, checkdimIP_l, user_norm)
            for (int itemID = 0; itemID < n_proc_item; itemID++) {
                const float *item_vecs = data_item.getVector(itemID);
                const float left_IP = InnerProduct(user_vecs, item_vecs, check_dim_);

                checkdimIP_l[itemID] = left_IP;

                const float item_norm = item_norm_l_[itemID];
                const float right_IP = item_norm * user_norm;

                const float lower_bound = left_IP - right_IP;
                const float upper_bound = left_IP + right_IP;
                itemIP_l[itemID] = std::make_pair(lower_bound, upper_bound);
            }

        }

        void ComputeRemainDim(const VectorMatrix &data_item, const int &n_proc_item,
                              const float *user_vecs,
                              const std::vector<bool> &is_rank_larger_l, const std::vector<bool> &is_rank_lower_l,
                              std::vector<float> &queryIP_l) const {

            const float *user_check_vecs = user_vecs + check_dim_;

#pragma omp parallel for default(none) shared(n_proc_item, data_item, user_check_vecs, is_rank_larger_l, is_rank_lower_l, queryIP_l)
            for (int itemID = 0; itemID < n_proc_item; itemID++) {
                if (is_rank_larger_l[itemID] || is_rank_lower_l[itemID]) {
                    continue;
                }
                const float *item_check_vecs = data_item.getVector(itemID, check_dim_);

                const float left_IP = queryIP_l[itemID];

                const float right_IP = InnerProduct(user_check_vecs, item_check_vecs, remain_dim_);
                queryIP_l[itemID] = left_IP + right_IP;
            }

        }

        void ComputeRemainDimNoParallel(const VectorMatrix &data_item, const int &n_proc_item,
                                        const float *user_vecs,
                                        const std::vector<char> &is_rank_larger_l,
                                        const std::vector<char> &is_rank_lower_l,
                                        std::vector<float> &queryIP_l) const {

            const float *user_check_vecs = user_vecs + check_dim_;

//#pragma omp parallel for default(none) shared(n_proc_item, data_item, user_check_vecs, is_rank_larger_l, is_rank_lower_l, queryIP_l)
            for (int itemID = 0; itemID < n_proc_item; itemID++) {
                if (is_rank_larger_l[itemID] || is_rank_lower_l[itemID]) {
                    continue;
                }
                const float *item_check_vecs = data_item.getVector(itemID, check_dim_);

                const float left_IP = queryIP_l[itemID];

                const float right_IP = InnerProduct(user_check_vecs, item_check_vecs, remain_dim_);
                queryIP_l[itemID] = left_IP + right_IP;
            }

        }

        uint64_t IndexSizeByte() {
            uint64_t index_size = sizeof(float) * n_user_;
            return index_size;
        }

    };
}
#endif //REVERSE_KRANKS_PARTDIMPARTNORMREFINE_HPP
