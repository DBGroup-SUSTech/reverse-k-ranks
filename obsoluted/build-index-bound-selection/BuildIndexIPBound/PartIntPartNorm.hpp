//
// Created by BianZheng on 2022/5/19.
//

#ifndef REVERSE_KRANKS_PARTINTPARTNORM_HPP
#define REVERSE_KRANKS_PARTINTPARTNORM_HPP

#include <cfloat>

#include "BaseIPBound.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "alg/SVD.hpp"

namespace ReverseMIPS {

    class PartIntPartNorm : public BaseIPBound {
        int n_user_, n_data_item_, vec_dim_;
        //SVD
        SVD svd_ins_;
        int check_dim_;

        //PartIntPartNorm
        double scale_;
        std::unique_ptr<int[]> user_int_ptr_; // n_user_ * check_dim_
        std::unique_ptr<int[]> user_int_sum_ptr_; // n_user_
        std::unique_ptr<int[]> item_int_ptr_; // n_data_item_ * check_dim_
        std::unique_ptr<int[]> item_int_sum_ptr_; // n_data_item_
        double convert_coe_;

        //norm
        std::unique_ptr<double[]> user_norm_l_; // n_user_
        std::unique_ptr<double[]> item_norm_l_; // n_data_item_

    public:

        inline PartIntPartNorm() {
            n_user_ = -1;
            n_data_item_ = -1;
            vec_dim_ = -1;
            check_dim_ = -1;

            scale_ = -1;
            user_int_ptr_ = nullptr;
            user_int_sum_ptr_ = nullptr;
            item_int_ptr_ = nullptr;
            item_int_sum_ptr_ = nullptr;
            convert_coe_ = -1;

            user_norm_l_ = nullptr;
            item_norm_l_ = nullptr;
        };

        inline PartIntPartNorm(const int &n_user, const int &n_data_item, const int &vec_dim, const double &scale) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->vec_dim_ = vec_dim;
            this->scale_ = scale;
        }

        [[nodiscard]] double MatrixMaxVal(const VectorMatrix &vm) const {
            double max_dim = vm.getVector(0)[0];
            const int &n_vector = vm.n_vector_;
            const int &vec_dim = vm.vec_dim_;

            for (int vecsID = 0; vecsID < n_vector; vecsID++) {
                const double *vecs = vm.getVector(vecsID);
                for (int dim = 0; dim < check_dim_; dim++) {
                    max_dim = std::max(max_dim, vecs[dim]);
                }
            }
            max_dim = std::abs(max_dim);
            return max_dim;
        }

        void IntArrayVal(const VectorMatrix &vm, const double &max_dim_pair,
                         int *vm_int_ptr, int *vm_int_sum_ptr) const {
            const int &n_vector = vm.n_vector_;
            const int &vec_dim = vm.vec_dim_;
            for (int vecsID = 0; vecsID < n_vector; vecsID++) {
                int *tmp_int_vecs = vm_int_ptr + vecsID * check_dim_;
                double *tmp_double_vecs = vm.getVector(vecsID);
                vm_int_sum_ptr[vecsID] = 0;
                for (int dim = 0; dim < check_dim_; dim++) {
                    tmp_int_vecs[dim] = std::floor(tmp_double_vecs[dim] * scale_ / max_dim_pair);
                    vm_int_sum_ptr[vecsID] += std::abs(tmp_int_vecs[dim]);
                }

            }
        }

        void Preprocess(VectorMatrix &user, VectorMatrix &data_item) override {
            const double SIGMA = 0.7;
            check_dim_ = svd_ins_.Preprocess(user, data_item, SIGMA);
            int remain_dim = vec_dim_ - check_dim_;

            //int
            user_int_ptr_ = std::make_unique<int[]>(n_user_ * check_dim_);
            user_int_sum_ptr_ = std::make_unique<int[]>(n_user_);
            item_int_ptr_ = std::make_unique<int[]>(n_data_item_ * check_dim_);
            item_int_sum_ptr_ = std::make_unique<int[]>(n_data_item_);

            const double user_max_dim = MatrixMaxVal(user);
            const double item_max_dim = MatrixMaxVal(data_item);

            IntArrayVal(user, user_max_dim, user_int_ptr_.get(), user_int_sum_ptr_.get());

            for (int userID = 0; userID < n_user_; userID++) {
                user_int_sum_ptr_[userID] += check_dim_;
            }

            IntArrayVal(data_item, item_max_dim, item_int_ptr_.get(), item_int_sum_ptr_.get());

            convert_coe_ = user_max_dim * item_max_dim / (scale_ * scale_);

            //norm
            user_norm_l_ = std::make_unique<double[]>(n_user_);
            item_norm_l_ = std::make_unique<double[]>(n_data_item_);

            for (int userID = 0; userID < n_user_; userID++) {
                const double *user_vecs = user.getVector(userID, check_dim_);
                const double IP = InnerProduct(user_vecs, user_vecs, remain_dim);
                const double norm = std::sqrt(IP);
                user_norm_l_[userID] = norm;
            }

            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                const double *item_vecs = data_item.getVector(itemID, check_dim_);
                const double IP = InnerProduct(item_vecs, item_vecs, remain_dim);
                const double norm = std::sqrt(IP);
                item_norm_l_[itemID] = norm;
            }

        }

        void PreprocessQuery(const double *query_vecs, const int &vec_dim, double *query_write_vecs) override {
            svd_ins_.TransferQuery(query_vecs, vec_dim_, query_write_vecs);
        }

        double IPUpperBound(const double *user_vecs, const int &userID,
                            const double *item_vecs, const int &itemID) override {
            const int *user_int_vecs = user_int_ptr_.get() + userID * check_dim_;
            const int *item_int_vecs = item_int_ptr_.get() + itemID * check_dim_;

            int leftIP = InnerProduct(user_int_vecs, item_int_vecs, check_dim_);
            int left_otherIP = user_int_sum_ptr_[userID] + item_int_sum_ptr_[itemID];
            int ub_left_part = leftIP + left_otherIP;

            double rightIP = user_norm_l_[userID] * item_norm_l_[itemID];
            double upper_bound = convert_coe_ * ub_left_part + rightIP;
            return upper_bound;
        }

        double IPLowerBound(const double *user_vecs, const int &userID,
                            const double *item_vecs, const int &itemID) override {
            const int *user_int_vecs = user_int_ptr_.get() + userID * check_dim_;
            const int *item_int_vecs = item_int_ptr_.get() + itemID * check_dim_;

            int leftIP = InnerProduct(user_int_vecs, item_int_vecs, check_dim_);
            int left_otherIP = user_int_sum_ptr_[userID] + item_int_sum_ptr_[itemID];
            int lb_left_part = leftIP - left_otherIP;

            double rightIP = -user_norm_l_[userID] * item_norm_l_[itemID];
            double lower_bound = convert_coe_ * lb_left_part + rightIP;
            return lower_bound;
        }

        std::pair<double, double>
        IPBound(const double *user_vecs, const int &userID, const double *item_vecs, const int &itemID) override {
            const int *user_int_vecs = user_int_ptr_.get() + userID * check_dim_;
            const int *item_int_vecs = item_int_ptr_.get() + itemID * check_dim_;

            int leftIP = InnerProduct(user_int_vecs, item_int_vecs, check_dim_);
            int left_otherIP = user_int_sum_ptr_[userID] + item_int_sum_ptr_[itemID];
            int lb_left_part = leftIP - left_otherIP;
            int ub_left_part = leftIP + left_otherIP;

            double rightIP_lb = -user_norm_l_[userID] * item_norm_l_[itemID];
            double rightIP_ub = user_norm_l_[userID] * item_norm_l_[itemID];

            double lower_bound = convert_coe_ * lb_left_part + rightIP_lb;
            double upper_bound = convert_coe_ * ub_left_part + rightIP_ub;

            return std::make_pair(lower_bound, upper_bound);
        }

        void
        IPBound(const double *user_vecs, const int &userID,
                const std::vector<int> &item_cand_l,
                const VectorMatrix &item,
                std::pair<double, double> *IPbound_l) override {
            for (const int &itemID: item_cand_l) {
                const double *item_vecs = item.getVector(itemID);
                IPbound_l[itemID] = IPBound(user_vecs, userID, item_vecs, itemID);
            }
        }

    };
}

#endif //REVERSE_KRANKS_PARTINTPARTNORM_HPP
