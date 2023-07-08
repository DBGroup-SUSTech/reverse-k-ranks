//
// Created by BianZheng on 2022/5/19.
//

#ifndef REVERSE_KRANKS_FULLINT_HPP
#define REVERSE_KRANKS_FULLINT_HPP

#include <cfloat>

#include "BaseIPBound.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "alg/SVD.hpp"

namespace ReverseMIPS {

    class FullInt : public BaseIPBound {
        int n_user_, n_data_item_, vec_dim_;
        //SVD
        SVD svd_ins_;
        int check_dim_, remain_dim_;

        //FullInt
        double scale_;
        std::unique_ptr<int[]> user_int_ptr_; // n_user_ * vec_dim_
        std::unique_ptr<std::pair<int, int>[]> user_int_sum_ptr_; // n_user_
        std::unique_ptr<int[]> item_int_ptr_; // n_data_item_ * vec_dim_
        std::unique_ptr<std::pair<int, int>[]> item_int_sum_ptr_; // n_data_item_
        std::pair<double, double> convert_coe_;

    public:

        inline FullInt() {
            n_user_ = -1;
            n_data_item_ = -1;
            vec_dim_ = -1;
            check_dim_ = -1;
            remain_dim_ = -1;

            user_int_ptr_ = nullptr;
            user_int_sum_ptr_ = nullptr;
            item_int_ptr_ = nullptr;
            item_int_sum_ptr_ = nullptr;
            convert_coe_ = std::make_pair(-1, -1);
        };

        inline FullInt(const int &n_user, const int &n_data_item, const int &vec_dim, const double &scale) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->vec_dim_ = vec_dim;
            this->scale_ = scale;

            user_int_ptr_ = std::make_unique<int[]>(n_user_ * vec_dim_);
            user_int_sum_ptr_ = std::make_unique<std::pair<int, int>[]>(n_user_);
            item_int_ptr_ = std::make_unique<int[]>(n_data_item_ * vec_dim_);
            item_int_sum_ptr_ = std::make_unique<std::pair<int, int>[]>(n_data_item_);
        }

        [[nodiscard]] std::pair<double, double> MatrixPartMaxVal(const VectorMatrix &vm) const {
            std::pair<double, double> max_dim_pair;
            max_dim_pair.first = vm.getVector(0)[0];
            max_dim_pair.second = vm.getVector(0)[check_dim_];
            const int &n_vector = vm.n_vector_;
            const int &vec_dim = vm.vec_dim_;

            for (int vecsID = 0; vecsID < n_vector; vecsID++) {
                const double *vecs = vm.getVector(vecsID);
                for (int dim = 0; dim < check_dim_; dim++) {
                    max_dim_pair.first = std::max(max_dim_pair.first, vecs[dim]);
                }
                for (int dim = check_dim_; dim < vec_dim; dim++) {
                    max_dim_pair.second = std::max(max_dim_pair.second, vecs[dim]);
                }
            }
            max_dim_pair.first = std::abs(max_dim_pair.first);
            max_dim_pair.second = std::abs(max_dim_pair.second);
            return max_dim_pair;
        }

        void IntArrayVal(const VectorMatrix &vm, const std::pair<double, double> &max_dim_pair,
                         int *vm_int_ptr, std::pair<int, int> *vm_int_sum_ptr) const {
            const int &n_vector = vm.n_vector_;
            const int &vec_dim = vm.vec_dim_;
            for (int vecsID = 0; vecsID < n_vector; vecsID++) {
                int *tmp_int_vecs = vm_int_ptr + vecsID * vec_dim;
                double *tmp_double_vecs = vm.getVector(vecsID);
                vm_int_sum_ptr[vecsID].first = 0;
                for (int dim = 0; dim < check_dim_; dim++) {
                    tmp_int_vecs[dim] = std::floor(tmp_double_vecs[dim] * scale_ / max_dim_pair.first);
                    vm_int_sum_ptr[vecsID].first += std::abs(tmp_int_vecs[dim]);
                }

                vm_int_sum_ptr[vecsID].second = 0;
                for (int dim = check_dim_; dim < vec_dim_; dim++) {
                    tmp_int_vecs[dim] = std::floor(tmp_double_vecs[dim] * scale_ / max_dim_pair.second);
                    vm_int_sum_ptr[vecsID].second += std::abs(tmp_int_vecs[dim]);
                }
            }
        }

        void Preprocess(VectorMatrix &user, VectorMatrix &data_item) override {
            const double SIGMA = 0.7;
            check_dim_ = svd_ins_.Preprocess(user, data_item, SIGMA);
            remain_dim_ = vec_dim_ - check_dim_;

            std::pair<double, double> user_max_dim_pair = MatrixPartMaxVal(user);
            std::pair<double, double> item_max_dim_pair = MatrixPartMaxVal(data_item);

            IntArrayVal(user, user_max_dim_pair, user_int_ptr_.get(), user_int_sum_ptr_.get());

            for (int userID = 0; userID < n_user_; userID++) {
                user_int_sum_ptr_[userID].first += check_dim_;
                user_int_sum_ptr_[userID].second += remain_dim_;
            }

            IntArrayVal(data_item, item_max_dim_pair, item_int_ptr_.get(), item_int_sum_ptr_.get());

            convert_coe_.first = user_max_dim_pair.first * item_max_dim_pair.first / (scale_ * scale_);
            convert_coe_.second = user_max_dim_pair.second * item_max_dim_pair.second / (scale_ * scale_);

        }

        void PreprocessQuery(const double *query_vecs, const int &vec_dim, double *query_write_vecs) override {
            svd_ins_.TransferQuery(query_vecs, vec_dim_, query_write_vecs);
        }

        double IPUpperBound(const double *user_vecs, const int &userID,
                            const double *item_vecs, const int &itemID) override {
            const int *user_int_vecs = user_int_ptr_.get() + userID * vec_dim_;
            const int *item_int_vecs = item_int_ptr_.get() + itemID * vec_dim_;

            int leftIP = InnerProduct(user_int_vecs, item_int_vecs, check_dim_);
            int left_otherIP = user_int_sum_ptr_[userID].first + item_int_sum_ptr_[itemID].first;
            int ub_left_part = leftIP + left_otherIP;

            const int *user_remain_int_vecs = user_int_vecs + check_dim_;
            const int *item_remain_int_vecs = item_int_vecs + check_dim_;

            int rightIP = InnerProduct(user_remain_int_vecs, item_remain_int_vecs, remain_dim_);
            int right_otherIP = user_int_sum_ptr_[userID].second + item_int_sum_ptr_[itemID].second;
            int ub_right_part = rightIP + right_otherIP;
            double upper_bound = convert_coe_.first * ub_left_part + convert_coe_.second * ub_right_part;
            return upper_bound;
        }

        double IPLowerBound(const double *user_vecs, const int &userID,
                            const double *item_vecs, const int &itemID) override {
            const int *user_int_vecs = user_int_ptr_.get() + userID * vec_dim_;
            const int *item_int_vecs = item_int_ptr_.get() + itemID * vec_dim_;

            int leftIP = InnerProduct(user_int_vecs, item_int_vecs, check_dim_);
            int left_otherIP = user_int_sum_ptr_[userID].first + item_int_sum_ptr_[itemID].first;
            int lb_left_part = leftIP - left_otherIP;

            const int *user_remain_int_vecs = user_int_vecs + check_dim_;
            const int *item_remain_int_vecs = item_int_vecs + check_dim_;

            int rightIP = InnerProduct(user_remain_int_vecs, item_remain_int_vecs, remain_dim_);
            int right_otherIP = user_int_sum_ptr_[userID].second + item_int_sum_ptr_[itemID].second;
            int lb_right_part = rightIP - right_otherIP;

            double lower_bound = convert_coe_.first * lb_left_part + convert_coe_.second * lb_right_part;
            return lower_bound;
        }

        std::pair<double, double>
        IPBound(const double *user_vecs, const int &userID, const double *item_vecs, const int &itemID) override {
            const int *user_int_vecs = user_int_ptr_.get() + userID * vec_dim_;
            const int *item_int_vecs = item_int_ptr_.get() + itemID * vec_dim_;

            int leftIP = InnerProduct(user_int_vecs, item_int_vecs, check_dim_);
            int left_otherIP = user_int_sum_ptr_[userID].first + item_int_sum_ptr_[itemID].first;
            int ub_left_part = leftIP + left_otherIP;
            int lb_left_part = leftIP - left_otherIP;

            const int *user_remain_int_vecs = user_int_vecs + check_dim_;
            const int *item_remain_int_vecs = item_int_vecs + check_dim_;

            int rightIP = InnerProduct(user_remain_int_vecs, item_remain_int_vecs, remain_dim_);
            int right_otherIP = user_int_sum_ptr_[userID].second + item_int_sum_ptr_[itemID].second;
            int ub_right_part = rightIP + right_otherIP;
            int lb_right_part = rightIP - right_otherIP;

            double upper_bound = convert_coe_.first * ub_left_part + convert_coe_.second * ub_right_part;
            double lower_bound = convert_coe_.first * lb_left_part + convert_coe_.second * lb_right_part;
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
#endif //REVERSE_KRANKS_FULLINT_HPP
