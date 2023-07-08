//
// Created by BianZheng on 2022/5/19.
//

#ifndef REVERSE_KRANKS_PARTDIMPARTINT_HPP
#define REVERSE_KRANKS_PARTDIMPARTINT_HPP

#include <cfloat>

#include "BaseIPBound.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "alg/SVD.hpp"

namespace ReverseMIPS {

    class PartDimPartInt : public BaseIPBound {
        int n_user_, n_data_item_, vec_dim_;
        //SVD
        SVD svd_ins_;
        int check_dim_, remain_dim_;

        //PartDimPartInt
        double scale_;
        std::unique_ptr<int[]> user_int_ptr_; // n_user_ * remain_dim_
        std::unique_ptr<int[]> user_int_sum_ptr_; // n_user_
        std::unique_ptr<int[]> item_int_ptr_; // n_data_item_ * remain_dim_
        std::unique_ptr<int[]> item_int_sum_ptr_; // n_data_item_
        double convert_coe_;

    public:

        inline PartDimPartInt() {
            n_user_ = -1;
            n_data_item_ = -1;
            vec_dim_ = -1;
            check_dim_ = -1;
            remain_dim_ = -1;

            user_int_ptr_ = nullptr;
            user_int_sum_ptr_ = nullptr;
            item_int_ptr_ = nullptr;
            item_int_sum_ptr_ = nullptr;
            convert_coe_ = -1;
        };

        inline PartDimPartInt(const int &n_user, const int &n_data_item, const int &vec_dim, const double &scale) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->vec_dim_ = vec_dim;
            this->scale_ = scale;
        }

        [[nodiscard]] double MatrixRemainMaxVal(const VectorMatrix &vm) const {
            double max_dim = vm.getVector(0)[check_dim_];
            const int &n_vector = vm.n_vector_;
            const int &vec_dim = vm.vec_dim_;

            for (int vecsID = 0; vecsID < n_vector; vecsID++) {
                const double *vecs = vm.getVector(vecsID);
                for (int dim = check_dim_; dim < vec_dim; dim++) {
                    max_dim = std::max(max_dim, vecs[dim]);
                }
            }
            max_dim = std::abs(max_dim);
            return max_dim;
        }

        void IntArrayVal(const VectorMatrix &vm, const double &max_dim,
                         int *vm_int_ptr, int *vm_int_sum_ptr) const {
            const int &n_vector = vm.n_vector_;
            const int &vec_dim = vm.vec_dim_;
            for (int vecsID = 0; vecsID < n_vector; vecsID++) {
                int *tmp_int_vecs = vm_int_ptr + vecsID * remain_dim_;
                double *tmp_double_vecs = vm.getVector(vecsID);

                vm_int_sum_ptr[vecsID] = 0;
                for (int dim = check_dim_; dim < vec_dim_; dim++) {
                    tmp_int_vecs[dim - check_dim_] = std::floor(tmp_double_vecs[dim] * scale_ / max_dim);
                    vm_int_sum_ptr[vecsID] += std::abs(tmp_int_vecs[dim - check_dim_]);
                }
            }
        }

        void Preprocess(VectorMatrix &user, VectorMatrix &data_item) override {
            const double SIGMA = 0.7;
            check_dim_ = svd_ins_.Preprocess(user, data_item, SIGMA);
            remain_dim_ = vec_dim_ - check_dim_;

            user_int_ptr_ = std::make_unique<int[]>(n_user_ * remain_dim_);
            user_int_sum_ptr_ = std::make_unique<int[]>(n_user_);
            item_int_ptr_ = std::make_unique<int[]>(n_data_item_ * remain_dim_);
            item_int_sum_ptr_ = std::make_unique<int[]>(n_data_item_);

            const double &user_max_dim = MatrixRemainMaxVal(user);
            const double &item_max_dim = MatrixRemainMaxVal(data_item);

            IntArrayVal(user, user_max_dim, user_int_ptr_.get(), user_int_sum_ptr_.get());

            for (int userID = 0; userID < n_user_; userID++) {
                user_int_sum_ptr_[userID] += remain_dim_;
            }

            IntArrayVal(data_item, item_max_dim, item_int_ptr_.get(), item_int_sum_ptr_.get());

            convert_coe_ = user_max_dim * item_max_dim / (scale_ * scale_);

        }

        void PreprocessQuery(const double *query_vecs, const int &vec_dim, double *query_write_vecs) override {
            svd_ins_.TransferQuery(query_vecs, vec_dim_, query_write_vecs);
        }

        double IPUpperBound(const double *user_vecs, const int &userID,
                            const double *item_vecs, const int &itemID) override {
            double leftIP = InnerProduct(user_vecs, item_vecs, check_dim_);

            const int *user_remain_int_vecs = user_int_ptr_.get() + userID * remain_dim_;
            const int *item_remain_int_vecs = item_int_ptr_.get() + itemID * remain_dim_;

            int rightIP = InnerProduct(user_remain_int_vecs, item_remain_int_vecs, remain_dim_);
            int right_otherIP = user_int_sum_ptr_[userID] + item_int_sum_ptr_[itemID];
            int ub_right_part = rightIP + right_otherIP;
            double upper_bound = leftIP + convert_coe_ * ub_right_part;
            return upper_bound;
        }

        double IPLowerBound(const double *user_vecs, const int &userID,
                            const double *item_vecs, const int &itemID) override {
            double leftIP = InnerProduct(user_vecs, item_vecs, check_dim_);

            const int *user_remain_int_vecs = user_int_ptr_.get() + userID * remain_dim_;
            const int *item_remain_int_vecs = item_int_ptr_.get() + itemID * remain_dim_;

            int rightIP = InnerProduct(user_remain_int_vecs, item_remain_int_vecs, remain_dim_);
            int right_otherIP = user_int_sum_ptr_[userID] + item_int_sum_ptr_[itemID];
            int lb_right_part = rightIP - right_otherIP;

            double lower_bound = leftIP + convert_coe_ * lb_right_part;
            return lower_bound;
        }

        std::pair<double, double>
        IPBound(const double *user_vecs, const int &userID, const double *item_vecs, const int &itemID) override {
            double leftIP = InnerProduct(user_vecs, item_vecs, check_dim_);

            const int *user_remain_int_vecs = user_int_ptr_.get() + userID * remain_dim_;
            const int *item_remain_int_vecs = item_int_ptr_.get() + itemID * remain_dim_;

            int rightIP = InnerProduct(user_remain_int_vecs, item_remain_int_vecs, remain_dim_);
            int right_otherIP = user_int_sum_ptr_[userID] + item_int_sum_ptr_[itemID];
            int lb_right_part = rightIP - right_otherIP;
            int ub_right_part = rightIP + right_otherIP;
            double lower_bound = leftIP + convert_coe_ * lb_right_part;
            double upper_bound = leftIP + convert_coe_ * ub_right_part;
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
#endif //REVERSE_KRANKS_PARTDIMPARTINT_HPP
