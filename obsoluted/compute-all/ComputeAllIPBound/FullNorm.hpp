//
// Created by BianZheng on 2022/5/19.
//

#ifndef REVERSE_KRANKS_FULLNORM_HPP
#define REVERSE_KRANKS_FULLNORM_HPP

#include <cfloat>

#include "BaseIPBound.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "alg/SVD.hpp"

namespace ReverseMIPS {

    class FullNorm : public BaseIPBound {
        int n_user_, n_data_item_, vec_dim_;
        //SVD
        SVD svd_ins_;
        int check_dim_;
        //IPBound prune
        std::unique_ptr<std::pair<double, double>[]> user_norm_l_;
        std::unique_ptr<std::pair<double, double>[]> item_norm_l_;

    public:

        inline FullNorm() {
            n_user_ = -1;
            n_data_item_ = -1;
            vec_dim_ = -1;
            check_dim_ = -1;
            user_norm_l_ = nullptr;
            item_norm_l_ = nullptr;
        };

        inline FullNorm(const int &n_user, const int &n_data_item, const int &vec_dim) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->vec_dim_ = vec_dim;

            user_norm_l_ = std::make_unique<std::pair<double, double>[]>(n_user_);
            item_norm_l_ = std::make_unique<std::pair<double, double>[]>(n_data_item_);
        }

        void Preprocess(VectorMatrix &user, VectorMatrix &data_item) override {
            const double SIGMA = 0.7;
            check_dim_ = svd_ins_.Preprocess(user, data_item, SIGMA);

            int remain_dim = vec_dim_ - check_dim_;

            for (int userID = 0; userID < n_user_; userID++) {
                double left_norm = InnerProduct(user.getVector(userID), user.getVector(userID), check_dim_);
                left_norm = std::sqrt(left_norm);

                double right_norm = InnerProduct(user.getVector(userID, check_dim_), user.getVector(userID, check_dim_),
                                                 remain_dim);
                right_norm = std::sqrt(right_norm);
                user_norm_l_[userID] = std::make_pair(left_norm, right_norm);
            }

            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                double left_norm = InnerProduct(data_item.getVector(itemID), data_item.getVector(itemID), check_dim_);
                left_norm = std::sqrt(left_norm);

                double right_norm = InnerProduct(data_item.getVector(itemID, check_dim_),
                                                 data_item.getVector(itemID, check_dim_),
                                                 remain_dim);
                right_norm = std::sqrt(right_norm);
                item_norm_l_[itemID] = std::make_pair(left_norm, right_norm);
            }

        }

        void PreprocessQuery(const double *query_vecs, const int &vec_dim, double *query_write_vecs) override {
            svd_ins_.TransferQuery(query_vecs, vec_dim_, query_write_vecs);
        }

        double IPUpperBound(const double *user_vecs, const int &userID,
                            const double *item_vecs, const int &itemID) override {
            const std::pair<double, double> user_norm_pair = user_norm_l_[userID];
            const std::pair<double, double> item_norm_pair = item_norm_l_[itemID];
            const double IP_ub =
                    user_norm_pair.first * item_norm_pair.first + user_norm_pair.second * item_norm_pair.second;
            return IP_ub;
        }

        double IPLowerBound(const double *user_vecs, const int &userID,
                            const double *item_vecs, const int &itemID) override {
            const std::pair<double, double> user_norm_pair = user_norm_l_[userID];
            const std::pair<double, double> item_norm_pair = item_norm_l_[itemID];
            const double IP_lb =
                    -user_norm_pair.first * item_norm_pair.first - user_norm_pair.second * item_norm_pair.second;
            return IP_lb;
        }

        std::pair<double, double>
        IPBound(const double *user_vecs, const int &userID, const double *item_vecs, const int &itemID) override {
            const std::pair<double, double> user_norm_pair = user_norm_l_[userID];
            const std::pair<double, double> item_norm_pair = item_norm_l_[itemID];
            const double IP_lb =
                    -user_norm_pair.first * item_norm_pair.first - user_norm_pair.second * item_norm_pair.second;
            const double IP_ub =
                    user_norm_pair.first * item_norm_pair.first + user_norm_pair.second * item_norm_pair.second;
            return std::make_pair(IP_lb, IP_ub);
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
#endif //REVERSE_KRANKS_FULLNORM_HPP
