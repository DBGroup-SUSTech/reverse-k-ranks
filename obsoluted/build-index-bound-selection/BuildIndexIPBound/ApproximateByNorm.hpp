//
// Created by BianZheng on 2022/10/6.
//

#ifndef REVERSE_KRANKS_APPROXIMATEBYNORM_HPP
#define REVERSE_KRANKS_APPROXIMATEBYNORM_HPP

#include <cfloat>

#include "BaseIPBound.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "alg/SpaceEuclidean.hpp"

namespace ReverseMIPS {

    class ApproximateByNorm : public BaseIPBound {
        int n_user_, n_data_item_, vec_dim_;
        int n_sample_item_;
        std::unique_ptr<double[]> sample_ip_l_; //n_sample_item_ * n_user_, record the sampled ip
        std::unique_ptr<double[]> sample_item_vecs_; //n_sample_item * vec_dim_, record the sampled item vectors
        std::unique_ptr<double[]> user_abs_sum_l_; //n_user_, record the summation absolute value of user in each dimension

    public:

        inline ApproximateByNorm() {
            this->n_user_ = -1;
            this->n_data_item_ = -1;
            this->vec_dim_ = -1;
        }

        inline ApproximateByNorm(const int &n_user, const int &n_data_item, const int &vec_dim,
                                 const int &n_sample_item) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->vec_dim_ = vec_dim;
            this->n_sample_item_ = n_sample_item;
            sample_ip_l_ = std::make_unique<double[]>(n_sample_item_ * n_user_);
            sample_item_vecs_ = std::make_unique<double[]>(n_sample_item_ * vec_dim_);
            user_abs_sum_l_ = std::make_unique<double[]>(n_user_);
        }

        void AssignUserAbsSum(const VectorMatrix &user) {
            for (int userID = 0; userID < n_user_; userID++) {
                double abs_sum = 0;
                const double *user_vecs = user.getVector(userID);
                for (int dim = 0; dim < vec_dim_; dim++) {
                    abs_sum += std::abs(user_vecs[dim]);
                }
                user_abs_sum_l_[userID] = abs_sum;
            }

        }

        void SelectItemCandidate(const VectorMatrix &data_item, std::vector<int> &itemID_l) {
            assert(itemID_l.size() == n_sample_item_);
            //random select the item
            itemID_l[0] = 0;
            std::vector<bool> is_cand_l(n_data_item_, false);
            is_cand_l[0] = true;
            //select the nect item
            for (int item_candID = 1; item_candID < n_sample_item_; item_candID++) {

                int max_dist_itemID = -1;
                double max_dist = 0;
#pragma omp parallel for default(none) shared(is_cand_l, max_dist, max_dist_itemID, data_item, item_candID, itemID_l)
                for (int itemID = 0; itemID < n_data_item_; itemID++) {
                    if (is_cand_l[itemID]) {
                        continue;
                    }
                    double cand_item_min_dist = DBL_MAX;
                    const double *tmp_item_vecs = data_item.getVector(itemID);
                    for (int prev_item_candID = 0; prev_item_candID < item_candID; prev_item_candID++) {
                        assert(itemID != itemID_l[prev_item_candID]);
                        const double *cand_item_vecs = data_item.getVector(itemID_l[prev_item_candID]);
                        const double cand_item_dist = EuclideanDistance(tmp_item_vecs, cand_item_vecs, vec_dim_);
                        cand_item_min_dist = cand_item_min_dist < cand_item_dist ? cand_item_min_dist : cand_item_dist;
                    }
#pragma omp critical
                    {
                        if (max_dist < cand_item_min_dist) {
                            max_dist = cand_item_min_dist;
                            max_dist_itemID = itemID;
                        }
                    };

                }
                assert(max_dist_itemID != -1);
                itemID_l[item_candID] = max_dist_itemID;
                is_cand_l[max_dist_itemID] = true;
            }

        }

        void
        AssignItemCandidate(const VectorMatrix &user, const VectorMatrix &data_item, const std::vector<int> &itemID_l) {
            assert(itemID_l.size() == n_sample_item_);
#pragma omp parallel for default(none) shared(itemID_l, data_item, user)
            for (int sampleID = 0; sampleID < n_sample_item_; sampleID++) {
                const int itemID = itemID_l[sampleID];
                const double *item_vecs = data_item.getVector(itemID);
                memcpy(sample_item_vecs_.get() + sampleID * vec_dim_, item_vecs, vec_dim_ * sizeof(double));

                std::vector<double> itemIP_l(n_user_);
                for (int userID = 0; userID < n_user_; userID++) {
                    const double *user_vecs = user.getVector(userID);
                    const double itemIP = InnerProduct(item_vecs, user_vecs, vec_dim_);
                    itemIP_l[userID] = itemIP;
                }
                memcpy(sample_ip_l_.get() + sampleID * n_user_, itemIP_l.data(), n_user_ * sizeof(double));
            }
        }

        void Preprocess(VectorMatrix &user, VectorMatrix &data_item) override {

            AssignUserAbsSum(user);
            //select item with a large distance
            //first random select an item, then select the item with the maximum of minimum of all the other candidates, until all the elements has selected
            std::vector<int> itemID_l(n_sample_item_);
            SelectItemCandidate(data_item, itemID_l);
            //record their ip and vecs information
            AssignItemCandidate(user, data_item, itemID_l);

        }

        void PreprocessQuery(const double *query_vecs, const int &vec_dim, double *query_write_vecs) override {
            memcpy(query_write_vecs, query_vecs, vec_dim * sizeof(double));
        }

        double IPUpperBound(const double *user_vecs, const int &userID,
                            const double *item_vecs, const int &itemID) override {
            const double itemIP = InnerProduct(user_vecs, item_vecs, vec_dim_);
            return itemIP;
        }

        double IPLowerBound(const double *user_vecs, const int &userID,
                            const double *item_vecs, const int &itemID) override {
            const double itemIP = InnerProduct(user_vecs, item_vecs, vec_dim_);
            return itemIP;
        }

        std::pair<double, double>
        IPBound(const double *user_vecs, const int &userID, const double *item_vecs, const int &itemID) override {
            const double itemIP = InnerProduct(user_vecs, item_vecs, vec_dim_);
            return std::make_pair(itemIP, itemIP);
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
#endif //REVERSE_KRANKS_APPROXIMATEBYNORM_HPP
