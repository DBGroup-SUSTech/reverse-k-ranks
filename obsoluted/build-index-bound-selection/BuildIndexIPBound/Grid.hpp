//
// Created by BianZheng on 2022/5/19.
//

#ifndef REVERSE_K_RANKS_GRID_HPP
#define REVERSE_K_RANKS_GRID_HPP

#include <cfloat>

#include "BaseIPBound.hpp"

namespace ReverseMIPS {

    //compute the min value and max value that the number > 0
    std::pair<double, double> MinMaxValue(const VectorMatrix &vm) {
        const int &n_vector = vm.n_vector_;
        const int &vec_dim = vm.vec_dim_;
        double min_val = FLT_MAX;
        double max_val = 0;
        for (int vecsID = 0; vecsID < n_vector; vecsID++) {
            const double *vecs = vm.getVector(vecsID);
            for (int dim = 0; dim < vec_dim; dim++) {
                double vecs_val = vecs[dim];
                if (vecs_val >= 0) {
                    min_val = std::min(vecs_val, min_val);
                    max_val = std::max(vecs_val, max_val);
                }
            }
        }
        assert(min_val <= max_val);
        if (min_val - 0.01 >= 0) {
            min_val -= 0.01;
        }
        max_val += 0.01;
        return std::make_pair(min_val, max_val);
    }

    class Grid : public BaseIPBound {
        int n_user_, n_data_item_, vec_dim_, n_partition_;
        // n_partition_ * n_partition_, each cell stores the IP lower_bound and upper_bound index
        // first dimension is the user, second dimension is the item
        std::unique_ptr<std::pair<double, double>[]> grid_;
        std::unique_ptr<unsigned char[]> user_codeword_; //n_user_ * vec_dim_
        std::unique_ptr<unsigned char[]> item_codeword_; //n_data_item * vec_dim_

        unsigned char NEGATIVE_ = 255;

    public:

        inline Grid() {
            this->n_user_ = -1;
            this->n_data_item_ = -1;
            this->vec_dim_ = -1;
            this->n_partition_ = -1;
        };

        inline Grid(const int &n_user, const int &n_data_item, const int &vec_dim, const int &n_partition) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->vec_dim_ = vec_dim;
            this->n_partition_ = n_partition;
            const int lshift = sizeof(unsigned char) * 8;
            if (n_partition >= (1 << lshift)) {
                spdlog::error("n_partition too large, program exit");
                exit(-1);
            }

            grid_ = std::make_unique<std::pair<double, double>[]>(n_partition_ * n_partition_);
            user_codeword_ = std::make_unique<unsigned char[]>(n_user_ * vec_dim_);
            item_codeword_ = std::make_unique<unsigned char[]>(n_data_item_ * vec_dim_);
        }

        void Preprocess(VectorMatrix &user, VectorMatrix &data_item) {
            BuildIndex(data_item, user);
        }

        void PreprocessQuery(const double *query_vecs, const int &vec_dim, double *query_write_vecs) {
            memcpy(query_write_vecs, query_vecs, vec_dim * sizeof(double));
        }

        void BuildIndex(const VectorMatrix &data_item, const VectorMatrix &user) {
            //get the minimum / maximum value of user and item
            std::pair<double, double> item_minmax_pair = MinMaxValue(data_item);
            std::pair<double, double> user_minmax_pair = MinMaxValue(user);
            //get the distance of user / item
            const double user_dist = (user_minmax_pair.second - user_minmax_pair.first) / n_partition_;
            const double item_dist = (item_minmax_pair.second - item_minmax_pair.first) / n_partition_;
            //compute and assign the IP bound of Grid
            for (int user_partID = 0; user_partID < n_partition_; user_partID++) {
                const double user_lb = user_minmax_pair.first + user_dist * user_partID;
                const double user_ub = user_minmax_pair.first + user_dist * (user_partID + 1);
                for (int item_partID = 0; item_partID < n_partition_; item_partID++) {
                    const double item_lb = item_minmax_pair.first + item_dist * item_partID;
                    const double item_ub = item_minmax_pair.first + item_dist * (item_partID + 1);
                    const double ip_lb = user_lb * item_lb;
                    const double ip_ub = user_ub * item_ub;
                    grid_[user_partID * n_partition_ + item_partID] = std::make_pair(ip_lb, ip_ub);
                }
            }
            //encode user and item
            for (int userID = 0; userID < n_user_; userID++) {
                const double *user_vecs = user.getVector(userID);
                for (int dim = 0; dim < vec_dim_; dim++) {
                    if (user_vecs[dim] <= 0) {
                        user_codeword_[userID * vec_dim_ + dim] = NEGATIVE_;
                    } else { // user_vecs[dim] > 0
                        unsigned char bktID = std::floor((user_vecs[dim] - user_minmax_pair.first) / user_dist);
                        assert(0 <= bktID && bktID < n_partition_);
                        user_codeword_[userID * vec_dim_ + dim] = bktID;
                    }
                }
            }

            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                const double *item_vecs = data_item.getVector(itemID);
                for (int dim = 0; dim < vec_dim_; dim++) {
                    if (item_vecs[dim] <= 0) {
                        item_codeword_[itemID * vec_dim_ + dim] = NEGATIVE_;
                    } else { // item_vecs[dim] > 0
                        unsigned char bktID = std::floor((item_vecs[dim] - item_minmax_pair.first) / item_dist);
                        assert(0 <= bktID && bktID < n_partition_);
                        item_codeword_[itemID * vec_dim_ + dim] = bktID;
                    }
                }
            }

        }

        double
        IPUpperBound(const double *user_vecs, const int &userID, const double *item_vecs, const int &itemID) override {
            const unsigned char *user_code_ptr = user_codeword_.get() + userID * vec_dim_;
            const unsigned char *item_code_ptr = item_codeword_.get() + itemID * vec_dim_;
            double IP_ub = 0;
            for (int dim = 0; dim < vec_dim_; dim++) {
                if (user_code_ptr[dim] == NEGATIVE_ || item_code_ptr[dim] == NEGATIVE_) {
                    double IP = user_vecs[dim] * item_vecs[dim];
                    IP_ub += IP;
                } else {
                    const unsigned char user_code = user_code_ptr[dim];
                    const unsigned char item_code = item_code_ptr[dim];
                    const std::pair<double, double> bound_pair = grid_[user_code * n_partition_ + item_code];
                    IP_ub += bound_pair.second;
                }
            }
            return IP_ub;
        }

        double
        IPLowerBound(const double *user_vecs, const int &userID, const double *item_vecs, const int &itemID) override {
            const unsigned char *user_code_ptr = user_codeword_.get() + userID * vec_dim_;
            const unsigned char *item_code_ptr = item_codeword_.get() + itemID * vec_dim_;
            double IP_lb = 0;
            for (int dim = 0; dim < vec_dim_; dim++) {
                if (user_code_ptr[dim] == NEGATIVE_ || item_code_ptr[dim] == NEGATIVE_) {
                    double IP = user_vecs[dim] * item_vecs[dim];
                    IP_lb += IP;
                } else {
                    const unsigned char user_code = user_code_ptr[dim];
                    const unsigned char item_code = item_code_ptr[dim];
                    const std::pair<double, double> bound_pair = grid_[user_code * n_partition_ + item_code];
                    IP_lb += bound_pair.first;
                }
            }
            return IP_lb;
        }

        std::pair<double, double>
        IPBound(const double *user_vecs, const int &userID, const double *item_vecs, const int &itemID) override {
            const unsigned char *user_code_ptr = user_codeword_.get() + userID * vec_dim_;
            const unsigned char *item_code_ptr = item_codeword_.get() + itemID * vec_dim_;
            double IP_ub = 0;
            double IP_lb = 0;
            for (int dim = 0; dim < vec_dim_; dim++) {
                if (user_code_ptr[dim] == NEGATIVE_ || item_code_ptr[dim] == NEGATIVE_) {
                    double IP = user_vecs[dim] * item_vecs[dim];
                    IP_lb += IP;
                    IP_ub += IP;
                } else {
                    const unsigned char user_code = user_code_ptr[dim];
                    const unsigned char item_code = item_code_ptr[dim];
                    const std::pair<double, double> bound_pair = grid_[user_code * n_partition_ + item_code];
                    IP_lb += bound_pair.first;
                    IP_ub += bound_pair.second;
                }
            }
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

#endif //REVERSE_K_RANKS_GRID_HPP
