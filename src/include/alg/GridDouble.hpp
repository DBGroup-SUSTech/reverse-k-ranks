//
// Created by bianzheng on 2023/5/30.
//

#ifndef REVERSE_KRANKS_GRIDDOUBLE_HPP
#define REVERSE_KRANKS_GRIDDOUBLE_HPP

#include <cfloat>

namespace ReverseMIPS {

    //compute the min value and max value that the number > 0
    std::pair<double, double> MinMaxValue(const double *vm, const size_t &n_vector, const size_t &vec_dim) {
        double min_val = DBL_MAX;
        double max_val = 0;
        for (int vecsID = 0; vecsID < n_vector; vecsID++) {
            const double *vecs = vm + vecsID * vec_dim;
            for (int dim = 0; dim < vec_dim; dim++) {
                double vecs_val = std::abs(vecs[dim]);
                min_val = std::min(vecs_val, min_val);
                max_val = std::max(vecs_val, max_val);
            }
        }
        assert(min_val <= max_val);
        if (min_val - 0.01 >= 0) {
            min_val -= 0.01;
        }
        max_val += 0.01;
        return std::make_pair(min_val, max_val);
    }

    class GridDouble {
        int n_user_, n_data_item_, vec_dim_, n_partition_;
        // n_partition_ * n_partition_, each cell stores the IP lower_bound and upper_bound index
        // first dimension is the user, second dimension is the item
        std::unique_ptr<std::pair<double, double>[]> grid_;
        std::unique_ptr<int[]> user_codeword_; //n_user_ * vec_dim_
        std::unique_ptr<int[]> item_codeword_; //n_data_item * vec_dim_

        std::unique_ptr<char[]> user_negative_; // n_user * vec_dim_
        std::unique_ptr<char[]> item_negative_; // n_data_item * vec_dim_

    public:

        inline GridDouble() {
            this->n_user_ = -1;
            this->n_data_item_ = -1;
            this->vec_dim_ = -1;
            this->n_partition_ = -1;
        };

        inline GridDouble(const int &n_user, const int &n_data_item, const int &vec_dim, const int &n_partition) {
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
            user_codeword_ = std::make_unique<int[]>(n_user_ * vec_dim_);
            item_codeword_ = std::make_unique<int[]>(n_data_item_ * vec_dim_);
            user_negative_ = std::make_unique<char[]>(n_user_ * vec_dim_);
            item_negative_ = std::make_unique<char[]>(n_data_item_ * vec_dim_);
        }

        void Preprocess(const double *user, const double *data_item) {
            BuildIndex(data_item, user);
        }

        void BuildIndex(const double *data_item, const double *user) {
            //get the minimum / maximum value of user and item
            std::pair<double, double> item_minmax_pair = MinMaxValue(data_item, n_data_item_, vec_dim_);
            std::pair<double, double> user_minmax_pair = MinMaxValue(user, n_user_, vec_dim_);
            //get the distance of user / item
            const double user_dist = (user_minmax_pair.second - user_minmax_pair.first) / n_partition_;
            const double item_dist = (item_minmax_pair.second - item_minmax_pair.first) / n_partition_;
            //compute and assign the IP bound of GridDouble
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
                const double *user_vecs = user + userID * vec_dim_;
                for (int dim = 0; dim < vec_dim_; dim++) {
                    if (user_vecs[dim] <= 0) {
                        user_negative_[userID * vec_dim_ + dim] = true;
                    } else {
                        user_negative_[userID * vec_dim_ + dim] = false;
                    }
                    int bktID = std::floor((std::abs(user_vecs[dim]) - user_minmax_pair.first) / user_dist);
                    assert(0 <= bktID && bktID < n_partition_);
                    user_codeword_[userID * vec_dim_ + dim] = bktID;
                }
            }

            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                const double *item_vecs = data_item + itemID * vec_dim_;
                for (int dim = 0; dim < vec_dim_; dim++) {
                    if (item_vecs[dim] <= 0) {
                        item_negative_[itemID * vec_dim_ + dim] = true;
                    } else {
                        item_negative_[itemID * vec_dim_ + dim] = false;
                    }
                    // item_vecs[dim] > 0
                    int bktID = std::floor((std::abs(item_vecs[dim]) - item_minmax_pair.first) / item_dist);
                    assert(0 <= bktID && bktID < n_partition_);
                    item_codeword_[itemID * vec_dim_ + dim] = bktID;
                }
            }

        }

        double
        IPUpperBound(const int &userID, const int &itemID) const {
            const int *user_code_ptr = user_codeword_.get() + userID * vec_dim_;
            const int *item_code_ptr = item_codeword_.get() + itemID * vec_dim_;
            const char *user_negative_ptr = user_negative_.get() + userID * vec_dim_;
            const char *item_negative_ptr = item_negative_.get() + itemID * vec_dim_;
            double IP_ub = 0;
            for (int dim = 0; dim < vec_dim_; dim++) {
                const int user_code = user_code_ptr[dim];
                const int item_code = item_code_ptr[dim];
                const std::pair<double, double> bound_pair = grid_[user_code * n_partition_ + item_code];
                if (user_negative_ptr[dim] == true ^ item_negative_ptr[dim] == true) {
                    IP_ub += -bound_pair.first;
                } else {
                    IP_ub += bound_pair.second;
                }
            }
            return IP_ub;
        }

        double
        IPLowerBound(const int &userID, const int &itemID) const {
            const int *user_code_ptr = user_codeword_.get() + userID * vec_dim_;
            const int *item_code_ptr = item_codeword_.get() + itemID * vec_dim_;
            const char *user_negative_ptr = user_negative_.get() + userID * vec_dim_;
            const char *item_negative_ptr = item_negative_.get() + itemID * vec_dim_;
            double IP_lb = 0;
            for (int dim = 0; dim < vec_dim_; dim++) {
                const int user_code = user_code_ptr[dim];
                const int item_code = item_code_ptr[dim];
                const std::pair<double, double> bound_pair = grid_[user_code * n_partition_ + item_code];
                if (user_negative_ptr[dim] == true ^ item_negative_ptr[dim] == true) {
                    IP_lb += -bound_pair.second;
                } else {
                    IP_lb += bound_pair.first;
                }
            }
            return IP_lb;
        }

        double
        IPLowerBoundNoNegative(const int &userID, const int &itemID) const {
            const int *user_code_ptr = user_codeword_.get() + userID * vec_dim_;
            const int *item_code_ptr = item_codeword_.get() + itemID * vec_dim_;
            double IP_lb = 0;
            for (int dim = 0; dim < vec_dim_; dim++) {
                const int user_code = user_code_ptr[dim];
                const int item_code = item_code_ptr[dim];
                const std::pair<double, double> bound_pair = grid_[user_code * n_partition_ + item_code];
                IP_lb += bound_pair.first;
            }
            return IP_lb;
        }

        uint64_t IndexSize() {
            const uint64_t grid_size = sizeof(std::pair<double, double>) * n_partition_ * n_partition_;
            const uint64_t user_codeword_size = sizeof(int) * n_user_ * vec_dim_;
            const uint64_t item_codeword_size = sizeof(int) * n_data_item_ * vec_dim_;
            return grid_size + user_codeword_size + item_codeword_size;
        }

    };
}
#endif //REVERSE_KRANKS_GRIDDOUBLE_HPP
