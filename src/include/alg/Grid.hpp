//
// Created by BianZheng on 2022/5/19.
//

#ifndef REVERSE_K_RANKS_GRID_HPP
#define REVERSE_K_RANKS_GRID_HPP

#include <cfloat>

namespace ReverseMIPS {

    //compute the min value and max value that the number > 0
    std::pair<float, float> MinMaxValue(const VectorMatrix &vm) {
        const int &n_vector = vm.n_vector_;
        const int &vec_dim = vm.vec_dim_;
        float min_val = FLT_MAX;
        float max_val = 0;
        for (int vecsID = 0; vecsID < n_vector; vecsID++) {
            const float *vecs = vm.getVector(vecsID);
            for (int dim = 0; dim < vec_dim; dim++) {
                float vecs_val = std::abs(vecs[dim]);
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

    class Grid {
        int n_user_, n_data_item_, vec_dim_, n_partition_;
        // n_partition_ * n_partition_, each cell stores the IP lower_bound and upper_bound index
        // first dimension is the user, second dimension is the item
//        std::unique_ptr<float[]> grid_;
        std::vector<std::vector<float>> grid_;
//        std::unique_ptr<int[]> user_codeword_; //n_user_ * vec_dim_
        std::vector<std::vector<int>> user_codeword_;
//        std::unique_ptr<int[]> item_codeword_; //n_data_item * vec_dim_
        std::vector<std::vector<int>> item_codeword_;

//        std::unique_ptr<char[]> user_negative_; // n_user * vec_dim_
        std::vector<std::vector<char>> user_negative_;
//        std::unique_ptr<char[]> item_negative_; // n_data_item * vec_dim_
        std::vector<std::vector<char>> item_negative_;

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

            grid_.resize(n_partition_);
            for (int i = 0; i < n_partition_; i++) {
                grid_[i].resize(n_partition_);
            }

//            grid_ = std::make_unique<float[]>(n_partition_ * n_partition_);

            user_codeword_.resize(n_user_);
            for (int i = 0; i < n_user_; i++) {
                user_codeword_[i].resize(vec_dim_);
            }
//            user_codeword_ = std::make_unique<int[]>(n_user_ * vec_dim_);

            item_codeword_.resize(n_data_item_);
            for (int i = 0; i < n_data_item_; i++) {
                item_codeword_[i].resize(vec_dim_);
            }
//            item_codeword_ = std::make_unique<int[]>(n_data_item_ * vec_dim_);

            user_negative_.resize(n_user_);
            for (int i = 0; i < n_user_; i++) {
                user_negative_[i].resize(vec_dim_);
            }
//            user_negative_ = std::make_unique<char[]>(n_user_ * vec_dim_);

            item_negative_.resize(n_data_item_);
            for (int i = 0; i < n_data_item_; i++) {
                item_negative_[i].resize(vec_dim_);
            }
//            item_negative_ = std::make_unique<char[]>(n_data_item_ * vec_dim_);
        }

        void Preprocess(const VectorMatrix &user, const VectorMatrix &data_item) {
            BuildIndex(data_item, user);
        }

        void BuildIndex(const VectorMatrix &data_item, const VectorMatrix &user) {
            //get the minimum / maximum value of user and item
            std::pair<float, float> item_minmax_pair = MinMaxValue(data_item);
            std::pair<float, float> user_minmax_pair = MinMaxValue(user);
            //get the distance of user / item
            const float user_dist = (user_minmax_pair.second - user_minmax_pair.first) / (n_partition_ - 1);
            const float item_dist = (item_minmax_pair.second - item_minmax_pair.first) / (n_partition_ - 1);
            //compute and assign the IP bound of Grid
            for (int user_partID = 0; user_partID < n_partition_; user_partID++) {
                const float user_lb = user_minmax_pair.first + user_dist * user_partID;
                for (int item_partID = 0; item_partID < n_partition_; item_partID++) {
                    const float item_lb = item_minmax_pair.first + item_dist * item_partID;
                    const float ip_lb = user_lb * item_lb;
//                    grid_[user_partID * n_partition_ + item_partID] = ip_lb;
                    grid_[user_partID][item_partID] = ip_lb;
                }
            }

            //encode user and item
            for (int userID = 0; userID < n_user_; userID++) {
                const float *user_vecs = user.getVector(userID);
                for (int dim = 0; dim < vec_dim_; dim++) {
                    if (user_vecs[dim] <= 0) {
//                        user_negative_[userID * vec_dim_ + dim] = true;
                        user_negative_[userID][dim] = true;
                    } else {
//                        user_negative_[userID * vec_dim_ + dim] = false;
                        user_negative_[userID][dim] = false;
                    }
                    int bktID = std::floor((std::abs(user_vecs[dim]) - user_minmax_pair.first) / user_dist);
                    assert(0 <= bktID && bktID < n_partition_ - 1);
//                    user_codeword_[userID * vec_dim_ + dim] = bktID;
                    user_codeword_[userID][dim] = bktID;
                }
            }

            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                const float *item_vecs = data_item.getVector(itemID);
                for (int dim = 0; dim < vec_dim_; dim++) {
                    if (item_vecs[dim] <= 0) {
//                        item_negative_[itemID * vec_dim_ + dim] = true;
                        item_negative_[itemID][dim] = true;
                    } else {
//                        item_negative_[itemID * vec_dim_ + dim] = false;
                        item_negative_[itemID][dim] = false;
                    }
                    // item_vecs[dim] > 0
                    int bktID = std::floor((std::abs(item_vecs[dim]) - item_minmax_pair.first) / item_dist);
                    assert(0 <= bktID && bktID < n_partition_ - 1);
//                    item_codeword_[itemID * vec_dim_ + dim] = bktID;
                    item_codeword_[itemID][dim] = bktID;
                }
            }

        }

        float
        IPUpperBound(const int &userID, const int &itemID) const {
//            const int *user_code_ptr = user_codeword_.get() + userID * vec_dim_;
//            const int *item_code_ptr = item_codeword_.get() + itemID * vec_dim_;
//            const char *user_negative_ptr = user_negative_.get() + userID * vec_dim_;
//            const char *item_negative_ptr = item_negative_.get() + itemID * vec_dim_;
            float IP_ub = 0;
            for (int dim = 0; dim < vec_dim_; dim++) {
//                const int user_code = user_code_ptr[dim];
//                const int item_code = item_code_ptr[dim];
//                const float tmp_ip_lb = grid_[user_code * n_partition_ + item_code];
//                const float tmp_ip_ub = grid_[(user_code + 1) * n_partition_ + (item_code + 1)];
//                if (user_negative_ptr[dim] == true ^ item_negative_ptr[dim] == true) {
//                    IP_ub += -tmp_ip_lb;
//                } else {
//                    IP_ub += tmp_ip_ub;
//                }
                const int user_code = user_codeword_[userID][dim];
                const int item_code = item_codeword_[itemID][dim];
                const float tmp_ip_lb = grid_[user_code][item_code];
                const float tmp_ip_ub = grid_[user_code + 1][item_code + 1];
                if (user_negative_[userID][dim] ^ item_negative_[itemID][dim]) {
                    IP_ub += -tmp_ip_lb;
                } else {
                    IP_ub += tmp_ip_ub;
                }
            }
            return IP_ub;
        }

        float
        IPLowerBound(const int &userID, const int &itemID) const {
//            const int *user_code_ptr = user_codeword_.get() + userID * vec_dim_;
//            const int *item_code_ptr = item_codeword_.get() + itemID * vec_dim_;
//            const char *user_negative_ptr = user_negative_.get() + userID * vec_dim_;
//            const char *item_negative_ptr = item_negative_.get() + itemID * vec_dim_;
//            float IP_lb = 0;
//            for (int dim = 0; dim < vec_dim_; dim++) {
//                const int user_code = user_code_ptr[dim];
//                const int item_code = item_code_ptr[dim];
//                const float tmp_ip_lb = grid_[user_code * n_partition_ + item_code];
//                const float tmp_ip_ub = grid_[(user_code + 1) * n_partition_ + (item_code + 1)];
//                if (user_negative_ptr[dim] == true ^ item_negative_ptr[dim] == true) {
//                    IP_lb += -tmp_ip_ub;
//                } else {
//                    IP_lb += tmp_ip_lb;
//                }
//            }
            float IP_lb = 0;
            for (int dim = 0; dim < vec_dim_; dim++) {
                const int user_code = user_codeword_[userID][dim];
                const int item_code = item_codeword_[itemID][dim];
                const float tmp_ip_lb = grid_[user_code][item_code];
                const float tmp_ip_ub = grid_[user_code + 1][item_code + 1];
                if (user_negative_[userID][dim] ^ item_negative_[itemID][dim]) {
                    IP_lb += -tmp_ip_ub;
                } else {
                    IP_lb += tmp_ip_lb;
                }
            }
            return IP_lb;
        }

        float
        IPLowerBoundNoNegative(const int &userID, const int &itemID) const {
//            const int *user_code_ptr = user_codeword_.get() + userID * vec_dim_;
//            const int *item_code_ptr = item_codeword_.get() + itemID * vec_dim_;
//            float IP_lb = 0;
//            for (int dim = 0; dim < vec_dim_; dim++) {
//                const int user_code = user_code_ptr[dim];
//                const int item_code = item_code_ptr[dim];
//                const float tmp_ip_lb = grid_[user_code * n_partition_ + item_code];
//                IP_lb += tmp_ip_lb;
//            }
            float IP_lb = 0;
            for (int dim = 0; dim < vec_dim_; dim++) {
                const int user_code = user_codeword_[userID][dim];
                const int item_code = item_codeword_[itemID][dim];
                const float tmp_ip_lb = grid_[user_code][item_code];
                IP_lb += tmp_ip_lb;
            }
            return IP_lb;
        }

        uint64_t IndexSize() {
            const uint64_t grid_size = sizeof(float) * n_partition_ * n_partition_;
            const uint64_t user_codeword_size = sizeof(int) * n_user_ * vec_dim_;
            const uint64_t item_codeword_size = sizeof(int) * n_data_item_ * vec_dim_;
            return grid_size + user_codeword_size + item_codeword_size;
        }

    };
}

#endif //REVERSE_K_RANKS_GRID_HPP
