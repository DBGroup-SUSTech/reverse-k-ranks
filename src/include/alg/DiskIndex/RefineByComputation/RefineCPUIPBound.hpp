//
// Created by bianzheng on 2023/3/28.
//

#ifndef REVERSE_KRANKS_REFINECPUIPBOUND_HPP
#define REVERSE_KRANKS_REFINECPUIPBOUND_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "alg/DiskIndex/RefineByComputation/RefineCPUPartDimPartNorm.hpp"

#include <algorithm>
#include <vector>
#include <iostream>

namespace ReverseMIPS {

    struct RefineCPUIPBoundParallel {
        std::vector<std::pair<float, float>> itemIP_bound_l_;
        std::vector<char> is_query_rank_larger_l_; // larger means rank(u,q) < rank(u, p)
        std::vector<char> is_query_rank_lower_l_; // lower means rank(u, q) > rank(u, p)
        std::vector<float> itemIP_l_;
    };

    class RefineCPUIPBound {

        uint64_t n_user_, n_data_item_, vec_dim_;
        int check_dim_;
        int n_thread_;
        const float *user_ptr_;

        RefineCPUPartDimPartNorm ip_bound_ins_;

        std::vector<float> item_norm_l_;
        VectorMatrix proc_data_item_;

        mutable std::vector<RefineCPUIPBoundParallel> parallel_l_;

    public:
        RefineCPUIPBound() = default;

        inline RefineCPUIPBound(const VectorMatrix &user, const VectorMatrix &data_item, const int &check_dim,
                                const int &n_thread) {
            this->user_ptr_ = user.getRawData();
            this->n_user_ = user.n_vector_;
            this->n_data_item_ = data_item.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->check_dim_ = check_dim;
            this->n_thread_ = n_thread;
            assert(user.vec_dim_ == data_item.vec_dim_);

            std::unique_ptr<float[]> data_item_proc_ptr = std::make_unique<float[]>(n_data_item_ * vec_dim_);
            item_norm_l_.resize(n_data_item_);

            //compute norm
            const float *data_item_ptr = data_item.getRawData();
            std::vector<float> item_norm_l(n_data_item_);
            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                const float *item_vecs = data_item_ptr + itemID * vec_dim_;
                const float norm = std::sqrt(InnerProduct(item_vecs, item_vecs, (int) vec_dim_));
                item_norm_l[itemID] = norm;
            }

            //arg sort by norm size, descending
            std::vector<int> item_idx_l(n_data_item_);
            std::iota(item_idx_l.begin(), item_idx_l.end(), 0);
            std::sort(item_idx_l.begin(), item_idx_l.end(),
                      [&item_norm_l](const int i1, const int i2) { return item_norm_l[i1] > item_norm_l[i2]; });

            //assign the input
            for (int origin_itemID = 0; origin_itemID < n_data_item_; origin_itemID++) {
                const int now_itemID = item_idx_l[origin_itemID];
                assert(0 <= now_itemID && now_itemID < n_data_item_);
                item_norm_l_[origin_itemID] = item_norm_l[now_itemID];
                memcpy(data_item_proc_ptr.get() + origin_itemID * vec_dim_, data_item_ptr + now_itemID * vec_dim_,
                       vec_dim_ * sizeof(float));
            }
            proc_data_item_.init(data_item_proc_ptr, (int) n_data_item_, (int) vec_dim_);

            ip_bound_ins_ = std::move(
                    RefineCPUPartDimPartNorm((int) n_user_, (int) n_data_item_, (int) vec_dim_, check_dim));
            ip_bound_ins_.Preprocess(user, proc_data_item_);

            parallel_l_.resize(n_thread_);
            for (int threadID = 0; threadID < n_thread_; threadID++) {
                parallel_l_[threadID].itemIP_bound_l_.resize(n_data_item_);
                parallel_l_[threadID].is_query_rank_larger_l_.resize(n_data_item_);
                parallel_l_[threadID].is_query_rank_lower_l_.resize(n_data_item_);
                parallel_l_[threadID].itemIP_l_.resize(n_data_item_);
            }

        }

        [[nodiscard]] int
        RefineRank(const float &queryIP, const int &userID, int64_t &n_compute_item, int64_t &ip_cost,
                   double &tmp_refine_time, const int &threadID) const {
            ip_cost = 0;
            parallel_l_[threadID].is_query_rank_lower_l_.assign(n_data_item_, false);
            parallel_l_[threadID].is_query_rank_larger_l_.assign(n_data_item_, false);
            const float slight_low_queryIP = queryIP - 8e-3f;
            const float slight_high_queryIP = queryIP + 8e-3f;
            const float *rank_ptr = std::lower_bound(item_norm_l_.data(), item_norm_l_.data() + n_data_item_,
                                                     slight_low_queryIP,
                                                     [](const float &arrIP, const float queryIP) {
                                                         return arrIP > queryIP;
                                                     });
            const int n_proc_item = rank_ptr - item_norm_l_.data();
            assert(0 <= n_proc_item && n_proc_item <= n_data_item_);
            n_compute_item = n_proc_item;

            TimeRecord record;
            const float *user_vecs = user_ptr_ + userID * vec_dim_;
            ip_bound_ins_.IPBoundNoParallel(proc_data_item_, n_proc_item, user_vecs, userID,
                                            parallel_l_[threadID].itemIP_bound_l_,
                                            parallel_l_[threadID].itemIP_l_); //this time, only calculate part of the IP
            ip_cost += (int) (n_proc_item * 1.0 * (check_dim_ + 1) / (int) vec_dim_);

            int base_rank = 0;
            for (int itemID = 0; itemID < n_proc_item; itemID++) {
                const float itemIP_lb = parallel_l_[threadID].itemIP_bound_l_[itemID].first;
                const float itemIP_ub = parallel_l_[threadID].itemIP_bound_l_[itemID].second;
                assert(itemIP_ub >= itemIP_lb);
                if (slight_low_queryIP > itemIP_ub) {
                    parallel_l_[threadID].is_query_rank_larger_l_[itemID] = true;
                } else if (slight_high_queryIP < itemIP_lb) {
                    parallel_l_[threadID].is_query_rank_lower_l_[itemID] = true;
                    base_rank++;
                }

            }

            ip_bound_ins_.ComputeRemainDimNoParallel(proc_data_item_, n_proc_item, user_vecs,
                                                     parallel_l_[threadID].is_query_rank_larger_l_,
                                                     parallel_l_[threadID].is_query_rank_lower_l_,
                                                     parallel_l_[threadID].itemIP_l_);

            for (int itemID = 0; itemID < n_proc_item; itemID++) {
                if (parallel_l_[threadID].is_query_rank_larger_l_[itemID] ||
                    parallel_l_[threadID].is_query_rank_lower_l_[itemID]) {
                    continue;
                }
                const float ip = parallel_l_[threadID].itemIP_l_[itemID];
                ip_cost++;
                if (ip >= queryIP) {
                    base_rank++;
                }
            }
            tmp_refine_time = record.get_elapsed_time_second();

            return base_rank;
        }

        void FinishCompute() {
        }
    };

}
#endif //REVERSE_KRANKS_REFINECPUIPBOUND_HPP
