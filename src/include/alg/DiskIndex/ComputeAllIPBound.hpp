//
// Created by bianzheng on 2023/3/27.
//

#ifndef REVERSE_KRANKS_COMPUTEALLIPBOUND_HPP
#define REVERSE_KRANKS_COMPUTEALLIPBOUND_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "struct/UserRankElement.hpp"
#include "alg/DiskIndex/RefineByComputation/RefineCPUIPBound.hpp"
#include "alg/DiskIndex/BaseCompute.hpp"

#include <memory>
#include <omp.h>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class ComputeAllIPBound : public BaseCompute {
        int n_data_item_, n_user_, vec_dim_;
        int n_thread_;

        TimeRecord record_;
        RefineCPUIPBound refine_cpu_;
        std::unique_ptr<int[]> user_candidate_l_;

        std::unique_ptr<int[]> ip_cost_parallel_l_;
        std::unique_ptr<int[]> n_compute_item_parallel_l_;
        std::unique_ptr<int[]> is_refine_parallel_l_;
        std::unique_ptr<double[]> single_refine_user_time_parallel_l_;

    public:

//        std::vector<UserRankElement> user_topk_cache_l_;

        inline ComputeAllIPBound() = default;

        inline ComputeAllIPBound(const VectorMatrix &user, const VectorMatrix &data_item, const int &check_dim,
                                 const int &n_thread = omp_get_num_procs()) {
            this->n_user_ = user.n_vector_;
            this->n_data_item_ = data_item.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->user_topk_cache_l_.resize(n_user_);
            this->user_candidate_l_ = std::make_unique<int[]>(n_user_);
            this->n_thread_ = n_thread;

            this->ip_cost_parallel_l_ = std::make_unique<int[]>(n_user_);
            this->n_compute_item_parallel_l_ = std::make_unique<int[]>(n_user_);
            this->is_refine_parallel_l_ = std::make_unique<int[]>(n_user_);
            this->single_refine_user_time_parallel_l_ = std::make_unique<double[]>(n_user_);

            refine_cpu_ = RefineCPUIPBound(user, data_item, check_dim, n_thread);

//            refine_cpu_ = RefineCPUIPBound(user, data_item, check_dim);
        }

        virtual void GetRank(const std::vector<float> &queryIP_l,
                             const std::vector<int> &rank_lb_l, const std::vector<int> &rank_ub_l,
                             const std::vector<char> &prune_l, const std::vector<char> &result_l,
                             const int &n_remain_result, size_t &refine_ip_cost, int &n_refine_user,
                             int64_t &n_compute_item,
                             double &refine_user_time, double &single_refine_user_time) override {

            //read disk and fine binary search
            refine_ip_cost = 0;
            n_refine_user = 0;
            refine_user_time = 0;
            n_compute_item = 0;
            single_refine_user_time = 0;
            if (n_remain_result == 0) {
                return;
            }

            int tmp_candID = 0;
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID] || result_l[userID]) {
                    continue;
                } else {
                    user_candidate_l_[tmp_candID] = userID;
                    tmp_candID++;
                }
            }

            const int n_candidate = tmp_candID;

            record_.reset();
#pragma omp parallel for default(none) shared(n_candidate, queryIP_l, rank_lb_l, rank_ub_l, refine_ip_cost, n_compute_item, n_refine_user, single_refine_user_time, n_remain_result) num_threads(n_thread_)
            for (int candID = 0; candID < n_candidate; candID++) {
                const int userID = user_candidate_l_[candID];
                const float queryIP = queryIP_l[userID];
                int64_t tmp_n_compute_item = 0;
                int64_t tmp_ip_cost = 0;

                double tmp_pred_refinement_time = 0;
                int rank;
                bool is_refine = false;
                if (rank_lb_l[userID] == rank_ub_l[userID]) {
                    rank = rank_lb_l[userID];
                } else {
                    is_refine = true;
                    rank = refine_cpu_.RefineRank(queryIP, userID, tmp_n_compute_item,
                                                  tmp_ip_cost,
                                                  tmp_pred_refinement_time, omp_get_thread_num());
                }

                user_topk_cache_l_[candID] = UserRankElement(userID, rank, queryIP);
                ip_cost_parallel_l_[candID] = (int) tmp_ip_cost;
                n_compute_item_parallel_l_[candID] = (int) tmp_n_compute_item;
                is_refine_parallel_l_[candID] = is_refine ? 1 : 0;
                single_refine_user_time_parallel_l_[candID] = tmp_pred_refinement_time;

            }
            refine_user_time += record_.get_elapsed_time_second();

#pragma omp parallel for reduction(+: refine_ip_cost) default(none) shared(n_candidate)
            for (int candID = 0; candID < n_candidate; candID++) {
                refine_ip_cost += ip_cost_parallel_l_[candID];
            }

#pragma omp parallel for reduction(+: n_compute_item) default(none) shared(n_candidate)
            for (int candID = 0; candID < n_candidate; candID++) {
                n_compute_item += n_compute_item_parallel_l_[candID];
            }

#pragma omp parallel for reduction(+: n_refine_user) default(none) shared(n_candidate)
            for (int candID = 0; candID < n_candidate; candID++) {
                n_refine_user += is_refine_parallel_l_[candID];
            }

#pragma omp parallel for reduction(+: single_refine_user_time) default(none) shared(n_candidate)
            for (int candID = 0; candID < n_candidate; candID++) {
                single_refine_user_time += single_refine_user_time_parallel_l_[candID];
            }

            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_candidate,
                      std::less());
        }

        virtual void FinishCompute() override {

        }

    };
}
#endif //REVERSE_KRANKS_COMPUTEALLIPBOUND_HPP
