//
// Created by bianzheng on 2023/6/16.
//

#ifndef REVERSE_KRANKS_COMPUTEALLIPBOUNDUPDATE_HPP
#define REVERSE_KRANKS_COMPUTEALLIPBOUNDUPDATE_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "struct/UserRankElement.hpp"
#include "util/TimeMemory.hpp"
#include "alg/DiskIndex/RefineByComputation/RefineCPUIPBoundUpdate.hpp"
#include "alg/DiskIndex/BaseCompute.hpp"

#include <memory>
#include <omp.h>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class ComputeAllIPBoundUpdate {
        int n_user_, vec_dim_;
        int n_thread_;

        TimeRecord record_;
        RefineCPUIPBoundUpdate refine_cpu_;
        std::unique_ptr<int[]> user_candidate_l_;

        std::unique_ptr<int[]> ip_cost_parallel_l_;
        std::unique_ptr<int[]> n_compute_item_parallel_l_;
        std::unique_ptr<int[]> is_refine_parallel_l_;
        std::unique_ptr<double[]> single_refine_user_time_parallel_l_;

    public:

        std::vector<UserRankElement> user_topk_cache_l_;

        inline ComputeAllIPBoundUpdate() = default;

        inline ComputeAllIPBoundUpdate(const VectorMatrixUpdate &user, const VectorMatrixUpdate &data_item, const int &check_dim,
                                       const int &n_thread = omp_get_num_procs()) {
            this->n_user_ = user.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->n_thread_ = n_thread;

            refine_cpu_ = RefineCPUIPBoundUpdate(user, data_item, check_dim, n_thread);
            ProcessUser();
        }

        void ProcessUser() {
            this->user_topk_cache_l_.resize(n_user_);
            this->user_candidate_l_ = std::make_unique<int[]>(n_user_);

            this->ip_cost_parallel_l_ = std::make_unique<int[]>(n_user_);
            this->n_compute_item_parallel_l_ = std::make_unique<int[]>(n_user_);
            this->is_refine_parallel_l_ = std::make_unique<int[]>(n_user_);
            this->single_refine_user_time_parallel_l_ = std::make_unique<double[]>(n_user_);
        }

        void InsertUser(const VectorMatrixUpdate &insert_user) {
            n_user_ += insert_user.n_vector_;
            ProcessUser();
            refine_cpu_.InsertUser(insert_user);
        }

        void DeleteUser(const std::vector<int> &del_userID_l) {
            n_user_ -= del_userID_l.size();
            ProcessUser();
            refine_cpu_.DeleteUser(del_userID_l);
        }

        void InsertItem(const VectorMatrixUpdate &insert_data_item) {
            refine_cpu_.InsertItem(insert_data_item);
        }

        void DeleteItem(const VectorMatrixUpdate &data_item_after, const int &n_delete_item) {
            refine_cpu_.DeleteItem(data_item_after, n_delete_item);
        }

        virtual void GetRank(const std::vector<float> &queryIP_l,
                             const std::vector<int> &rank_lb_l, const std::vector<int> &rank_ub_l,
                             const std::vector<char> &prune_l, const std::vector<char> &result_l,
                             const int &n_remain_result, size_t &refine_ip_cost, int &n_refine_user,
                             int64_t &n_compute_item,
                             double &refine_user_time, double &single_refine_user_time, const int &queryID) {

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
#pragma omp parallel for default(none) shared(queryID, n_candidate, queryIP_l, rank_lb_l, rank_ub_l, refine_ip_cost, n_compute_item, n_refine_user, single_refine_user_time, n_remain_result) num_threads(n_thread_)
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

                assert(rank_ub_l[userID] <= rank && rank <= rank_lb_l[userID]);

                user_topk_cache_l_[candID] = UserRankElement(userID, rank, queryIP);
                ip_cost_parallel_l_[candID] = (int) tmp_ip_cost;
                n_compute_item_parallel_l_[candID] = (int) tmp_n_compute_item;
                is_refine_parallel_l_[candID] = is_refine ? 1 : 0;
                single_refine_user_time_parallel_l_[candID] = tmp_pred_refinement_time;

//                if (queryID == 55 && userID == 232) {
//                    spdlog::info(
//                            "queryID: {}, userID: {}, queryIP: {:.3f}, rank_lb: {}, rank_ub: {}, rank {}",
//                            queryID, userID, queryIP_l[userID],
//                            rank_lb_l[userID], rank_ub_l[userID], rank);
//                    printf("is_refine %d\n", is_refine);
//                }
//                if (queryID == 55 && userID == 16) {
//                    spdlog::info(
//                            "queryID: {}, userID: {}, queryIP: {:.3f}, rank_lb: {}, rank_ub: {}, rank {}",
//                            queryID, userID, queryIP_l[userID],
//                            rank_lb_l[userID], rank_ub_l[userID], rank);
//                    printf("is_refine %d\n", is_refine);
//                }

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

        virtual void FinishCompute() {

        }

    };
}
#endif //REVERSE_KRANKS_COMPUTEALLIPBOUNDUPDATE_HPP
