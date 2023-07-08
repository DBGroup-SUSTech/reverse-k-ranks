//
// Created by bianzheng on 2023/5/16.
//

#ifndef REVERSE_KRANKS_COMPUTELEMP_HPP
#define REVERSE_KRANKS_COMPUTELEMP_HPP

#include "LEMP/mips/mips.h"

#include "alg/SpaceInnerProduct.hpp"
#include "struct/UserRankElement.hpp"
#include "alg/DiskIndex/BaseCompute.hpp"
#include "struct/VectorMatrix.hpp"
#include "util/TimeMemory.hpp"

#include <memory>
#include <omp.h>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class ComputeLEMP : public BaseCompute {
        int n_data_item_, n_user_, vec_dim_;
        int n_thread_;
        const float *user_ptr_;
        mips::VectorMatrixLEMP data_item_LEMP_;
        std::vector<mips::Lemp> algo_ins_l_;
        std::vector<mips::VectorMatrixLEMP> user_lemp_l_;
        std::unique_ptr<int[]> user_candidate_l_;

        std::unique_ptr<double[]> initQueryBatches_time_parallel_l_;
        std::unique_ptr<double[]> initializeRetrievers_time_parallel_l_;
        std::unique_ptr<double[]> initListsInBuckets_time_parallel_l_;
        std::unique_ptr<double[]> tune_time_parallel_l_;
        std::unique_ptr<double[]> run_time_parallel_l_;
        std::unique_ptr<int[]> n_refine_ip_parallel_l_;
        std::unique_ptr<char[]> is_refine_parallel_l_;
        std::unique_ptr<double[]> single_refine_user_time_parallel_l_;

        double initQueryBatches_time = 0;
        double initializeRetrievers_time = 0;
        double initListsInBuckets_time = 0;
        double tune_time = 0;
        double run_time = 0;

        TimeRecord record_;

    public:

//        std::vector<UserRankElement> user_topk_cache_l_;

        inline ComputeLEMP() = default;

        inline ComputeLEMP(const VectorMatrix &user, const VectorMatrix &data_item,
                           const int &n_thread = omp_get_num_procs()) :
                data_item_LEMP_(data_item.getRawData(), data_item.n_vector_, data_item.vec_dim_) {
            this->user_ptr_ = user.getRawData();
            this->n_user_ = user.n_vector_;
            this->n_data_item_ = data_item.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->n_thread_ = n_thread;
            this->user_topk_cache_l_.resize(n_user_);
            this->user_candidate_l_ = std::make_unique<int[]>(n_user_);

            initQueryBatches_time_parallel_l_ = std::make_unique<double[]>(n_user_);
            initializeRetrievers_time_parallel_l_ = std::make_unique<double[]>(n_user_);
            initListsInBuckets_time_parallel_l_ = std::make_unique<double[]>(n_user_);
            tune_time_parallel_l_ = std::make_unique<double[]>(n_user_);
            run_time_parallel_l_ = std::make_unique<double[]>(n_user_);
            n_refine_ip_parallel_l_ = std::make_unique<int[]>(n_user_);
            is_refine_parallel_l_ = std::make_unique<char[]>(n_user_);
            single_refine_user_time_parallel_l_ = std::make_unique<double[]>(n_user_);

            algo_ins_l_.resize(omp_get_num_procs());

            for (int i = 0; i < omp_get_num_procs(); i++) {
                mips::InputArguments args;
                args.k = 0;
//                args.threads = omp_get_max_threads();
                args.threads = 1;
                args.logFile = "none";

                // LEMP will also need some special arguments
                mips::LEMP_Method method = mips::LEMP_LI; // choose one of the methods LEMP provides (LEMP_X where X: L, LI, LC, I, C, TA, TREE, AP, LSH)
                int cacheSizeinKB = 2048; // specify your cache size per core (important since LEMP tries to optimize for cache utilization)
                float R = 0.60; // recall parameter for LEMP_LSH
                bool isTARR = false; // relevant for LEMP_TA (true will use ROUND ROBIN, false will use a MAX HEAP)
                float epsilon = 0.3; // epsilon value for LEMP_LI with absolute or relative approximation

                // create an instance of the algorithm you wish to use
                mips::Lemp algo(args, cacheSizeinKB, method, isTARR, R, epsilon);

                // initialize the algorithm with a probe matrix (algorithm will store a copy of it inside)
                algo.initialize(data_item_LEMP_);

                algo_ins_l_[i] = algo;
            }

            user_lemp_l_.resize(n_user_);
            for (int userID = 0; userID < n_user_; userID++) {
                user_lemp_l_[userID] = mips::VectorMatrixLEMP(user_ptr_ + userID * vec_dim_, 1, vec_dim_);
            }

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
#pragma omp parallel for default(none) shared(n_candidate, rank_lb_l, rank_ub_l, queryIP_l, refine_ip_cost, n_compute_item, n_refine_user, single_refine_user_time, n_remain_result) num_threads(n_thread_)
            for (int candID = 0; candID < n_candidate; candID++) {
                const int userID = user_candidate_l_[candID];

                TimeRecord pred_refinement_record;
                pred_refinement_record.reset();
                double tmp_initQueryBatches_time = 0, tmp_initializeRetrievers_time = 0,
                        tmp_initListsInBuckets_time = 0, tmp_tune_time = 0, tmp_run_time = 0;
                int rank;
                bool is_refine = false;
                size_t n_refine_ip;
                if (rank_lb_l[userID] == rank_ub_l[userID]) {
                    rank = rank_lb_l[userID];
                } else {
                    is_refine = true;
                    const int threadID = omp_get_thread_num();

                    algo_ins_l_[threadID].setTheta(queryIP_l[userID]);

                    algo_ins_l_[threadID].runAboveTheta(n_data_item_, user_lemp_l_[userID], n_refine_ip, rank,
                                                        tmp_initQueryBatches_time, tmp_initializeRetrievers_time,
                                                        tmp_initListsInBuckets_time, tmp_tune_time, tmp_run_time);
                }
                const double tmp_pred_refinement_time = pred_refinement_record.get_elapsed_time_second();
                initQueryBatches_time_parallel_l_[candID] = tmp_initQueryBatches_time;
                initializeRetrievers_time_parallel_l_[candID] = tmp_initializeRetrievers_time;
                initListsInBuckets_time_parallel_l_[candID] = tmp_initListsInBuckets_time;
                tune_time_parallel_l_[candID] = tmp_tune_time;
                run_time_parallel_l_[candID] = tmp_run_time;
                n_refine_ip_parallel_l_[candID] = n_refine_ip;
                is_refine_parallel_l_[candID] = is_refine ? 1 : 0;
                single_refine_user_time_parallel_l_[candID] = tmp_pred_refinement_time;

                user_topk_cache_l_[candID] = UserRankElement(userID, rank, queryIP_l[userID]);

            }
            refine_user_time += record_.get_elapsed_time_second();

#pragma omp parallel for reduction(+:initQueryBatches_time, initializeRetrievers_time, initListsInBuckets_time, tune_time, run_time) default(none) shared(n_candidate)
            for (int candID = 0; candID < n_candidate; candID++) {
                initQueryBatches_time += initQueryBatches_time_parallel_l_[candID];
                initializeRetrievers_time += initializeRetrievers_time_parallel_l_[candID];
                initListsInBuckets_time += initListsInBuckets_time_parallel_l_[candID];
                tune_time += tune_time_parallel_l_[candID];
                run_time += run_time_parallel_l_[candID];
            }

#pragma omp parallel for reduction(+:refine_ip_cost, n_compute_item) default(none) shared(n_candidate)
            for (int candID = 0; candID < n_candidate; candID++) {
                refine_ip_cost += n_refine_ip_parallel_l_[candID];
                n_compute_item += n_refine_ip_parallel_l_[candID];
            }

#pragma omp parallel for reduction(+:n_refine_user) default(none) shared(n_candidate)
            for (int candID = 0; candID < n_candidate; candID++) {
                n_refine_user += is_refine_parallel_l_[candID];
            }

#pragma omp parallel for reduction(+:single_refine_user_time) default(none) shared(n_candidate)
            for (int candID = 0; candID < n_candidate; candID++) {
                single_refine_user_time += single_refine_user_time_parallel_l_[candID];
            }

            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_candidate,
                      std::less());
        }

        virtual void FinishCompute() override {
            spdlog::info(
                    "initQueryBatches_time {:.3f}s, initializeRetrievers_time {:.3f}s, initListsInBuckets_time {:.3f}s, tune_time {:.3f}s, run_time {:.3f}s",
                    initQueryBatches_time, initializeRetrievers_time,
                    initListsInBuckets_time, tune_time, run_time);
            initQueryBatches_time = 0;
            initializeRetrievers_time = 0;
            initListsInBuckets_time = 0;
            tune_time = 0;
            run_time = 0;
        }

    };
}
#endif //REVERSE_KRANKS_COMPUTELEMP_HPP
