//
// Created by bianzheng on 2023/5/26.
//

#ifndef REVERSE_KRANKS_LINEARSCANINDEX_HPP
#define REVERSE_KRANKS_LINEARSCANINDEX_HPP

#include "LEMP/mips/mips.h"

#include "alg/SpaceInnerProduct.hpp"
#include "struct/UserRankElement.hpp"
#include "alg/DiskIndex/BaseCompute.hpp"
#include "struct/VectorMatrix.hpp"
#include "util/TimeMemory.hpp"

#include <queue>
#include <memory>
#include <omp.h>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class LinearScanIndex {
        int n_data_item_, n_user_, vec_dim_;
        const float *user_ptr_;
        mips::VectorMatrixLEMP data_item_LEMP_;
        std::vector<mips::Lemp> algo_ins_l_;
        std::vector<mips::VectorMatrixLEMP> user_lemp_l_;

        int batch_n_user_ = 2048;
        int n_batch_;

        TimeRecord record_;

        double initQueryBatches_time = 0;
        double initializeRetrievers_time = 0;
        double initListsInBuckets_time = 0;
        double tune_time = 0;
        double run_time = 0;

    public:

        inline LinearScanIndex() = default;

        inline LinearScanIndex(const VectorMatrix &user, const VectorMatrix &data_item) :
                data_item_LEMP_(data_item.getRawData(), data_item.n_vector_, data_item.vec_dim_) {
            this->user_ptr_ = user.getRawData();
            this->n_user_ = user.n_vector_;
            this->n_data_item_ = data_item.n_vector_;
            this->vec_dim_ = user.vec_dim_;

            n_batch_ = n_user_ / batch_n_user_ + (n_user_ % batch_n_user_ == 0 ? 0 : 1);

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

        void GetRank(const std::vector<float> &queryIP_l, const int &topk, const double &remain_time,
                     std::vector<UserRankElement> &topk_result_l, size_t &refine_ip_cost, bool &is_finish) {

            record_.reset();

            if (topk > batch_n_user_) {
                batch_n_user_ = topk;
                n_batch_ = n_user_ / batch_n_user_ + (n_user_ % batch_n_user_ == 0 ? 0 : 1);
            }

            std::priority_queue<UserRankElement, std::vector<UserRankElement>, std::less<>> max_heap;

            assert(topk_result_l.size() == topk);
            //read disk and fine binary search
            refine_ip_cost = 0;
            int n_candidate = 0;
            is_finish = true;

            int threshold = n_data_item_ + 1;

            {//batchID == 0
                const int batchID = 0;
                const int start_userID = batchID * batch_n_user_;
                const int end_userID = std::min(topk, (batchID + 1) * batch_n_user_);

#pragma omp parallel for default(none) shared(start_userID, end_userID, queryIP_l, refine_ip_cost, n_candidate, topk, max_heap, threshold)
                for (int userID = start_userID; userID < end_userID; userID++) {
                    double tmp_initQueryBatches_time = 0, tmp_initializeRetrievers_time = 0,
                            tmp_initListsInBuckets_time = 0, tmp_tune_time = 0, tmp_run_time = 0;
                    const int threadID = omp_get_thread_num();

                    size_t tmp_n_refine_ip = 0;
                    int rank = 0;

                    algo_ins_l_[threadID].setTheta(queryIP_l[userID]);

                    algo_ins_l_[threadID].runAboveTheta(threshold, user_lemp_l_[userID], tmp_n_refine_ip, rank,
                                                        tmp_initQueryBatches_time, tmp_initializeRetrievers_time,
                                                        tmp_initListsInBuckets_time, tmp_tune_time, tmp_run_time);

#pragma omp critical
                    {
                        initQueryBatches_time += tmp_initQueryBatches_time;
                        initializeRetrievers_time += tmp_initializeRetrievers_time;
                        initListsInBuckets_time += tmp_initListsInBuckets_time;
                        tune_time += tmp_tune_time;
                        run_time += tmp_run_time;

                        max_heap.emplace(userID, rank, queryIP_l[userID]);
                        refine_ip_cost += tmp_n_refine_ip;
                        n_candidate++;
                    }
                }

                threshold = max_heap.top().rank_;

            }

            for (int batchID = 0; batchID < n_batch_; batchID++) {

                const int start_userID = batchID == 0 ? topk : batchID * batch_n_user_;
                const int end_userID = std::min((batchID + 1) * batch_n_user_, n_user_);

#pragma omp parallel for default(none) shared(start_userID, end_userID, queryIP_l, refine_ip_cost, n_candidate, threshold, max_heap)
                for (int userID = start_userID; userID < end_userID; userID++) {
                    double tmp_initQueryBatches_time = 0, tmp_initializeRetrievers_time = 0,
                            tmp_initListsInBuckets_time = 0, tmp_tune_time = 0, tmp_run_time = 0;
                    const int threadID = omp_get_thread_num();

                    size_t tmp_n_refine_ip = 0;
                    int rank = 0;

                    algo_ins_l_[threadID].setTheta(queryIP_l[userID]);

                    algo_ins_l_[threadID].runAboveTheta(threshold, user_lemp_l_[userID], tmp_n_refine_ip, rank,
                                                        tmp_initQueryBatches_time, tmp_initializeRetrievers_time,
                                                        tmp_initListsInBuckets_time, tmp_tune_time, tmp_run_time);


#pragma omp critical
                    {
                        initQueryBatches_time += tmp_initQueryBatches_time;
                        initializeRetrievers_time += tmp_initializeRetrievers_time;
                        initListsInBuckets_time += tmp_initListsInBuckets_time;
                        tune_time += tmp_tune_time;
                        run_time += tmp_run_time;

                        if (rank <= threshold) {
                            max_heap.pop();
                            max_heap.emplace(userID, rank, queryIP_l[userID]);
                            threshold = max_heap.top().rank_;
                        }

                        refine_ip_cost += tmp_n_refine_ip;
                        n_candidate++;
                    }
                }

                if (record_.get_elapsed_time_second() > remain_time) {
                    is_finish = false;
                    break;
                }
            }

            assert(max_heap.size() == topk);

            for (int i = 0; i < topk; i++) {
                UserRankElement ele = max_heap.top();
                topk_result_l[i] = ele;
                max_heap.pop();
            }

        }

        void FinishCompute() {
            spdlog::info(
                    "initQueryBatches_time {:.3f}s, initializeRetrievers_time {:.3f}s, initListsInBuckets_time {:.3f}s, tune_time {:.3f}s, run_time {:.3f}s",
                    initQueryBatches_time, initializeRetrievers_time,
                    initListsInBuckets_time, tune_time, run_time);
        }

    };
}
#endif //REVERSE_KRANKS_LINEARSCANINDEX_HPP
