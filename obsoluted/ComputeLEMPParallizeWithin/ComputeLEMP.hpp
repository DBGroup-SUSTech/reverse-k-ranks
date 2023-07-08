//
// Created by bianzheng on 2023/5/16.
//

#ifndef REVERSE_KRANKS_COMPUTELEMP_HPP
#define REVERSE_KRANKS_COMPUTELEMP_HPP

#include "LEMP/mips/mips.h"

#include "alg/SpaceInnerProduct.hpp"
#include "struct/UserRankElement.hpp"
#include "alg/DiskIndex/BaseCompute.hpp"

#include <memory>
#include <omp.h>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class ComputeLEMP : public BaseCompute {
        int n_data_item_, n_user_, vec_dim_;
        const float *user_ptr_;
        mips::VectorMatrixLEMP data_item_LEMP_;
        mips::Lemp algo_ins_;
        std::vector<mips::VectorMatrixLEMP> user_lemp_l_;

        TimeRecord record_;

    public:

//        std::vector<UserRankElement> user_topk_cache_l_;

        inline ComputeLEMP() = default;

        inline ComputeLEMP(const VectorMatrix &user, const VectorMatrix &data_item) :
                data_item_LEMP_(data_item.getRawData(), data_item.n_vector_, data_item.vec_dim_) {
            this->user_ptr_ = user.getRawData();
            this->n_user_ = user.n_vector_;
            this->n_data_item_ = data_item.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->user_topk_cache_l_.resize(n_user_);

            mips::InputArguments args;
            args.k = 0;
            args.threads = omp_get_num_procs();
//            args.threads = 1;
            args.logFile = "none";

            // LEMP will also need some special arguments
            mips::LEMP_Method method = mips::LEMP_LI; // choose one of the methods LEMP provides (LEMP_X where X: L, LI, LC, I, C, TA, TREE, AP, LSH)
            int cacheSizeinKB = 2048; // specify your cache size per core (important since LEMP tries to optimize for cache utilization)
            float R = 0.60; // recall parameter for LEMP_LSH
            bool isTARR = false; // relevant for LEMP_TA (true will use ROUND ROBIN, false will use a MAX HEAP)
            float epsilon = 0.3; // epsilon value for LEMP_LI with absolute or relative approximation

            // create an instance of the algorithm you wish to use
            mips::Lemp algo(args, cacheSizeinKB, method, isTARR, R, epsilon);

            algo.initialize(data_item_LEMP_);
            algo_ins_ = algo;

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
                             double &refine_user_time) override {

            //read disk and fine binary search
            refine_ip_cost = 0;
            n_refine_user = 0;
            refine_user_time = 0;
            n_compute_item = 0;
            if (n_remain_result == 0) {
                return;
            }
            int n_candidate = 0;
            int n_result_rank_ub = 0;
            bool is_continue = false;
            record_.reset();
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID] || result_l[userID]) {
                    continue;
                }
                if (is_continue) {
                    continue;
                }

                int rank;
                size_t n_refine_ip;
                if (rank_lb_l[userID] == rank_ub_l[userID]) {
                    rank = rank_lb_l[userID];
                } else {

                    const int threadID = omp_get_thread_num();
                    const int this_rank_ub = rank_ub_l[userID];

                    algo_ins_.setTheta(queryIP_l[userID]);

                    algo_ins_.runAboveTheta(user_lemp_l_[userID], n_refine_ip, rank);
                }

#pragma omp critical
                {
                    refine_ip_cost += n_refine_ip;
                    n_compute_item += n_refine_ip;
                    user_topk_cache_l_[n_candidate] = UserRankElement(userID, rank, queryIP_l[userID]);
                    n_candidate++;

                    if (rank == rank_ub_l[userID]) {
                        n_result_rank_ub++;
                    }
                    if (n_result_rank_ub == n_remain_result) {
                        is_continue = true;
                    }
                };


            }

            refine_user_time += record_.get_elapsed_time_second();

            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_candidate,
                      std::less());
            n_refine_user = n_candidate;
        }

    };
}
#endif //REVERSE_KRANKS_COMPUTELEMP_HPP
