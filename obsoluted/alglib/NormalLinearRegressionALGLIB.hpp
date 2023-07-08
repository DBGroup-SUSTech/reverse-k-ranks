//
// Created by bianzheng on 2023/5/5.
//

#ifndef REVERSE_KRANKS_NORMALLINEARREGRESSIONALGLIB_HPP
#define REVERSE_KRANKS_NORMALLINEARREGRESSIONALGLIB_HPP

#include "alg/RegressionPruning/BaseLinearRegression.hpp"
#include "struct/DistancePair.hpp"
#include "util/MathUtil.hpp"
#include "linear_program/sdlp/sdlp.hpp"

#include "ALGLIB/stdafx.h"
#include "ALGLIB/optimization.h"

#include <iostream>
#include <memory>
#include <cfloat>
#include <eigen3/Eigen/Dense>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class NormalLinearRegressionALGLIB : public BaseLinearRegression {

        std::string method_name_;
        size_t n_data_item_, n_user_;
        static constexpr int n_predict_parameter_ = 2; // (a, b) for linear estimation
        static constexpr int n_distribution_parameter_ = 2; // mu, sigma
        static constexpr float sqrt_2_ = sqrt(2.0);
        int n_sample_rank_;
        std::unique_ptr<int[]> sample_rank_l_; // n_sample_rank
        std::unique_ptr<float[]> predict_para_l_; // n_user_ * n_predict_parameter
        std::unique_ptr<float[]> distribution_para_l_; // n_user_ * n_distribution_parameter
        std::unique_ptr<float[]> error_l_; //n_user_

        int n_batch_;

        //used for loading
        std::vector<double *> batch_cache_A_l_; // for each A, (4 * n_sample_rank) * (n_predict_parameter_ + 1), constraint matrix
        double *preprocess_cache_left_constrain_; // (4 * n_sample_rank)
        double *preprocess_cache_right_constrain_; // (4 * n_sample_rank), constraint bound
        double *preprocess_cache_c_; // 3, objective coefficients
    public:

        inline NormalLinearRegressionALGLIB() {}

        inline NormalLinearRegressionALGLIB(const int &n_data_item, const int &n_user, const std::string &method_name) {
            this->method_name_ = method_name;
            this->n_data_item_ = n_data_item;
            this->n_user_ = n_user;
            this->predict_para_l_ = std::make_unique<float[]>(n_user * n_predict_parameter_);
            this->distribution_para_l_ = std::make_unique<float[]>(n_user * n_distribution_parameter_);
            this->error_l_ = std::make_unique<float[]>(n_user);
            this->n_batch_ = (int) n_user_ / batch_n_user_ + (n_user_ % batch_n_user_ == 0 ? 0 : 1);
            static_assert(n_predict_parameter_ == 2 && n_distribution_parameter_ == 2);
        }

        inline NormalLinearRegressionALGLIB(const char *index_basic_dir, const char *dataset_name,
                                            const std::string &method_name,
                                            const size_t &n_sample, const size_t &n_sample_query,
                                            const size_t &sample_topk) {
            LoadIndex(index_basic_dir, dataset_name, method_name,
                      n_sample, n_sample_query, sample_topk);
            this->method_name_ = method_name;
            this->n_batch_ = (int) n_user_ / batch_n_user_ + (n_user_ % batch_n_user_ == 0 ? 0 : 1);
        }

        void StartPreprocess(const int *sample_rank_l, const int &n_sample_rank) override {
            this->n_sample_rank_ = n_sample_rank;
            this->sample_rank_l_ = std::make_unique<int[]>(n_sample_rank);

            this->batch_cache_A_l_.resize(batch_n_user_);
            for (int batchID = 0; batchID < batch_n_user_; batchID++) {
                batch_cache_A_l_[batchID] = new double[(4 * n_sample_rank) * (n_predict_parameter_ + 1)];
            }

            preprocess_cache_left_constrain_ = new double[4 * n_sample_rank];
            for (int i = 0; i < 4 * n_sample_rank; i++) {
                preprocess_cache_left_constrain_[i] = alglib::fp_neginf;
            }

            this->preprocess_cache_right_constrain_ = new double[4 * n_sample_rank];
            this->preprocess_cache_c_ = new double[3];
            for (int sampleID = 0; sampleID < n_sample_rank_ - 1; sampleID++) {
                sample_rank_l_[sampleID] = sample_rank_l[sampleID];

                preprocess_cache_right_constrain_[4 * sampleID] = -sampleID - 1;
                preprocess_cache_right_constrain_[4 * sampleID + 1] = sampleID + 1;
                preprocess_cache_right_constrain_[4 * sampleID + 2] = -sampleID - 1;
                preprocess_cache_right_constrain_[4 * sampleID + 3] = sampleID + 1;
            }
            preprocess_cache_right_constrain_[4 * (n_sample_rank_ - 1) + 0] = -n_sample_rank_;
            preprocess_cache_right_constrain_[4 * (n_sample_rank_ - 1) + 1] = n_sample_rank_;
            preprocess_cache_right_constrain_[4 * (n_sample_rank_ - 1) + 2] = 0;
            preprocess_cache_right_constrain_[4 * (n_sample_rank_ - 1) + 3] = 0;
            sample_rank_l_[n_sample_rank_ - 1] = sample_rank_l[n_sample_rank_ - 1];

            preprocess_cache_c_[0] = 0;
            preprocess_cache_c_[1] = 0;
            preprocess_cache_c_[2] = 1;

        }

        float ComputeAverage(const float *sampleIP_l) const {
            float average = 0;
            for (int sampleID = 0; sampleID < n_sample_rank_; sampleID++) {
                average += sampleIP_l[sampleID];
            }
            return average / (float) n_sample_rank_;
        }

        float ComputeStd(const float *sampleIP_l, const float &average) const {
            float sigma = 0;
            for (int sampleID = 0; sampleID < n_sample_rank_; sampleID++) {
                const float minus = sampleIP_l[sampleID] - average;
                const float term = minus * minus;
                sigma += term;
            }
            sigma /= (float) n_sample_rank_;
            return std::sqrt(sigma);
        }

        float CDFPhi(float x) const {
            // constants
            constexpr float a1 = 0.254829592;
            constexpr float a2 = -0.284496736;
            constexpr float a3 = 1.421413741;
            constexpr float a4 = -1.453152027;
            constexpr float a5 = 1.061405429;
            constexpr float p = 0.3275911;

            // Save the sign of x
            int sign = 1;
            if (x < 0)
                sign = -1;
            x = std::fabs(x) / sqrt_2_;

            // A&S formula 7.1.26
            float t = 1.0 / (1.0 + p * x);
            float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

            return 0.5 * (1.0 + sign * y);
        }

        void AssignCache(double *preprocess_cache_A, const float *sampleIP_l, const int &userID) {
            //compute average, std
            const float mu = ComputeAverage(sampleIP_l);
            const float sigma = ComputeStd(sampleIP_l, mu);
            distribution_para_l_[userID * n_distribution_parameter_] = mu;
            distribution_para_l_[userID * n_distribution_parameter_ + 1] = sigma;

//#pragma omp parallel for default(none) shared(sampleIP_l, mu, sigma)
            for (int sampleID = 0; sampleID < n_sample_rank_ - 1; sampleID++) {
                const float input_next_x = sampleIP_l[sampleID + 1];
                const float normal_num = (input_next_x - mu) / sigma;
                const float next_cdf = CDFPhi(normal_num);
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1)] = -next_cdf;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 1] = -1;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 2] = -1;

                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 3] = next_cdf;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 4] = 1;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 5] = -1;

                const float input_x = sampleIP_l[sampleID];
                const float normal_num2 = (input_x - mu) / sigma;
                const float cdf = CDFPhi(normal_num2);
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 6] = -cdf;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 7] = -1;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 8] = -1;

                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 9] = cdf;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 10] = 1;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 11] = -1;
            }

            {
                const int sampleID = n_sample_rank_ - 1;
                const float input_x = sampleIP_l[sampleID];
                const float normal_num2 = (input_x - mu) / sigma;
                const float cdf = CDFPhi(normal_num2);
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 0] = -cdf;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 1] = -1;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 2] = -1;

                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 3] = cdf;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 4] = 1;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 5] = -1;
            }

            {
                const int sampleID = 0;
                const float input_x = sampleIP_l[sampleID];
                const float normal_num2 = (input_x - mu) / sigma;
                const float cdf = CDFPhi(normal_num2);
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 6] = -cdf;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 7] = -1;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 8] = -1;

                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 9] = cdf;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 10] = 1;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 11] = -1;
            }

        }

        void LinearProgram(double *preprocess_cache_A, const int &userID) {
            const int n_row = 4 * n_sample_rank_;
            const int n_col = n_predict_parameter_ + 1;
            alglib::real_2d_array a;
            a.setcontent(n_row, n_col, preprocess_cache_A);
//            a.attach_to_ptr(n_row, n_col, preprocess_cache_A);

            alglib::real_1d_array left_constrain;
            left_constrain.setcontent(n_row, preprocess_cache_left_constrain_);
//            left_constrain.attach_to_ptr(n_row, preprocess_cache_left_constrain_);

            alglib::real_1d_array right_constrain;
            right_constrain.setcontent(n_row, preprocess_cache_right_constrain_);
//            right_constrain.attach_to_ptr(n_row, preprocess_cache_right_constrain_);

            alglib::real_1d_array c;
            c.setcontent(n_col, preprocess_cache_c_);
//            c.attach_to_ptr(n_col, preprocess_cache_c_);

//            alglib::real_1d_array s = "[1000,1000,1]";
//            alglib::real_1d_array bndl;
//            std::vector<double> bndl_l = {alglib::fp_neginf, alglib::fp_neginf, alglib::fp_neginf};
//            bndl.setcontent(n_col, bndl_l.data());
//            alglib::real_1d_array bndu;
//            std::vector<double> bndu_l = {alglib::fp_posinf, alglib::fp_posinf, alglib::fp_posinf};
//            bndl.setcontent(n_col, bndu_l.data());

            alglib::real_1d_array x;
            alglib::minlpstate state;
            alglib::minlpreport rep;

            minlpcreate(n_col, state);
//            printf("after create\n");
            minlpsetcost(state, c);
//            printf("after setcost\n");
//            minlpsetbc(state, bndl, bndu);
            minlpsetbcall(state, alglib::fp_neginf, alglib::fp_posinf);
//            printf("after setbc\n");
            minlpsetlc2dense(state, a, left_constrain, right_constrain, n_row);
//            printf("after lc2dense\n");

//            minlpsetscale(state, s);
//            printf("after scale\n");

            // Solve
//            try {
            minlpoptimize(state);
//            } catch (alglib::ap_error &e) {
//                printf("error msg: %s\n", e.msg.c_str());
//            }
//            printf("after optimize\n");
            minlpresults(state, x, rep);
//            printf("after results\n");

            if (!((1 <= rep.terminationtype && rep.terminationtype <= 4) || rep.terminationtype == 7)) {
                ::printf("terminationtype %td\n", rep.terminationtype);
            }
            assert((1 <= rep.terminationtype && rep.terminationtype <= 4) || rep.terminationtype == 7);
//            printf("function value %.3f\n", rep.f);

            float error = (float) rep.f;
//            printf("original error %.3f, a %.3f, b %.3f, epsilon %.3f\n", error, (float) x.getcontent()[0],
//                   (float) x.getcontent()[1], (float) x.getcontent()[2]);
            if (error < 0 || error > (float) n_sample_rank_) {
                error = (float) n_sample_rank_;
            }

            error_l_[userID] = error;

            //assign parameter
            for (int paraID = 0; paraID < n_predict_parameter_; paraID++) {
                predict_para_l_[userID * n_predict_parameter_ + paraID] = (float) x.getcontent()[paraID];
            }

        }

        void CalcError(const float *sampleIP_l, const int &userID, const float &mu, const float &sigma) {
            //assign error
            float error = -1;
//#pragma omp parallel for default(none) reduction(+:error) shared(userID, sampleIP_l, mu, sigma)
            for (int sampleID = 0; sampleID < n_sample_rank_ - 1; sampleID++) {
                const float input_x = sampleIP_l[sampleID];
                const float normal_num2 = (input_x - mu) / sigma;
                const float cdf = CDFPhi(normal_num2);

                const float input_next_x = sampleIP_l[sampleID + 1];
                const float normal_num = (input_next_x - mu) / sigma;
                const float next_cdf = CDFPhi(normal_num);

                const float tmp_a = predict_para_l_[userID * n_predict_parameter_];
                const float tmp_b = predict_para_l_[userID * n_predict_parameter_ + 1];
                const float pred_rank1 = next_cdf * tmp_a + tmp_b;
                const float pred_rank2 = cdf * tmp_a + tmp_b;
                const float tmp_error1 = std::abs((float) (sampleID + 1) - pred_rank1);
                const float tmp_error2 = std::abs((float) (sampleID + 1) - pred_rank2);
                const float tmp_error = std::max(tmp_error1, tmp_error2) + 0.01f;
//#pragma omp critical
                error = std::max(tmp_error, error);
            }

            {
                const int sampleID = n_sample_rank_ - 1;
                const float input_x = sampleIP_l[sampleID];
                const float normal_num2 = (input_x - mu) / sigma;
                const float cdf = CDFPhi(normal_num2);

                const float tmp_a = predict_para_l_[userID * n_predict_parameter_];
                const float tmp_b = predict_para_l_[userID * n_predict_parameter_ + 1];
                const float pred_rank2 = cdf * tmp_a + tmp_b;
                const float tmp_error = std::abs((float) (sampleID + 1) - pred_rank2) + 0.01f;
                error = std::max(tmp_error, error);
            }

            {
                const int sampleID = 0;
                const float input_x = sampleIP_l[sampleID];
                const float normal_num2 = (input_x - mu) / sigma;
                const float cdf = CDFPhi(normal_num2);

                const float tmp_a = predict_para_l_[userID * n_predict_parameter_];
                const float tmp_b = predict_para_l_[userID * n_predict_parameter_ + 1];
                const float pred_rank2 = cdf * tmp_a + tmp_b;
                const float tmp_error = std::abs((float) sampleID - pred_rank2) + 0.01f;
                error = std::max(tmp_error, error);
            }

            error = std::min(error, (float) n_sample_rank_);
            error_l_[userID] = error;
            assert(error > 0);
            const float tmp_a = predict_para_l_[userID * n_predict_parameter_];
            const float tmp_b = predict_para_l_[userID * n_predict_parameter_ + 1];
            if (userID % 10000 == 0) {
                printf("userID %d, a %.3f, b %.3f, error_l %.3f, n_sample_rank %d\n",
                       userID, tmp_a, tmp_b, error_l_[userID], n_sample_rank_);
            }
            assert(-1 < error && error < n_sample_rank_ + 0.1);
        }

        void BatchLoopPreprocess(const std::vector<const float *> &sampleIP_l_l,
                                 const int &start_userID, const int &n_proc_user,
                                 double &assign_cache_time, double &linear_program_time,
                                 double &calc_error_time) override {
            TimeRecord record;
            record.reset();

#pragma omp parallel for default(none) shared(start_userID, n_proc_user, sampleIP_l_l, n_sample_rank_, distribution_para_l_)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const float *sampleIP_l = sampleIP_l_l[proc_userID];
                double *preprocess_cache_A = batch_cache_A_l_[proc_userID];
                const int &userID = proc_userID + start_userID;
                AssignCache(preprocess_cache_A, sampleIP_l, userID);
            }
            assign_cache_time = record.get_elapsed_time_second();

            record.reset();
#pragma omp parallel for default(none) shared(n_proc_user, start_userID, n_sample_rank_, predict_para_l_, batch_cache_A_l_)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                double *preprocess_cache_A = batch_cache_A_l_[proc_userID];
                const int userID = proc_userID + start_userID;
                LinearProgram(preprocess_cache_A, userID);
            }
            linear_program_time = record.get_elapsed_time_second();

            record.reset();
//#pragma omp parallel for default(none) shared(sampleIP_l_l, n_proc_user, n_sample_rank_, predict_para_l_, error_l_, start_userID, distribution_para_l_)
//            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
//                const float *sampleIP_l = sampleIP_l_l[proc_userID];
//                const int userID = proc_userID + start_userID;
//
//                const float mu = distribution_para_l_[userID * n_distribution_parameter_];
//                const float sigma = distribution_para_l_[userID * n_distribution_parameter_ + 1];
//                CalcError(sampleIP_l, userID, mu, sigma);
//            }
            calc_error_time = record.get_elapsed_time_second();

        }

        void FinishPreprocess() override {
            for (int batchID = 0; batchID < batch_n_user_; batchID++) {
                delete[] batch_cache_A_l_[batchID];
                batch_cache_A_l_[batchID] = nullptr;
            }
            batch_cache_A_l_.clear();
            delete[] preprocess_cache_left_constrain_;
            delete[] preprocess_cache_right_constrain_;
            delete[] preprocess_cache_c_;
            preprocess_cache_left_constrain_ = nullptr;
            preprocess_cache_right_constrain_ = nullptr;
            preprocess_cache_c_ = nullptr;
        }

        inline void
        ComputeRankBound(const float &queryIP, const int &userID,
                         int &rank_lb, int &rank_ub, const int &queryID) const {

            if (error_l_[userID] >= n_sample_rank_ - 1e-3) {
                rank_lb = (int) n_data_item_;
                rank_ub = 0;
                return;
            }

            const size_t distribution_pos = userID * n_distribution_parameter_;
            const float mu = distribution_para_l_[distribution_pos];
            const float sigma = distribution_para_l_[distribution_pos + 1];
            const float normalize_x = (queryIP - mu) / sigma;
            const float input_x = CDFPhi(normalize_x);

            const size_t pred_pos = userID * n_predict_parameter_;
            const float pred_rank = input_x * predict_para_l_[pred_pos] + predict_para_l_[pred_pos + 1];
            const int pred_sample_rank_lb = std::ceil(pred_rank + error_l_[userID]) + 1;
            const int pred_sample_rank_ub = std::floor(pred_rank - error_l_[userID]) - 1;
            assert(0 <= error_l_[userID] && error_l_[userID] <= n_sample_rank_ + 0.1);

//            if (queryID == 3 && userID == 865) {
//                printf("queryID %d, userID %d, queryIP %.3f, pred_int_rank %d, error %d, pred_sample_rank_lb %d, pred_sample_rank_ub %d\n",
//                       queryID, userID, queryIP, pred_int_rank, error_l_[userID], pred_sample_rank_lb,
//                       pred_sample_rank_ub);
//            }

            if (pred_sample_rank_lb >= n_sample_rank_) {
                rank_lb = (int) n_data_item_;
            } else if (pred_sample_rank_lb < 0) {
                rank_lb = sample_rank_l_[0];
            } else {
                rank_lb = sample_rank_l_[pred_sample_rank_lb];
            }

            if (pred_sample_rank_ub >= n_sample_rank_) {
                rank_ub = (int) n_data_item_;
            } else if (pred_sample_rank_ub < 0) {
                rank_ub = 0;
            } else {
                rank_ub = sample_rank_l_[pred_sample_rank_ub];
            }

        }


        void RankBound(const std::vector<std::pair<float, float>> &queryIP_l,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l, const int &queryID) const {

#pragma omp parallel for default(none) shared(queryIP_l, queryID, rank_lb_l, rank_ub_l) num_threads(omp_get_num_procs())
            for (int batchID = 0; batchID < n_batch_; batchID++) {
                const int start_userID = batchID * batch_n_user_;
                const int end_userID = std::min((int) n_user_, (batchID + 1) * batch_n_user_);
                for (int userID = start_userID; userID < end_userID; userID++) {
                    const float queryIP_lb = queryIP_l[userID].first;
                    int qIP_lb_tmp_lower_rank, qIP_lb_tmp_upper_rank;

                    ComputeRankBound(queryIP_lb, userID,
                                     qIP_lb_tmp_lower_rank, qIP_lb_tmp_upper_rank, queryID);
                    assert(qIP_lb_tmp_upper_rank <= qIP_lb_tmp_lower_rank);

                    const float queryIP_ub = queryIP_l[userID].second;
                    assert(queryIP_lb <= queryIP_ub);
                    int qIP_ub_tmp_lower_rank, qIP_ub_tmp_upper_rank;
                    ComputeRankBound(queryIP_ub, userID,
                                     qIP_ub_tmp_lower_rank, qIP_ub_tmp_upper_rank, queryID);
                    assert(qIP_ub_tmp_upper_rank <= qIP_ub_tmp_lower_rank);

                    rank_lb_l[userID] = qIP_lb_tmp_lower_rank;
                    rank_ub_l[userID] = qIP_ub_tmp_upper_rank;

//                    if (userID % 50000 == 0) {
//                        const size_t pred_pos = userID * n_predict_parameter_;
//                        printf("userID %d, rank_lb %d, rank_ub %d, queryIP_lb %.3f, queryIP_ub %.3f, error %.3f, a %.3f, b %.3f\n",
//                               userID, rank_lb_l[userID], rank_ub_l[userID], queryIP_lb, queryIP_ub, error_l_[userID],
//                               predict_para_l_[pred_pos], predict_para_l_[pred_pos + 1]);
//
//                    }
                    if (qIP_ub_tmp_upper_rank > qIP_lb_tmp_lower_rank) {
                        const size_t pred_pos = userID * n_predict_parameter_;
                        spdlog::error(
                                "qIP_lb_tmp_lower_rank {}, qIP_lb_tmp_upper_rank {}, qIP_ub_tmp_lower_rank {}, qIP_ub_tmp_upper_rank {}",
                                qIP_lb_tmp_lower_rank, qIP_lb_tmp_upper_rank, qIP_ub_tmp_lower_rank,
                                qIP_ub_tmp_upper_rank);

                        const size_t distribution_pos = userID * n_distribution_parameter_;
                        const float mu = distribution_para_l_[distribution_pos];
                        const float sigma = distribution_para_l_[distribution_pos + 1];
                        const float normalize_x_lb = (queryIP_l[userID].first - mu) / sigma;
                        const float input_x_lb = CDFPhi(normalize_x_lb);

                        const float normalize_x_ub = (queryIP_l[userID].second - mu) / sigma;
                        const float input_x_ub = CDFPhi(normalize_x_ub);

                        spdlog::error("error {}, a {}, b {}", error_l_[userID], predict_para_l_[pred_pos],
                                      predict_para_l_[pred_pos + 1]);
                        spdlog::error("queryIP_lb {}, queryIP_ub {}, input_x_lb {}, input_x_ub {}",
                                      queryIP_l[userID].first, queryIP_l[userID].second,
                                      input_x_lb, input_x_ub);
                    }
                    assert(qIP_ub_tmp_upper_rank <= qIP_lb_tmp_lower_rank);
                }
            }
        }

        void SaveIndex(const char *index_basic_dir, const char *dataset_name, const size_t &n_sample_query,
                       const size_t &sample_topk) override {
            char index_path[256];
            sprintf(index_path,
                    "%s/memory_index/NormalLinearRegressionALGLIB-%s-%s-n_sample_%d-n_sample_query_%ld-sample_topk_%ld.index",
                    index_basic_dir, method_name_.c_str(), dataset_name, n_sample_rank_, n_sample_query, sample_topk);


            std::ofstream out_stream_ = std::ofstream(index_path, std::ios::binary | std::ios::out);
            if (!out_stream_) {
                spdlog::error("error in write result, not found index");
                exit(-1);
            }
            out_stream_.write((char *) &n_data_item_, sizeof(size_t));
            out_stream_.write((char *) &n_user_, sizeof(size_t));
            out_stream_.write((char *) &n_sample_rank_, sizeof(int));

            out_stream_.write((char *) sample_rank_l_.get(), (int64_t) (n_sample_rank_ * sizeof(int)));
            out_stream_.write((char *) predict_para_l_.get(),
                              (int64_t) (n_user_ * n_predict_parameter_ * sizeof(float)));
            out_stream_.write((char *) distribution_para_l_.get(),
                              (int64_t) (n_user_ * n_distribution_parameter_ * sizeof(float)));
            out_stream_.write((char *) error_l_.get(), (int64_t) (n_user_ * sizeof(float)));

            out_stream_.close();
        }

        void LoadIndex(const char *index_basic_dir, const char *dataset_name, const std::string &method_name,
                       const size_t &n_sample, const size_t &n_sample_query, const size_t &sample_topk) {
            char index_path[256];
            sprintf(index_path,
                    "%s/memory_index/NormalLinearRegressionALGLIB-%s-%s-n_sample_%ld-n_sample_query_%ld-sample_topk_%ld.index",
                    index_basic_dir, method_name.c_str(), dataset_name, n_sample, n_sample_query, sample_topk);
            spdlog::info("index path {}", index_path);

            std::ifstream index_stream = std::ifstream(index_path, std::ios::binary | std::ios::in);
            if (!index_stream) {
                spdlog::error("error in reading index");
                exit(-1);
            }

            index_stream.read((char *) &n_data_item_, sizeof(size_t));
            index_stream.read((char *) &n_user_, sizeof(size_t));
            index_stream.read((char *) &n_sample_rank_, sizeof(int));

            sample_rank_l_ = std::make_unique<int[]>(n_sample_rank_);
            index_stream.read((char *) sample_rank_l_.get(), (int64_t) (sizeof(int) * n_sample_rank_));

            predict_para_l_ = std::make_unique<float[]>(n_user_ * n_predict_parameter_);
            index_stream.read((char *) predict_para_l_.get(),
                              (int64_t) (sizeof(float) * n_user_ * n_predict_parameter_));

            distribution_para_l_ = std::make_unique<float[]>(n_user_ * n_distribution_parameter_);
            index_stream.read((char *) distribution_para_l_.get(),
                              (int64_t) (sizeof(float) * n_user_ * n_distribution_parameter_));

            error_l_ = std::make_unique<float[]>(n_user_);
            index_stream.read((char *) error_l_.get(),
                              (int64_t) (sizeof(float) * n_user_));

            index_stream.close();
        }

        uint64_t IndexSizeByte() const {
            const uint64_t sample_rank_size = sizeof(int) * n_sample_rank_;
            const uint64_t para_size = sizeof(float) * n_user_ * (n_predict_parameter_ + n_distribution_parameter_);
            const uint64_t error_size = sizeof(int) * n_user_;
            return sample_rank_size + para_size + error_size;
        }

    };
}
#endif //REVERSE_KRANKS_NORMALLINEARREGRESSIONALGLIB_HPP
