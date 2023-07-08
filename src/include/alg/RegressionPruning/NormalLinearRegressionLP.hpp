//
// Created by BianZheng on 2022/10/29.
//

#ifndef REVERSE_K_RANKS_MINMAXHEADLINEARREGRESSION_HPP
#define REVERSE_K_RANKS_MINMAXHEADLINEARREGRESSION_HPP

#include "alg/RegressionPruning/BaseLinearRegression.hpp"
#include "struct/DistancePair.hpp"
#include "util/MathUtil.hpp"
#include "linear_program/sdlp/sdlp.hpp"
#include "util/TimeMemory.hpp"

#include <iostream>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class NormalLinearRegressionLP : public BaseLinearRegression {

        std::string regression_index_name_;
        size_t n_data_item_, n_user_;
        static constexpr int n_predict_parameter_ = 2; // (a, b) for linear estimation
        static constexpr int n_distribution_parameter_ = 2; // mu, sigma
        static constexpr float sqrt_2_ = sqrt(2.0);
        int n_sample_rank_;
        int n_thread_;
        std::unique_ptr<int[]> sample_rank_l_; // n_sample_rank
        std::unique_ptr<float[]> predict_para_l_; // n_user_ * n_predict_parameter
        std::unique_ptr<float[]> distribution_para_l_; // n_user_ * n_distribution_parameter
        std::unique_ptr<float[]> error_l_; //n_user_

        std::unique_ptr<float[]> sample_score_extreme_l_; //n_user_ * 2

        int n_report_user_;

        //used for loading
        std::vector<float *> batch_cache_A_l_; // for each A, (4 * (n_sample_rank - 1)) * (n_predict_parameter_ + 1), constraint matrix
        float *preprocess_cache_b_; // (4 * (n_sample_rank - 1)), constraint bound
        float *preprocess_cache_c_; // 3, objective coefficients
        int *preprocess_b_random_permutation_; // 4 * n_sample_rank_
    public:

        inline NormalLinearRegressionLP() {}

        inline NormalLinearRegressionLP(const int &n_data_item, const int &n_user,
                                        const std::string &regression_index_name) {
            this->regression_index_name_ = regression_index_name;
            this->n_data_item_ = n_data_item;
            this->n_user_ = n_user;
            this->predict_para_l_ = std::make_unique<float[]>(n_user * n_predict_parameter_);
            this->distribution_para_l_ = std::make_unique<float[]>(n_user * n_distribution_parameter_);
            this->error_l_ = std::make_unique<float[]>(n_user);
            this->sample_score_extreme_l_ = std::make_unique<float[]>(n_user_ * 2);

            this->n_report_user_ = n_user / 10;
            static_assert(n_predict_parameter_ == 2 && n_distribution_parameter_ == 2);
        }

        inline NormalLinearRegressionLP(const char *index_basic_dir, const char *dataset_name,
                                        const std::string &regression_index_name,
                                        const size_t &n_sample, const size_t &n_sample_query,
                                        const size_t &sample_topk, const int &n_thread = omp_get_num_procs()) {
            LoadIndex(index_basic_dir, dataset_name, regression_index_name,
                      n_sample, n_sample_query, sample_topk);
            this->n_thread_ = n_thread;
            this->regression_index_name_ = regression_index_name;
        }

        void StartPreprocess(const int *sample_rank_l, const int &n_sample_rank) override {
            this->n_sample_rank_ = n_sample_rank;
            this->sample_rank_l_ = std::make_unique<int[]>(n_sample_rank);
            std::memcpy(this->sample_rank_l_.get(), sample_rank_l, n_sample_rank * sizeof(int));

            this->batch_cache_A_l_.resize(batch_n_user_);
            for (int batchID = 0; batchID < batch_n_user_; batchID++) {
                batch_cache_A_l_[batchID] = new float[(4 * (n_sample_rank - 1)) * (n_predict_parameter_ + 1)];
            }

            this->preprocess_cache_b_ = new float[4 * (n_sample_rank - 1)];
            this->preprocess_cache_c_ = new float[3];
#pragma omp parallel for default(none) shared(sample_rank_l)
            for (int sampleID = 0; sampleID < n_sample_rank_ - 1; sampleID++) {

                preprocess_cache_b_[4 * sampleID] = -sampleID - 1;
                preprocess_cache_b_[4 * sampleID + 1] = sampleID + 1;
                preprocess_cache_b_[4 * sampleID + 2] = -sampleID - 1;
                preprocess_cache_b_[4 * sampleID + 3] = sampleID + 1;
            }

            preprocess_cache_c_[0] = 0;
            preprocess_cache_c_[1] = 0;
            preprocess_cache_c_[2] = 1;

            this->preprocess_b_random_permutation_ = new int[4 * (n_sample_rank - 1)];
            std::iota(preprocess_b_random_permutation_, preprocess_b_random_permutation_ + 4 * (n_sample_rank - 1), 0);
            std::shuffle(preprocess_b_random_permutation_, preprocess_b_random_permutation_ + 4 * (n_sample_rank - 1),
                         std::mt19937(std::random_device()()));

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

        void AssignCache(float *preprocess_cache_A, const float *sampleIP_l, const int &userID) {
            //compute average, std
            const float mu = ComputeAverage(sampleIP_l);
            const float sigma = ComputeStd(sampleIP_l, mu);
            distribution_para_l_[userID * n_distribution_parameter_] = mu;
            distribution_para_l_[userID * n_distribution_parameter_ + 1] = sigma;

            sample_score_extreme_l_[userID * 2] = sampleIP_l[0];
            sample_score_extreme_l_[userID * 2 + 1] = sampleIP_l[n_sample_rank_ - 1];

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

        }

        void LinearProgram(const float *preprocess_cache_A, const int &userID) {
            using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, n_predict_parameter_ + 1, Eigen::RowMajor>;
            Eigen::Map<const RowMatrixXf> A(preprocess_cache_A, 4 * (n_sample_rank_ - 1),
                                            (int64_t) n_predict_parameter_ + 1);

            Eigen::Map<const Eigen::VectorXf> b(preprocess_cache_b_, 4 * (n_sample_rank_ - 1));

            Eigen::Map<const Eigen::Matrix<float, n_predict_parameter_ + 1, 1>> c(preprocess_cache_c_,
                                                                                  n_predict_parameter_ + 1);
            Eigen::Matrix<float, n_predict_parameter_ + 1, 1> x;

//            sdlp::linprog<3>(c, A, b, x);
            const float max_val = sdlp::linprog<3>(c, A, b, preprocess_b_random_permutation_, x);
            float error = x[2];
            const float tmp_a = x[0];
            const float tmp_b = x[1];
            if (tmp_a > 0 || tmp_b < 0) {
                error = -1;
            } else if (0 < error && error < (float) n_sample_rank_) {
                error += 0.01f;
            } else {
                error = -1;
            }
            error_l_[userID] = error;

            assert(x.rows() == n_predict_parameter_ + 1);

            //assign parameter
            for (int paraID = 0; paraID < n_predict_parameter_; paraID++) {
                predict_para_l_[userID * n_predict_parameter_ + paraID] = x[paraID];
            }

        }

        void CalcError(const float *sampleIP_l, const int &userID, const float &mu, const float &sigma) {
            //assign error
            float error = -1;
            for (int sampleID = 0; sampleID < n_sample_rank_ - 1; sampleID++) {
                const float input_x = sampleIP_l[sampleID];
                const float normal_num2 = (input_x - mu) / sigma;
                const float cdf = CDFPhi(normal_num2);

                const float input_next_x = sampleIP_l[sampleID + 1];
                const float normal_num = (input_next_x - mu) / sigma;
                const float next_cdf = CDFPhi(normal_num);

                const float tmp_a = predict_para_l_[userID * n_predict_parameter_];
                const float tmp_b = predict_para_l_[userID * n_predict_parameter_ + 1];
                const float pred_rank1 = cdf * tmp_a + tmp_b;
                const float pred_rank2 = next_cdf * tmp_a + tmp_b;
                const float tmp_error1 = std::abs((float) (sampleID + 1) - pred_rank1);
                const float tmp_error2 = std::abs((float) (sampleID + 1) - pred_rank2);
                const float tmp_error = std::max(tmp_error1, tmp_error2) + 0.01f;
                error = std::max(tmp_error, error);
            }

            error = std::min(error, (float) n_sample_rank_);
            error += 0.01f;
            error_l_[userID] = error;

            assert(error > 0);
            assert(-1 < error && error < n_sample_rank_ + 0.1);
        }

        void BatchLoopPreprocess(const std::vector<const float *> &sampleIP_l_l,
                                 const int &start_userID, const int &n_proc_user,
                                 double &assign_cache_time, double &linear_program_time,
                                 double &calc_error_time) override {
            TimeRecord record;
            record.reset();

#pragma omp parallel for default(none) shared(start_userID, n_proc_user, sampleIP_l_l, n_sample_rank_, distribution_para_l_) num_threads(omp_get_max_threads())
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const float *sampleIP_l = sampleIP_l_l[proc_userID];
                float *preprocess_cache_A = batch_cache_A_l_[proc_userID];
                const int userID = proc_userID + start_userID;
                AssignCache(preprocess_cache_A, sampleIP_l, userID);
            }
            assign_cache_time = record.get_elapsed_time_second();

            record.reset();
#pragma omp parallel for default(none) shared(n_proc_user, start_userID, sampleIP_l_l, n_sample_rank_, predict_para_l_, batch_cache_A_l_, preprocess_cache_b_, preprocess_cache_c_, preprocess_b_random_permutation_) num_threads(omp_get_max_threads())
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const float *preprocess_cache_A = batch_cache_A_l_[proc_userID];
                const float *sampleIP_l = sampleIP_l_l[proc_userID];
                const int userID = proc_userID + start_userID;
                LinearProgram(preprocess_cache_A, userID);
            }
            linear_program_time = record.get_elapsed_time_second();

            record.reset();
#pragma omp parallel for default(none) shared(sampleIP_l_l, n_proc_user, n_sample_rank_, predict_para_l_, error_l_, start_userID, distribution_para_l_)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const int userID = proc_userID + start_userID;
                if (error_l_[userID] < 0 || error_l_[userID] > n_sample_rank_ - 1e-3) {
                    const float *sampleIP_l = sampleIP_l_l[proc_userID];

                    const float mu = distribution_para_l_[userID * n_distribution_parameter_];
                    const float sigma = distribution_para_l_[userID * n_distribution_parameter_ + 1];

                    const float first_sampleIP = CDFPhi((sampleIP_l_l[proc_userID][0] - mu) / sigma);
                    const float last_sampleIP = CDFPhi((sampleIP_l_l[proc_userID][n_sample_rank_ - 1] - mu) / sigma);

                    const float a_value = (float) (2.0f - (float) n_sample_rank_) / (
                            first_sampleIP - last_sampleIP);
                    predict_para_l_[userID * n_predict_parameter_] = a_value;
                    const float b_value =
                            ((float) (n_sample_rank_) * first_sampleIP - last_sampleIP) /
                            (first_sampleIP - last_sampleIP);
                    predict_para_l_[userID * n_predict_parameter_ + 1] = b_value;
                    assert(predict_para_l_[userID * n_predict_parameter_] < 0);
                    assert(predict_para_l_[userID * n_predict_parameter_ + 1] > 0);

                    CalcError(sampleIP_l, userID, mu, sigma);
                }

            }
            calc_error_time = record.get_elapsed_time_second();

            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const int userID = proc_userID + start_userID;
                if (userID % n_report_user_ == 0) {
                    spdlog::info("userID {}, progress {:.3f}, a {:.3f}, b {:.3f}, error {:.3f}",
                                 userID, 1.0 * userID / (float) n_user_,
                                 predict_para_l_[userID * n_predict_parameter_],
                                 predict_para_l_[userID * n_predict_parameter_ + 1], error_l_[userID]);
                }
            }

        }

        void FinishPreprocess() override {
            for (int batchID = 0; batchID < batch_n_user_; batchID++) {
                delete[] batch_cache_A_l_[batchID];
                batch_cache_A_l_[batchID] = nullptr;
            }
            batch_cache_A_l_.clear();
            delete[] preprocess_cache_b_;
            delete[] preprocess_cache_c_;
            delete[]preprocess_b_random_permutation_;
            preprocess_cache_b_ = nullptr;
            preprocess_cache_c_ = nullptr;
            preprocess_b_random_permutation_ = nullptr;
        }

        inline void
        ComputeRankBound(const float &queryIP_lb, const float &queryIP_ub, const int &userID,
                         int &rank_lb, int &rank_ub) const {

            const size_t pred_pos = userID * n_predict_parameter_;
            const size_t distribution_pos = userID * n_distribution_parameter_;
            const float mu = distribution_para_l_[distribution_pos];
            const float sigma = distribution_para_l_[distribution_pos + 1];

            const float sampleIP_max = sample_score_extreme_l_[userID * 2];
            const float sampleIP_min = sample_score_extreme_l_[userID * 2 + 1];
            if (queryIP_lb > sampleIP_max) {
                rank_lb = sample_rank_l_[0];
            } else if (queryIP_lb < sampleIP_min) {
                rank_lb = (int) n_data_item_;
            } else if (error_l_[userID] >= n_sample_rank_ - 1e-3) {
                rank_lb = (int) n_data_item_;

            } else {
                const float normalize_x_lb = (queryIP_lb - mu) / sigma;
                const float input_x_lb = CDFPhi(normalize_x_lb);
                const float pred_rank_lb = input_x_lb * predict_para_l_[pred_pos] + predict_para_l_[pred_pos + 1];
                const int64_t pred_sample_rank_lb = std::ceil(pred_rank_lb + error_l_[userID]);
                if (pred_sample_rank_lb >= n_sample_rank_) {
                    rank_lb = (int) n_data_item_;
                } else if (pred_sample_rank_lb < 0) {
                    rank_lb = sample_rank_l_[0];
                } else {
                    rank_lb = sample_rank_l_[pred_sample_rank_lb];
                }

            }

            if (queryIP_ub > sampleIP_max) {
                rank_ub = 0;
            } else if (queryIP_ub < sampleIP_min) {
                rank_ub = (int) sample_rank_l_[n_sample_rank_ - 1];
            } else if (error_l_[userID] >= n_sample_rank_ - 1e-3) {
                rank_ub = 0;

            } else {
                const float normalize_x_ub = (queryIP_ub - mu) / sigma;
                const float input_x_ub = CDFPhi(normalize_x_ub);
                const float pred_rank_ub = input_x_ub * predict_para_l_[pred_pos] + predict_para_l_[pred_pos + 1];
                const int64_t pred_sample_rank_ub = std::floor(pred_rank_ub - error_l_[userID]);
                if (pred_sample_rank_ub >= n_sample_rank_) {
                    rank_ub = (int) n_data_item_;
                } else if (pred_sample_rank_ub < 0) {
                    rank_ub = 0;
                } else {
                    rank_ub = sample_rank_l_[pred_sample_rank_ub];
                }

            }

            if (rank_ub > rank_lb) {
//            if (userID % 500 == 0) {
                spdlog::error(
                        "userID {}, qIP_ub_tmp_upper_rank {}, qIP_lb_tmp_lower_rank {}, queryIP_lb {}, queryIP_ub {}, error {}, input_x_lb {:.2f}, input_x_ub {:.2f}",
                        userID, rank_ub, rank_lb, queryIP_lb, queryIP_ub, error_l_[userID],
                        CDFPhi((queryIP_lb - mu) / sigma), CDFPhi((queryIP_ub - mu) / sigma));
                spdlog::error("userID {} sampleIP_max {}, sampleIP_min {}",
                              userID, sample_score_extreme_l_[userID * 2], sample_score_extreme_l_[userID * 2 + 1]);
                spdlog::error("userID {}, para_a {}, para_b {}",
                              userID, predict_para_l_[pred_pos], predict_para_l_[pred_pos + 1]);
                spdlog::error("userID {} pred_pos {}, distribution_pos {}",
                              userID, pred_pos, distribution_pos);
                spdlog::error("userID {}, mu {}, sigma {}",
                              userID, mu, sigma);
            }

            assert(0 <= error_l_[userID] && error_l_[userID] <= n_sample_rank_ + 0.1);
        }

        void RankBound(const std::vector<std::pair<float, float>> &queryIP_l,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l, const int &queryID) const {

#pragma omp parallel for default(none) shared(queryIP_l, rank_lb_l, rank_ub_l, queryID) num_threads(n_thread_)
            for (int userID = 0; userID < n_user_; userID++) {
                const float queryIP_lb = queryIP_l[userID].first;
                const float queryIP_ub = queryIP_l[userID].second;
                assert(queryIP_lb <= queryIP_ub);
                int qIP_lb_tmp_lower_rank, qIP_ub_tmp_upper_rank;

                ComputeRankBound(queryIP_lb, queryIP_ub, userID,
                                 qIP_lb_tmp_lower_rank, qIP_ub_tmp_upper_rank);
                assert(qIP_ub_tmp_upper_rank <= qIP_lb_tmp_lower_rank);

                rank_lb_l[userID] = qIP_lb_tmp_lower_rank;
                rank_ub_l[userID] = qIP_ub_tmp_upper_rank;
            }
        }

        void SaveIndex(const char *index_basic_dir, const char *dataset_name, const size_t &n_sample_query,
                       const size_t &sample_topk) override {
            char index_path[256];
            sprintf(index_path,
                    "%s/memory_index/NormalLinearRegressionLP-%s-%s-n_sample_%d-n_sample_query_%ld-sample_topk_%ld.index",
                    index_basic_dir, regression_index_name_.c_str(), dataset_name, n_sample_rank_, n_sample_query,
                    sample_topk);


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
            out_stream_.write((char *) sample_score_extreme_l_.get(), (int64_t) (n_user_ * 2 * sizeof(float)));

            out_stream_.close();
        }

        void LoadIndex(const char *index_basic_dir, const char *dataset_name, const std::string &regression_index_name,
                       const size_t &n_sample, const size_t &n_sample_query, const size_t &sample_topk) {
            char index_path[256];
            sprintf(index_path,
                    "%s/memory_index/NormalLinearRegressionLP-%s-%s-n_sample_%ld-n_sample_query_%ld-sample_topk_%ld.index",
                    index_basic_dir, regression_index_name.c_str(), dataset_name, n_sample, n_sample_query,
                    sample_topk);
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

            sample_score_extreme_l_ = std::make_unique<float[]>(n_user_ * 2);
            index_stream.read((char *) sample_score_extreme_l_.get(),
                              (int64_t) (sizeof(float) * n_user_ * 2));

            index_stream.close();
        }

        uint64_t IndexSizeByte() const {
            const uint64_t sample_rank_size = sizeof(int) * n_sample_rank_;
            const uint64_t para_size = sizeof(float) * n_user_ * (n_predict_parameter_ + n_distribution_parameter_);
            const uint64_t error_size = sizeof(int) * n_user_;
            const uint64_t sample_score_size = sizeof(float) * 2 * n_user_;
            return sample_rank_size + para_size + error_size + sample_score_size;
        }

    };
}
#endif //REVERSE_K_RANKS_MINMAXHEADLINEARREGRESSION_HPP
