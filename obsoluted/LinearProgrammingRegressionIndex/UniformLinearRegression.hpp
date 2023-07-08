//
// Created by BianZheng on 2022/12/3.
//

#ifndef REVERSE_K_RANKS_UNIFORMLINEARREGRESSION_HPP
#define REVERSE_K_RANKS_UNIFORMLINEARREGRESSION_HPP

#include "struct/DistancePair.hpp"
#include "util/MathUtil.hpp"
#include "sdlp/sdlp.hpp"
#include "alg/RegressionPruning/BaseLinearRegression.hpp"

#include <iostream>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class UniformLinearRegression : public BaseLinearRegression {

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
        std::vector<float *> batch_cache_A_l_; // for each A, (4 * n_sample_rank) * (n_predict_parameter_ + 1), constraint matrix
        float *preprocess_cache_b_; // (4 * n_sample_rank), constraint bound
        float *preprocess_cache_c_; // 3, objective coefficients
    public:

        inline UniformLinearRegression() {}

        inline UniformLinearRegression(const int &n_data_item, const int &n_user, const std::string &method_name) {
            method_name_ = method_name;
            this->n_data_item_ = n_data_item;
            this->n_user_ = n_user;
            this->predict_para_l_ = std::make_unique<float[]>(n_user * n_predict_parameter_);
            this->distribution_para_l_ = std::make_unique<float[]>(n_user * n_distribution_parameter_);
            this->error_l_ = std::make_unique<float[]>(n_user);
            this->n_batch_ = (int) n_user_ / batch_n_user_ + (n_user_ % batch_n_user_ == 0 ? 0 : 1);
            static_assert(n_predict_parameter_ == 2 && n_distribution_parameter_ == 2);
        }

        inline UniformLinearRegression(const char *index_basic_dir, const char *dataset_name,
                                       const std::string &method_name,
                                       const size_t &n_sample, const size_t &n_sample_query,
                                       const size_t &sample_topk) {
            LoadIndex(index_basic_dir, dataset_name, method_name, n_sample, n_sample_query, sample_topk);
            this->method_name_ = method_name;
            this->n_batch_ = (int) n_user_ / batch_n_user_ + (n_user_ % batch_n_user_ == 0 ? 0 : 1);
        }

        void StartPreprocess(const int *sample_rank_l, const int &n_sample_rank) override {
            this->n_sample_rank_ = n_sample_rank;
            this->sample_rank_l_ = std::make_unique<int[]>(n_sample_rank);

            this->batch_cache_A_l_.resize(batch_n_user_);
            for (int batchID = 0; batchID < batch_n_user_; batchID++) {
                batch_cache_A_l_[batchID] = new float[(4 * n_sample_rank) * (n_predict_parameter_ + 1)];
            }

            this->preprocess_cache_b_ = new float[4 * n_sample_rank];
            this->preprocess_cache_c_ = new float[3];
            for (int sampleID = 0; sampleID < n_sample_rank_ - 1; sampleID++) {
                sample_rank_l_[sampleID] = sample_rank_l[sampleID];

                preprocess_cache_b_[4 * sampleID] = -sampleID - 1;
                preprocess_cache_b_[4 * sampleID + 1] = sampleID + 1;
                preprocess_cache_b_[4 * sampleID + 2] = -sampleID - 1;
                preprocess_cache_b_[4 * sampleID + 3] = sampleID + 1;
            }
            preprocess_cache_b_[4 * (n_sample_rank_ - 1) + 0] = -n_sample_rank_;
            preprocess_cache_b_[4 * (n_sample_rank_ - 1) + 1] = n_sample_rank_;
            preprocess_cache_b_[4 * (n_sample_rank_ - 1) + 2] = 0;
            preprocess_cache_b_[4 * (n_sample_rank_ - 1) + 3] = 0;
            sample_rank_l_[n_sample_rank_ - 1] = sample_rank_l[n_sample_rank_ - 1];

            preprocess_cache_c_[0] = 0;
            preprocess_cache_c_[1] = 0;
            preprocess_cache_c_[2] = 1;

        }

        void AssignCache(float *preprocess_cache_A, const float *sampleIP_l, const int &userID) {
            const int high_score_quantile_idx = (int) (n_sample_rank_ * 0.05);
            const int low_score_quantile_idx = (int) (n_sample_rank_ * 0.95);
            const float high_score = sampleIP_l[high_score_quantile_idx];
            const float low_score = sampleIP_l[low_score_quantile_idx];
            const float score_diff = high_score - low_score;
            distribution_para_l_[userID * n_distribution_parameter_] = low_score;
            distribution_para_l_[userID * n_distribution_parameter_ + 1] = score_diff;

//#pragma omp parallel for default(none) shared(sampleIP_l, low_score, score_diff)
            for (int sampleID = 0; sampleID < n_sample_rank_ - 1; sampleID++) {
                const float input_next_x = (sampleIP_l[sampleID + 1] - low_score) / score_diff;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1)] = -input_next_x;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 1] = -1;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 2] = -1;

                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 3] = input_next_x;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 4] = 1;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 5] = -1;

                const float input_x = (sampleIP_l[sampleID] - low_score) / score_diff;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 6] = -input_x;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 7] = -1;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 8] = -1;

                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 9] = input_x;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 10] = 1;
                preprocess_cache_A[(4 * sampleID) * (n_predict_parameter_ + 1) + 11] = -1;

            }

            {
                const int sampleID = n_sample_rank_ - 1;
                const float input_x = (sampleIP_l[sampleID] - low_score) / score_diff;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 0] = -input_x;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 1] = -1;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 2] = -1;

                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 3] = input_x;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 4] = 1;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 5] = -1;
            }

            {
                const int sampleID = 0;
                const float input_x = (sampleIP_l[sampleID] - low_score) / score_diff;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 6] = -input_x;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 7] = -1;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 8] = -1;

                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 9] = input_x;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 10] = 1;
                preprocess_cache_A[(4 * (n_sample_rank_ - 1)) * (n_predict_parameter_ + 1) + 11] = -1;
            }
        }

        void LinearProgram(const float *preprocess_cache_A, const int &userID) {
            using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, n_predict_parameter_ + 1, Eigen::RowMajor>;
            Eigen::Map<const RowMatrixXf> A(preprocess_cache_A, 4 * n_sample_rank_, (int64_t)
                                                                                            n_predict_parameter_ + 1);

            Eigen::Map<const Eigen::VectorXf> b(preprocess_cache_b_, 4 * n_sample_rank_);

            Eigen::Map<const Eigen::Matrix<float, n_predict_parameter_ + 1, 1>> c(preprocess_cache_c_,
                                                                                  n_predict_parameter_ + 1);
            Eigen::Matrix<float, n_predict_parameter_ + 1, 1> x;

//            sdlp::linprog<3>(c, A, b, x);
            float error = sdlp::linprog<3>(c, A, b, x);
            if (error < 0 || error > (float) n_sample_rank_) {
                error = (float) n_sample_rank_;
            }
            error_l_[userID] = error;

            assert(x.rows() == n_predict_parameter_ + 1);
//            printf("res rows %ld, cols %ld\n", res.rows(), res.cols());
//            printf("res [0]: %.3f, [1]: %.3f\n", res[0], res[1]);

            //assign parameter
            for (int paraID = 0; paraID < n_predict_parameter_; paraID++) {
                predict_para_l_[userID * n_predict_parameter_ + paraID] = (float) x[paraID];
            }
        }

        void CalcError(const float *sampleIP_l, const int &userID, const float &mu, const float &sigma) {
            //assign error

            const float low_score = distribution_para_l_[userID * n_distribution_parameter_];
            const float score_diff = distribution_para_l_[userID * n_distribution_parameter_ + 1];

            float error = -1;
            for (int sampleID = 0; sampleID < n_sample_rank_ - 1; sampleID++) {
                const float input_x = (sampleIP_l[sampleID] - low_score) / score_diff;
                const float input_next_x = (sampleIP_l[sampleID + 1] - low_score) / score_diff;

                const float tmp_a = predict_para_l_[userID * n_predict_parameter_];
                const float tmp_b = predict_para_l_[userID * n_predict_parameter_ + 1];
                const float pred_rank1 = input_next_x * tmp_a + tmp_b;
                const float pred_rank2 = input_x * tmp_a + tmp_b;
                const int real_rank = sampleID;
                const float tmp_error1 = std::abs((sampleID + 1) - pred_rank1);
                const float tmp_error2 = std::abs((sampleID + 1) - pred_rank2);
                const float tmp_error = std::max(tmp_error1, tmp_error2) + 0.1;
                error = std::max(tmp_error, error);
            }

            {
                const int sampleID = n_sample_rank_ - 1;
                const float input_x = (sampleIP_l[sampleID] - low_score) / score_diff;

                const float tmp_a = predict_para_l_[userID * n_predict_parameter_];
                const float tmp_b = predict_para_l_[userID * n_predict_parameter_ + 1];
                const float pred_rank2 = input_x * tmp_a + tmp_b;
                const float tmp_error = std::abs((sampleID + 1) - pred_rank2) + 0.01;
                error = std::max(tmp_error, error);
            }

            {
                const int sampleID = 0;
                const float input_x = (sampleIP_l[sampleID] - low_score) / score_diff;

                const float tmp_a = predict_para_l_[userID * n_predict_parameter_];
                const float tmp_b = predict_para_l_[userID * n_predict_parameter_ + 1];
                const float pred_rank2 = input_x * tmp_a + tmp_b;
                const float tmp_error = std::abs(sampleID - pred_rank2) + 0.01;
                error = std::max(tmp_error, error);
            }

            error = std::min(error, (float) n_sample_rank_);
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

#pragma omp parallel for default(none) shared(start_userID, n_proc_user, sampleIP_l_l, n_sample_rank_, distribution_para_l_)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const float *sampleIP_l = sampleIP_l_l[proc_userID];
                float *preprocess_cache_A = batch_cache_A_l_[proc_userID];
                const int &userID = proc_userID + start_userID;
                AssignCache(preprocess_cache_A, sampleIP_l, userID);
            }
            assign_cache_time = record.get_elapsed_time_second();

            record.reset();
#pragma omp parallel for default(none) shared(n_proc_user, start_userID, n_sample_rank_, preprocess_cache_b_, preprocess_cache_c_, predict_para_l_)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const float *preprocess_cache_A = batch_cache_A_l_[proc_userID];
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
            delete[] preprocess_cache_b_;
            delete[] preprocess_cache_c_;
            preprocess_cache_b_ = nullptr;
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
            const float low_score = distribution_para_l_[distribution_pos];
            const float score_diff = distribution_para_l_[distribution_pos + 1];
            const float input_x = (queryIP - low_score) / score_diff;

            const size_t pred_pos = userID * n_predict_parameter_;
            const float pred_rank = input_x * predict_para_l_[pred_pos] + predict_para_l_[pred_pos + 1];
            const int pred_sample_rank_lb = std::ceil(pred_rank + error_l_[userID]);
            const int pred_sample_rank_ub = std::floor(pred_rank - error_l_[userID]);

            if (pred_sample_rank_lb >= n_sample_rank_) {
                rank_lb = n_data_item_;
            } else if (pred_sample_rank_lb < 0) {
                rank_lb = sample_rank_l_[0];
            } else {
                rank_lb = sample_rank_l_[pred_sample_rank_lb];
            }

            if (pred_sample_rank_ub >= n_sample_rank_) {
                rank_ub = n_data_item_;
            } else if (pred_sample_rank_ub < 0) {
                rank_ub = 0;
            } else {
                rank_ub = sample_rank_l_[pred_sample_rank_ub];
            }

        }

//        void RankBound(const std::vector<double> &queryIP_l,
//                       const std::vector<char> &prune_l, const std::vector<char> &result_l,
//                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l) const {
//            assert(queryIP_l.size() == n_user_);
//            assert(prune_l.size() == n_user_);
//            assert(result_l.size() == n_user_);
//            assert(rank_lb_l.size() == n_user_);
//            assert(rank_ub_l.size() == n_user_);
//            for (int userID = 0; userID < n_user_; userID++) {
//                if (prune_l[userID] || result_l[userID]) {
//                    continue;
//                }
//                int lower_rank = rank_lb_l[userID];
//                int upper_rank = rank_ub_l[userID];
//                assert(upper_rank <= lower_rank);
//                double queryIP = queryIP_l[userID];
//
//                ComputeRankBound(queryIP, userID,
//                                 lower_rank, upper_rank);
//
//                rank_lb_l[userID] = lower_rank;
//                rank_ub_l[userID] = upper_rank;
//            }
//        }

        void RankBound(const std::vector<std::pair<float, float>> &queryIP_l,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l, const int &queryID) const {

#pragma omp parallel for default(none) shared(queryIP_l, queryID, rank_lb_l, rank_ub_l) num_threads(omp_get_num_procs()) schedule(dynamic, 1000)
            for (int batchID = 0; batchID < n_batch_; batchID++) {
                const int start_userID = batchID * batch_n_user_;
                const int end_userID = std::min((int) n_user_, (batchID + 1) * batch_n_user_);
                for (int userID = start_userID; userID < end_userID; userID++) {
                    const float queryIP_lb = queryIP_l[userID].first;
                    int qIP_lb_tmp_lower_rank, qIP_lb_tmp_upper_rank;

                    ComputeRankBound(queryIP_lb, userID,
                                     qIP_lb_tmp_lower_rank, qIP_lb_tmp_upper_rank, queryID);

                    const float queryIP_ub = queryIP_l[userID].second;
                    int qIP_ub_tmp_lower_rank, qIP_ub_tmp_upper_rank;
                    ComputeRankBound(queryIP_ub, userID,
                                     qIP_ub_tmp_lower_rank, qIP_ub_tmp_upper_rank, queryID);

                    rank_lb_l[userID] = qIP_lb_tmp_lower_rank;
                    rank_ub_l[userID] = qIP_ub_tmp_upper_rank;

                    assert(qIP_lb_tmp_upper_rank <= qIP_lb_tmp_lower_rank);
                    assert(qIP_ub_tmp_upper_rank <= qIP_ub_tmp_lower_rank);
                    assert(qIP_ub_tmp_upper_rank <= qIP_lb_tmp_lower_rank);
                }
            }
        }

        void SaveIndex(const char *index_basic_dir, const char *dataset_name, const size_t &n_sample_query,
                       const size_t &sample_topk) override {
            char index_path[256];
            sprintf(index_path,
                    "%s/memory_index/UniformLinearRegression-%s-%s-n_sample_%d-n_sample_query_%ld-sample_topk_%ld.index",
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
                    "%s/memory_index/UniformLinearRegression-%s-%s-n_sample_%ld-n_sample_query_%ld-sample_topk_%ld.index",
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
            const uint64_t error_size = sizeof(float) * n_user_;
            return sample_rank_size + para_size + error_size;
        }

    };
}
#endif //REVERSE_K_RANKS_UNIFORMLINEARREGRESSION_HPP
