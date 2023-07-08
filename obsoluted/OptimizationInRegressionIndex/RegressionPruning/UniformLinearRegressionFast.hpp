//
// Created by bianzheng on 2023/4/14.
//

#ifndef REVERSE_KRANKS_UNIFORMLINEARREGRESSIONFAST_HPP
#define REVERSE_KRANKS_UNIFORMLINEARREGRESSIONFAST_HPP

#include "BaseLinearRegression.hpp"
#include "struct/DistancePair.hpp"
#include "util/MathUtil.hpp"


#include <iostream>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <spdlog/spdlog.h>

#include "alg/RegressionPruning/GradientDescent/GDUniformFastCPU.hpp"


namespace ReverseMIPS {

    class UniformLinearRegressionFast : public BaseLinearRegression {

        std::string method_name_;
        size_t n_data_item_, n_user_;
        int epoch_rp_;
        static constexpr int n_predict_parameter_ = 2; // (a, b) for linear estimation
        static constexpr int n_distribution_parameter_ = 2; // low_value, high_value
        int n_sample_rank_;
        std::unique_ptr<int[]> sample_rank_l_; // n_sample_rank
        std::unique_ptr<float[]> predict_para_l_; // n_user_ * n_predict_parameter
        std::unique_ptr<float[]> distribution_para_l_; // n_user_ * n_distribution_parameter
        std::unique_ptr<float[]> error_l_; //n_user_

        //for build index
        GDUniformFastCPU gd_normal_fast_cpu_;

        int n_batch_;
    public:

        inline UniformLinearRegressionFast() {}

        inline UniformLinearRegressionFast(const int &n_data_item, const int &n_user, const int &epoch_rp,
                                           const std::string &method_name) {
            this->method_name_ = method_name;
            this->n_data_item_ = n_data_item;
            this->n_user_ = n_user;
            this->epoch_rp_ = epoch_rp;
            this->predict_para_l_ = std::make_unique<float[]>(n_user * n_predict_parameter_);
            this->distribution_para_l_ = std::make_unique<float[]>(n_user * n_distribution_parameter_);
            this->error_l_ = std::make_unique<float[]>(n_user);
            this->n_batch_ = (int) n_user_ / batch_n_user_ + (n_user_ % batch_n_user_ == 0 ? 0 : 1);
            static_assert(n_predict_parameter_ == 2 && n_distribution_parameter_ == 2);
        }

        inline UniformLinearRegressionFast(const char *index_basic_dir, const char *dataset_name,
                                           const std::string &method_name,
                                           const size_t &n_sample,
                                           const size_t &n_sample_query, const size_t &sample_topk,
                                           const int &epoch_rp) {
            LoadIndex(index_basic_dir, dataset_name, method_name,
                      n_sample, n_sample_query, sample_topk, epoch_rp);
            this->method_name_ = method_name;
            this->epoch_rp_ = epoch_rp;
            this->n_batch_ = (int) n_user_ / batch_n_user_ + (n_user_ % batch_n_user_ == 0 ? 0 : 1);
        }

        void StartPreprocess(const int *sample_rank_l, const int &n_sample_rank) override {
            this->n_sample_rank_ = n_sample_rank;
            this->sample_rank_l_ = std::make_unique<int[]>(n_sample_rank);

            for (int sampleID = 0; sampleID < n_sample_rank_ - 1; sampleID++) {
                sample_rank_l_[sampleID] = sample_rank_l[sampleID];
            }
            sample_rank_l_[n_sample_rank_ - 1] = sample_rank_l[n_sample_rank_ - 1];

            gd_normal_fast_cpu_ = GDUniformFastCPU(n_sample_rank_, batch_n_user_);

        }

        void BatchLoopPreprocess(const std::vector<const float *> &sampleIP_l_l,
                                 const int &start_userID, const int &n_proc_user,
                                 double &assign_cache_time, double &linear_program_time,
                                 double &calc_error_time) override {

            TimeRecord record;
            record.reset();

            std::vector<float> para_a_value_l(batch_n_user_);
            std::vector<float> para_b_value_l(batch_n_user_);

            std::vector<float> para_a_gradient_l(batch_n_user_);
            std::vector<float> para_b_gradient_l(batch_n_user_);
            gd_normal_fast_cpu_.Load(sampleIP_l_l, start_userID, n_proc_user);
            gd_normal_fast_cpu_.CalcDistributionPara(start_userID, n_proc_user,
                                                     n_distribution_parameter_, distribution_para_l_.get());

            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                //compute average, std
                const int userID = proc_userID + start_userID;
                const float low_value = distribution_para_l_[userID * n_distribution_parameter_];
                const float high_value = distribution_para_l_[userID * n_distribution_parameter_ + 1];

                const float first_sampleIP =
                        (sampleIP_l_l[proc_userID][0] - low_value) / (high_value - low_value);
                const float last_sampleIP =
                        (sampleIP_l_l[proc_userID][n_sample_rank_ - 1] - low_value) / (high_value - low_value);

                para_a_value_l[proc_userID] = (float) (2 - n_sample_rank_) / (
                        first_sampleIP - last_sampleIP);
                para_b_value_l[proc_userID] =
                        ((float) (n_sample_rank_) * first_sampleIP -
                         last_sampleIP) /
                        (first_sampleIP - last_sampleIP);
                assert(para_a_value_l[proc_userID] < 0);
                assert(para_b_value_l[proc_userID] > 0);

            }

            gd_normal_fast_cpu_.Precompute(distribution_para_l_.get(), n_distribution_parameter_,
                                           start_userID, n_proc_user);


            assign_cache_time = record.get_elapsed_time_second();

            record.reset();

            for (int epochID = 0; epochID < epoch_rp_; epochID++) {

                gd_normal_fast_cpu_.ForwardBatch(start_userID, n_proc_user,
                                                 para_a_value_l.data(), para_b_value_l.data(),
                                                 para_a_gradient_l.data(), para_b_gradient_l.data());

                for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                    const int userID = proc_userID + start_userID;
                    float lr = 1;
//                    if (epochID < 4) {
//                        lr = n_sample_rank_;
//                    } else if (4 <= epochID && epochID < 8) {
//                        lr = n_sample_rank_ / 100;
//                    } else if (8 <= epochID && epochID <= 10) {
//                        lr = n_sample_rank_ / 10000;
//                    }

                    float tmp_a = para_a_value_l[proc_userID];
                    float tmp_b = para_b_value_l[proc_userID];

                    tmp_a -= para_a_gradient_l[proc_userID] * lr;
                    tmp_b -= para_b_gradient_l[proc_userID] * lr;
                    para_a_value_l[proc_userID] = tmp_a;
                    para_b_value_l[proc_userID] = tmp_b;
                }

            }

            linear_program_time = record.get_elapsed_time_second();

            record.reset();
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const int userID = proc_userID + start_userID;
                //compute average, std
                predict_para_l_[userID * n_predict_parameter_] = para_a_value_l[proc_userID];
                predict_para_l_[userID * n_predict_parameter_ + 1] = para_b_value_l[proc_userID];

            }

            gd_normal_fast_cpu_.ForwardCalcError(start_userID, n_proc_user,
                                                 predict_para_l_.get(), n_predict_parameter_,
                                                 error_l_.get());

            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const int userID = proc_userID + start_userID;
                if (userID % 50000 == 0) {
                    printf("a %.3f, b %.3f, error %.3f\n", predict_para_l_[userID * n_predict_parameter_],
                           predict_para_l_[userID * n_predict_parameter_ + 1], error_l_[userID]);
                }
            }
            calc_error_time = record.get_elapsed_time_second();

        }

        void FinishPreprocess() override {
        }

        inline void
        ComputeRankBound(const float &queryIP, const int &userID,
                         int &rank_lb, int &rank_ub) const {

            if (error_l_[userID] >= n_sample_rank_ - 1e-3) {
                rank_lb = (int) n_data_item_;
                rank_ub = 0;
                return;
            }

            const size_t distribution_pos = userID * n_distribution_parameter_;
            const float low_value = distribution_para_l_[distribution_pos];
            const float high_value = distribution_para_l_[distribution_pos + 1];
            const float input_x = (queryIP - low_value) / (high_value - low_value);

            const size_t pred_pos = userID * n_predict_parameter_;
            const float pred_rank = input_x * predict_para_l_[pred_pos] + predict_para_l_[pred_pos + 1];
            const int pred_sample_rank_lb = std::ceil(pred_rank + error_l_[userID]);
            const int pred_sample_rank_ub = std::floor(pred_rank - error_l_[userID]);
            assert(0 <= error_l_[userID] && error_l_[userID] <= n_sample_rank_ + 0.1);

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

//            if (userID % 10000 == 0) {
//                printf("pred_rank %.3f, error_l %.3f, pred_sample_rank_lb %ld, pred_sample_rank_ub %ld, rank_lb %d, rank_ub %d, userID %d, n_sample_rank %d\n",
//                       pred_rank, error_l_[userID], pred_sample_rank_lb, pred_sample_rank_ub, rank_lb, rank_ub, userID,
//                       n_sample_rank_);
//            }
        }

        inline void
        ComputeRankBound(const float &queryIP_lb, const float &queryIP_ub, const int &userID,
                         int &rank_lb, int &rank_ub) const {

            if (error_l_[userID] >= n_sample_rank_ - 1e-3) {
                rank_lb = (int) n_data_item_;
                rank_ub = 0;
                return;
            }

            const size_t pred_pos = userID * n_predict_parameter_;

            const size_t distribution_pos = userID * n_distribution_parameter_;
            const float low_value = distribution_para_l_[distribution_pos];
            const float high_value = distribution_para_l_[distribution_pos + 1];

            const float input_x_lb = (queryIP_lb - low_value) / (high_value - low_value);
            const float pred_rank_lb = input_x_lb * predict_para_l_[pred_pos] + predict_para_l_[pred_pos + 1];

            const float input_x_ub = (queryIP_ub - low_value) / (high_value - low_value);
            const float pred_rank_ub = input_x_ub * predict_para_l_[pred_pos] + predict_para_l_[pred_pos + 1];


            const int pred_sample_rank_lb = std::ceil(pred_rank_lb + error_l_[userID]);
            const int pred_sample_rank_ub = std::floor(pred_rank_ub - error_l_[userID]);
            assert(0 <= error_l_[userID] && error_l_[userID] <= n_sample_rank_ + 0.1);

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

        void RankBound(const std::vector<float> &queryIP_l,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l) const {
            assert(queryIP_l.size() == n_user_);
            assert(rank_lb_l.size() == n_user_);
            assert(rank_ub_l.size() == n_user_);
#pragma omp parallel for default(none) shared(queryIP_l, rank_lb_l, rank_ub_l) num_threads(omp_get_num_procs())
            for (int userID = 0; userID < n_user_; userID++) {
                int lower_rank, upper_rank;

                const float queryIP = queryIP_l[userID];

                ComputeRankBound(queryIP, userID,
                                 lower_rank, upper_rank);
                assert(upper_rank <= lower_rank);

                rank_lb_l[userID] = lower_rank;
                rank_ub_l[userID] = upper_rank;
            }
        }

        void RankBound(const std::vector<std::pair<float, float>> &queryIP_l,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l) const {

#pragma omp parallel for default(none) shared(queryIP_l, rank_lb_l, rank_ub_l) num_threads(omp_get_num_procs())
            for (int batchID = 0; batchID < n_batch_; batchID++) {
                const int start_userID = batchID * batch_n_user_;
                const int end_userID = std::min((int) n_user_, (batchID + 1) * batch_n_user_);
                for (int userID = start_userID; userID < end_userID; userID++) {
                    const float queryIP_lb = queryIP_l[userID].first;
                    const float queryIP_ub = queryIP_l[userID].second;
                    assert(queryIP_lb <= queryIP_ub);
                    int qIP_lb_tmp_lower_rank, qIP_ub_tmp_upper_rank;

                    ComputeRankBound(queryIP_lb, queryIP_ub, userID,
                                     qIP_lb_tmp_lower_rank, qIP_ub_tmp_upper_rank);
                    assert(qIP_ub_tmp_upper_rank <= qIP_lb_tmp_lower_rank);

                    rank_lb_l[userID] = qIP_lb_tmp_lower_rank;
                    rank_ub_l[userID] = qIP_ub_tmp_upper_rank;

//                    if (userID % 50000 == 0) {
//                        const size_t pred_pos = userID * n_predict_parameter_;
//                        printf("userID %d, rank_lb %d, rank_ub %d, queryIP_lb %.3f, queryIP_ub %.3f, error %.3f, a %.3f, b %.3f\n",
//                               userID, rank_lb_l[userID], rank_ub_l[userID], queryIP_lb, queryIP_ub, error_l_[userID],
//                               predict_para_l_[pred_pos], predict_para_l_[pred_pos + 1]);
//
//                    }
                }
            }
        }

        void SaveIndex(const char *index_basic_dir, const char *dataset_name, const size_t &n_sample_query,
                       const size_t &sample_topk) override {
            char index_path[256];
            sprintf(index_path,
                    "%s/memory_index/UniformLinearRegressionFast-%s-%s-n_sample_%d-n_sample_query_%ld-sample_topk_%ld-epoch_rp_%d.index",
                    index_basic_dir, method_name_.c_str(), dataset_name, n_sample_rank_, n_sample_query, sample_topk,
                    epoch_rp_);


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
                       const size_t &n_sample, const size_t &n_sample_query, const size_t &sample_topk,
                       const int &epoch_rp) {
            char index_path[256];
            sprintf(index_path,
                    "%s/memory_index/UniformLinearRegressionFast-%s-%s-n_sample_%ld-n_sample_query_%ld-sample_topk_%ld-epoch_rp_%d.index",
                    index_basic_dir, method_name.c_str(), dataset_name, n_sample, n_sample_query, sample_topk,
                    epoch_rp);
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
#endif //REVERSE_KRANKS_UNIFORMLINEARREGRESSIONFAST_HPP
