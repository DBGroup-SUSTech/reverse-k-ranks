//
// Created by bianzheng on 2023/4/3.
//

#ifndef REVERSE_KRANKS_NORMALLINEARREGRESSIONFAST_HPP
#define REVERSE_KRANKS_NORMALLINEARREGRESSIONFAST_HPP

#include "BaseLinearRegression.hpp"
#include "struct/DistancePair.hpp"
#include "util/MathUtil.hpp"


#include <iostream>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <spdlog/spdlog.h>

//#ifdef USE_GPU
//#include "alg/RegressionPruning/GradientDescent/GDNormalFastGPU.hpp"
//#else

#include "alg/RegressionPruning/GradientDescent/GDNormalFastCPU.hpp"

//#endif

namespace ReverseMIPS {

    class NormalLinearRegressionFast : public BaseLinearRegression {

        std::string method_name_;
        size_t n_data_item_, n_user_;
        int epoch_rp_;
        static constexpr int n_predict_parameter_ = 2; // (a, b) for linear estimation
        static constexpr int n_distribution_parameter_ = 2; // mu, sigma
        static constexpr float sqrt_2_ = sqrt(2.0);
        int n_sample_rank_;
        std::unique_ptr<int[]> sample_rank_l_; // n_sample_rank
        std::unique_ptr<float[]> predict_para_l_; // n_user_ * n_predict_parameter
        std::unique_ptr<float[]> distribution_para_l_; // n_user_ * n_distribution_parameter
        std::unique_ptr<float[]> error_l_; //n_user_

        //for build index
//#ifdef USE_GPU
//        GDNormalFastGPU gd_normal_fast_gpu_;
//#else
        GDNormalFastCPU gd_normal_fast_cpu_;
//#endif

        int n_batch_;
        int n_report_user_;
    public:

        inline NormalLinearRegressionFast() {}

        inline NormalLinearRegressionFast(const int &n_data_item, const int &n_user, const int &epoch_rp,
                                          const std::string &method_name) {
            this->method_name_ = method_name;
            this->n_data_item_ = n_data_item;
            this->n_user_ = n_user;
            this->epoch_rp_ = epoch_rp;
            this->predict_para_l_ = std::make_unique<float[]>(n_user * n_predict_parameter_);
            this->distribution_para_l_ = std::make_unique<float[]>(n_user * n_distribution_parameter_);
            this->error_l_ = std::make_unique<float[]>(n_user);
            this->n_batch_ = (int) n_user_ / batch_n_user_ + (n_user_ % batch_n_user_ == 0 ? 0 : 1);
            this->n_report_user_ = n_user / 20;
            static_assert(n_predict_parameter_ == 2 && n_distribution_parameter_ == 2);
        }

        inline NormalLinearRegressionFast(const char *index_basic_dir, const char *dataset_name,
                                          const std::string &method_name,
                                          const size_t &n_sample, const size_t &n_sample_query,
                                          const size_t &sample_topk, const int &epoch_rp) {
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

//#ifdef USE_GPU
//            gd_normal_fast_gpu_ = GDNormalFastGPU(n_sample_rank_, batch_n_user_);
//#else
            gd_normal_fast_cpu_ = GDNormalFastCPU(n_sample_rank_, batch_n_user_);
//#endif

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
//#ifdef USE_GPU
//            gd_normal_fast_gpu_.Load(sampleIP_l_l, start_userID, n_proc_user);
//            gd_normal_fast_gpu_.CalcDistributionPara(start_userID, n_proc_user,
//                                                     n_distribution_parameter_, distribution_para_l_.get());
//
//            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
//                //compute average, std
//                para_a_value_l[proc_userID] = (float) (n_sample_rank_ - 1) / (
//                        sampleIP_l_l[proc_userID][0] - sampleIP_l_l[proc_userID][n_sample_rank_ - 1]);
//                para_b_value_l[proc_userID] = (sampleIP_l_l[proc_userID][0] -
//                                               n_sample_rank_ * sampleIP_l_l[proc_userID][n_sample_rank_ - 1]) / (
//                                                      sampleIP_l_l[proc_userID][0] -
//                                                      sampleIP_l_l[proc_userID][n_sample_rank_ - 1]);
//
//            }
//            gd_normal_fast_gpu_.Precompute(distribution_para_l_.get(), n_distribution_parameter_,
//                                           start_userID, n_proc_user);
//#else
            gd_normal_fast_cpu_.Load(sampleIP_l_l, start_userID, n_proc_user);
            gd_normal_fast_cpu_.CalcDistributionPara(start_userID, n_proc_user,
                                                     n_distribution_parameter_, distribution_para_l_.get());

#pragma omp parallel for default(none) schedule(static) num_threads(omp_get_num_procs()) shared(n_proc_user, start_userID, sampleIP_l_l, para_a_value_l, para_b_value_l, n_sample_rank_)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                //compute average, std
                const int userID = proc_userID + start_userID;
                const float mu = distribution_para_l_[userID * n_distribution_parameter_];
                const float sigma = distribution_para_l_[userID * n_distribution_parameter_ + 1];

                const float first_sampleIP = CDFPhi((sampleIP_l_l[proc_userID][0] - mu) / sigma);
                const float last_sampleIP = CDFPhi((sampleIP_l_l[proc_userID][n_sample_rank_ - 1] - mu) / sigma);

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
//#endif


            assign_cache_time = record.get_elapsed_time_second();

            record.reset();

            for (int epochID = 0; epochID < epoch_rp_; epochID++) {

//#ifdef USE_GPU
//                gd_normal_fast_gpu_.ForwardBatch(start_userID, n_proc_user,
//                                                 para_a_value_l.data(), para_b_value_l.data(),
//                                                 para_a_gradient_l.data(), para_b_gradient_l.data());
//#else
                gd_normal_fast_cpu_.ForwardBatch(start_userID, n_proc_user,
                                                 para_a_value_l.data(), para_b_value_l.data(),
                                                 para_a_gradient_l.data(), para_b_gradient_l.data());
//#endif

//#pragma omp parallel for default(none) shared(n_proc_user, start_userID, para_a_value_l, para_b_value_l, para_a_gradient_l, para_b_gradient_l)
                for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                    const int userID = proc_userID + start_userID;
//                    if (epochID < 4) {
//                        lr = n_sample_rank_;
//                    } else if (4 <= epochID && epochID < 8) {
//                        lr = n_sample_rank_ / 100;
//                    } else if (8 <= epochID && epochID <= 10) {
//                        lr = n_sample_rank_ / 10000;
//                    }

                    float tmp_a = para_a_value_l[proc_userID];
                    float tmp_b = para_b_value_l[proc_userID];

                    float lr = 0.05f * std::max(std::abs(tmp_a), std::abs(tmp_b));
                    const float para_a_gradient = para_a_gradient_l[proc_userID];
                    const float para_b_gradient = para_b_gradient_l[proc_userID];

                    const float max_value = std::max(std::abs(para_a_gradient), std::abs(para_b_gradient));
                    const float scale_num = max_value == 0 ? 1 : std::pow(10, std::floor(std::log10(max_value)));
                    const float para_a_first_digit = para_a_gradient / scale_num;
                    const float para_b_first_digit = para_b_gradient / scale_num;

                    tmp_a -= para_a_first_digit * lr;
                    tmp_b -= para_b_first_digit * lr;

//                    ::printf(
//                            "userID %d, para_a_gradient %.3f, para_b_gradient %.3f, para_a_first_digit %.3f, para_b_first_digit %.3f\n",
//                            userID, para_a_gradient, para_b_gradient, para_a_first_digit, para_b_first_digit);
//                    std::printf("tmp_a %.3f, tmp_b %.3f\n", tmp_a, tmp_b);

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

//#ifdef USE_GPU
//            gd_normal_fast_gpu_.ForwardCalcError(start_userID, n_proc_user,
//                                                 predict_para_l_.get(), n_predict_parameter_,
//                                                 error_l_.get());
//#else
            gd_normal_fast_cpu_.ForwardCalcError(start_userID, n_proc_user,
                                                 predict_para_l_.get(), n_predict_parameter_,
                                                 error_l_.get());
//#endif

//            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
//                const int userID = proc_userID + start_userID;
//                if (userID % n_report_user_ == 0) {
//                    spdlog::info("userID {}, progress {:.3f}, a {:.3f}, b {:.3f}, error {:.3f}",
//                                 userID, 1.0f * (float) userID / (float) n_user_,
//                                 predict_para_l_[userID * n_predict_parameter_],
//                                 predict_para_l_[userID * n_predict_parameter_ + 1], error_l_[userID]);
//                }
//            }
            calc_error_time = record.get_elapsed_time_second();

        }

        void FinishPreprocess() override {
//#ifdef USE_GPU
//            gd_normal_fast_gpu_.FinishPreprocess();
//#else
//#endif
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
            const float mu = distribution_para_l_[distribution_pos];
            const float sigma = distribution_para_l_[distribution_pos + 1];

            const float normalize_x_lb = (queryIP_lb - mu) / sigma;
            const float input_x_lb = CDFPhi(normalize_x_lb);
            const float pred_rank_lb = input_x_lb * predict_para_l_[pred_pos] + predict_para_l_[pred_pos + 1];

            const float normalize_x_ub = (queryIP_ub - mu) / sigma;
            const float input_x_ub = CDFPhi(normalize_x_ub);
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

        void RankBound(const std::vector<std::pair<float, float>> &queryIP_l,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l) const {

#pragma omp parallel for default(none) shared(queryIP_l, rank_lb_l, rank_ub_l) num_threads(omp_get_num_procs())
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
                    "%s/memory_index/NormalLinearRegressionFast-%s-%s-n_sample_%d-n_sample_query_%ld-sample_topk_%ld-epoch_rp_%d.index",
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
                    "%s/memory_index/NormalLinearRegressionFast-%s-%s-n_sample_%ld-n_sample_query_%ld-sample_topk_%ld-epoch_rp_%d.index",
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
#endif //REVERSE_KRANKS_NORMALLINEARREGRESSIONFAST_HPP
