//
// Created by BianZheng on 2022/10/14.
//

#ifndef REVERSE_K_RANKS_HEADLINEARREGRESSION_HPP
#define REVERSE_K_RANKS_HEADLINEARREGRESSION_HPP

#include "alg/RegressionPruning/BaseLinearRegression.hpp"
#include "struct/DistancePair.hpp"
#include "util/MathUtil.hpp"

#include <iostream>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <spdlog/spdlog.h>


namespace ReverseMIPS {

    class LeastSquareLinearRegression : public BaseLinearRegression {

        size_t n_data_item_, n_user_;
        static constexpr int n_predict_parameter_ = 2; // (a, b) for linear estimation
        static constexpr int n_distribution_parameter_ = 2; // mu, sigma
        static constexpr double sqrt_2_ = sqrt(2.0);
        int n_sample_rank_;
        std::unique_ptr<int[]> sample_rank_l_; // n_sample_rank
        std::unique_ptr<double[]> predict_para_l_; // n_user_ * n_predict_parameter
        std::unique_ptr<double[]> distribution_para_l_; // n_user_ * n_distribution_parameter
        std::unique_ptr<int[]> error_l_; //n_user_

        //used for loading
        double *preprocess_cache_X_; // n_sample_rank * n_predict_parameter_, store queryIP in the sampled rank
        double *preprocess_cache_Y_; // n_sample_rank, store the double type of sampled rank value
    public:

        inline LeastSquareLinearRegression() {}

        inline LeastSquareLinearRegression(const int &n_data_item, const int &n_user) {
            this->n_data_item_ = n_data_item;
            this->n_user_ = n_user;
            this->predict_para_l_ = std::make_unique<double[]>(n_user * n_predict_parameter_);
            this->distribution_para_l_ = std::make_unique<double[]>(n_user * n_distribution_parameter_);
            this->error_l_ = std::make_unique<int[]>(n_user);
        }

        inline LeastSquareLinearRegression(const char *index_basic_dir, const char *dataset_name,
                                           const size_t &n_sample, const size_t &n_sample_query,
                                           const size_t &sample_topk) {
            LoadIndex(index_basic_dir, dataset_name, n_sample, n_sample_query, sample_topk);
        }

        void StartPreprocess(const int *sample_rank_l, const int &n_sample_rank) override {
            this->n_sample_rank_ = n_sample_rank;
            this->sample_rank_l_ = std::make_unique<int[]>(n_sample_rank);
            this->preprocess_cache_X_ = new double[n_sample_rank * n_predict_parameter_];
            this->preprocess_cache_Y_ = new double[n_sample_rank];
            for (int sampleID = 0; sampleID < n_sample_rank; sampleID++) {
                preprocess_cache_Y_[sampleID] = sampleID;
                sample_rank_l_[sampleID] = sample_rank_l[sampleID];
            }

        }

        double ComputeAverage(const double *sampleIP_l) const {
            double average = 0;
            for (int sampleID = 0; sampleID < n_sample_rank_; sampleID++) {
                average += sampleIP_l[sampleID];
            }
            return average / n_sample_rank_;
        }

        double ComputeStd(const double *sampleIP_l, const double average) const {
            double sigma = 0;
            for (int sampleID = 0; sampleID < n_sample_rank_; sampleID++) {
                const double minus = sampleIP_l[sampleID] - average;
                const double term = minus * minus;
                sigma += term;
            }
            sigma /= n_sample_rank_;
            return std::sqrt(sigma);
        }

        double CDFPhi(double x) const {
            // constants
            double a1 = 0.254829592;
            double a2 = -0.284496736;
            double a3 = 1.421413741;
            double a4 = -1.453152027;
            double a5 = 1.061405429;
            double p = 0.3275911;

            // Save the sign of x
            int sign = 1;
            if (x < 0)
                sign = -1;
            x = fabs(x) / sqrt_2_;

            // A&S formula 7.1.26
            double t = 1.0 / (1.0 + p * x);
            double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

            return 0.5 * (1.0 + sign * y);
        }

        void LoopPreprocess(const double *sampleIP_l, const int &userID) override {
            //compute average, std
            const double mu = ComputeAverage(sampleIP_l);
            const double sigma = ComputeStd(sampleIP_l, mu);
            distribution_para_l_[userID * n_distribution_parameter_] = mu;
            distribution_para_l_[userID * n_distribution_parameter_ + 1] = sigma;

#pragma omp parallel for default(none) shared(sampleIP_l, mu, sigma)
            for (int sampleID = 0; sampleID < n_sample_rank_; sampleID++) {
                preprocess_cache_X_[sampleID * n_predict_parameter_] = 1;
                const double normal_num = (sampleIP_l[sampleID] - mu) / sigma;
                preprocess_cache_X_[sampleID * n_predict_parameter_ + 1] = CDFPhi(normal_num);
            }
            using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            Eigen::Map<RowMatrixXd> X(preprocess_cache_X_, n_sample_rank_, (int64_t) n_predict_parameter_);

//    printf("%.3f %.3f %.3f %.3f\n", X_cache[0], X_cache[1], X_cache[2], X_cache[3]);
//    std::cout << X.row(1) << std::endl;
//    std::cout << X.col(1).size() << std::endl;
//    printf("X rows %ld, cols %ld\n", X.MapBase<Eigen::Map<Eigen::Matrix<double, -1, -1, 1>, 0>, 0>::rows(),
//           X.MapBase<Eigen::Map<Eigen::Matrix<double, -1, -1, 1>, 0>, 0>::cols());

            Eigen::Map<Eigen::VectorXd> Y(preprocess_cache_Y_, n_sample_rank_);
            Eigen::VectorXd res = (X.transpose() * X).ldlt().solve(X.transpose() * Y);
            assert(res.rows() == n_predict_parameter_);
//            printf("res rows %ld, cols %ld\n", res.rows(), res.cols());
//            printf("res [0]: %.3f, [1]: %.3f\n", res[0], res[1]);

            //assign parameter
            for (int paraID = 0; paraID < n_predict_parameter_; paraID++) {
                predict_para_l_[userID * n_predict_parameter_ + paraID] = res[paraID];
            }

            //assign error
            int error = -1;
#pragma omp parallel for default(none) shared(userID, error)
            for (int sampleID = 0; sampleID < n_sample_rank_; sampleID++) {
                double pred_rank = 0;
                for (int paraID = 0; paraID < n_predict_parameter_; paraID++) {
                    pred_rank += predict_para_l_[userID * n_predict_parameter_ + paraID] *
                                 preprocess_cache_X_[sampleID * n_predict_parameter_ + paraID];
                }
                const int real_rank = sampleID;
                const int tmp_error = std::abs(std::floor(pred_rank) - real_rank);
#pragma omp critical
                error = std::max(tmp_error, error);
            }

            error_l_[userID] = error;
            assert(-1 < error && error < n_sample_rank_);
        }

        void FinishPreprocess() override {
            delete[] preprocess_cache_X_;
            delete[] preprocess_cache_Y_;
            preprocess_cache_X_ = nullptr;
            preprocess_cache_Y_ = nullptr;
        }

        inline void
        ComputeRankBound(const double &queryIP, const int &userID,
                         int &rank_lb, int &rank_ub, const int &queryID) const {

            const size_t distribution_pos = userID * n_distribution_parameter_;
            const double mu = distribution_para_l_[distribution_pos];
            const double sigma = distribution_para_l_[distribution_pos + 1];
            const double normalize_x = (queryIP - mu) / sigma;
            const double input_x = CDFPhi(normalize_x);

            const size_t pred_pos = userID * n_predict_parameter_;
            const double pred_rank = predict_para_l_[pred_pos] + input_x * predict_para_l_[pred_pos + 1];
            const int pred_int_rank = std::floor(pred_rank);
            const int pred_sample_rank_lb = pred_int_rank + error_l_[userID];
            const int pred_sample_rank_ub = pred_int_rank - error_l_[userID];

//            if (queryID == 3 && userID == 865) {
//                printf("queryID %d, userID %d, queryIP %.3f, pred_int_rank %d, error %d, pred_sample_rank_lb %d, pred_sample_rank_ub %d\n",
//                       queryID, userID, queryIP, pred_int_rank, error_l_[userID], pred_sample_rank_lb,
//                       pred_sample_rank_ub);
//            }

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
//                       const std::vector<bool> &prune_l, const std::vector<bool> &result_l,
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

        void RankBound(const std::vector<std::pair<double, double>> &queryIP_l,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l, const int &queryID) const {
            for (int userID = 0; userID < n_user_; userID++) {
                const double queryIP_lb = queryIP_l[userID].first;
                int qIP_lb_tmp_lower_rank, qIP_lb_tmp_upper_rank;

                ComputeRankBound(queryIP_lb, userID,
                                 qIP_lb_tmp_lower_rank, qIP_lb_tmp_upper_rank, queryID);

                const double queryIP_ub = queryIP_l[userID].second;
                int qIP_ub_tmp_lower_rank, qIP_ub_tmp_upper_rank;
                ComputeRankBound(queryIP_ub, userID,
                                 qIP_ub_tmp_lower_rank, qIP_ub_tmp_upper_rank, queryID);

                rank_lb_l[userID] = qIP_lb_tmp_lower_rank;
                rank_ub_l[userID] = qIP_ub_tmp_upper_rank;

//                if (queryID == 3 && userID == 865) {
//                    printf("queryID %d, userID %d, queryIP_lb %.3f, queryIP_ub %.3f, rank_lb %d, rank_ub %d\n",
//                           queryID, userID, queryIP_lb, queryIP_ub, rank_lb_l[userID], rank_ub_l[userID]);
//                }

                assert(qIP_lb_tmp_upper_rank <= qIP_lb_tmp_lower_rank);
                assert(qIP_ub_tmp_upper_rank <= qIP_ub_tmp_lower_rank);
                assert(qIP_ub_tmp_upper_rank <= qIP_lb_tmp_lower_rank);
            }
        }

        void SaveIndex(const char *index_basic_dir, const char *dataset_name,
                       const size_t &n_sample_query, const size_t &sample_topk) override {
            char index_path[256];
            sprintf(index_path,
                    "%s/memory_index/LeastSquareLinearRegression-%s-n_sample_%d-n_sample_query_%ld-sample_topk_%ld.index",
                    index_basic_dir, dataset_name, n_sample_rank_, n_sample_query, sample_topk);

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
                              (int64_t) (n_user_ * n_predict_parameter_ * sizeof(double)));
            out_stream_.write((char *) distribution_para_l_.get(),
                              (int64_t) (n_user_ * n_distribution_parameter_ * sizeof(double)));
            out_stream_.write((char *) error_l_.get(), (int64_t) (n_user_ * sizeof(int)));

            out_stream_.close();
        }

        void LoadIndex(const char *index_basic_dir, const char *dataset_name,
                       const size_t &n_sample, const size_t &n_sample_query, const size_t &sample_topk) {
            char index_path[256];
            sprintf(index_path,
                    "%s/memory_index/LeastSquareLinearRegression-%s-n_sample_%ld-n_sample_query_%ld-sample_topk_%ld.index",
                    index_basic_dir, dataset_name, n_sample, n_sample_query, sample_topk);
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

            predict_para_l_ = std::make_unique<double[]>(n_user_ * n_predict_parameter_);
            index_stream.read((char *) predict_para_l_.get(),
                              (int64_t) (sizeof(double) * n_user_ * n_predict_parameter_));

            distribution_para_l_ = std::make_unique<double[]>(n_user_ * n_distribution_parameter_);
            index_stream.read((char *) distribution_para_l_.get(),
                              (int64_t) (sizeof(double) * n_user_ * n_distribution_parameter_));

            error_l_ = std::make_unique<int[]>(n_user_);
            index_stream.read((char *) error_l_.get(),
                              (int64_t) (sizeof(int) * n_user_));

            index_stream.close();
        }


        uint64_t IndexSizeByte() const {
            const uint64_t sample_rank_size = sizeof(int) * n_sample_rank_;
            const uint64_t para_size = sizeof(double) * n_user_ * (n_predict_parameter_ + n_distribution_parameter_);
            const uint64_t error_size = sizeof(int) * n_user_;
            return sample_rank_size + para_size + error_size;
        }

    };
}
#endif //REVERSE_K_RANKS_HEADLINEARREGRESSION_HPP
