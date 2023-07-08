//
// Created by bianzheng on 2023/4/3.
//

#ifndef REVERSE_KRANKS_GDCPU_HPP
#define REVERSE_KRANKS_GDCPU_HPP
namespace ReverseMIPS {
    class GDNormalFastCPU {
        int n_sample_rank_, batch_n_user_;

        static constexpr float sqrt_2_ = sqrt(2.0);
        static constexpr int n_power = 8;

        std::vector<float> transform_array_; //batch_n_user_ * n_sample_rank_

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

    public:
        inline GDNormalFastCPU() {}

        inline GDNormalFastCPU(const int n_sample_rank, const int batch_n_user) {
            this->n_sample_rank_ = n_sample_rank;
            this->batch_n_user_ = batch_n_user;
            this->transform_array_.resize(batch_n_user_ * n_sample_rank_);
        }

        void Load(std::vector<const float *> sampleIP_l_l, const int start_userID, const int n_proc_user) {
#pragma omp parallel for default(none) schedule(static) num_threads(omp_get_num_procs()) shared(n_proc_user, start_userID, sampleIP_l_l)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                memcpy(transform_array_.data() + proc_userID * n_sample_rank_, sampleIP_l_l[proc_userID],
                       sizeof(float) * n_sample_rank_);
            }

        }

        void CalcDistributionPara(const int start_userID, const int n_proc_user,
                                  const int n_distribution_parameter, float *distribution_para_l) {
#pragma omp parallel for default(none) schedule(static) num_threads(omp_get_num_procs()) shared(n_proc_user, start_userID, distribution_para_l, n_distribution_parameter)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const int userID = proc_userID + start_userID;
                //compute average, std
                const float mu = ComputeAverage(transform_array_.data() + proc_userID * n_sample_rank_);
                const float sigma = ComputeStd(transform_array_.data() + proc_userID * n_sample_rank_, mu);
                distribution_para_l[userID * n_distribution_parameter] = mu;
                distribution_para_l[userID * n_distribution_parameter + 1] = sigma;

            }
        }

        void Precompute(const float *distribution_para_l, const int n_distribution_parameter,
                        const int start_userID, const int n_proc_user) {

#pragma omp parallel for default(none) schedule(static) num_threads(omp_get_max_threads()) shared(n_proc_user, start_userID, distribution_para_l, n_distribution_parameter)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const int userID = proc_userID + start_userID;
                const float mu = distribution_para_l[userID * n_distribution_parameter];
                const float sigma = distribution_para_l[userID * n_distribution_parameter + 1];

                float *sampleIP_l = transform_array_.data() + proc_userID * n_sample_rank_;
                for (int sampleID = 0; sampleID < n_sample_rank_; sampleID++) {
                    const float sampleIP = sampleIP_l[sampleID];
                    const float normal_num = (sampleIP - mu) / sigma;
                    const float cdf = CDFPhi(normal_num);
                    sampleIP_l[sampleID] = cdf;
                }

            }
        }

        void ForwardBatch(const int start_userID, const int n_proc_user,
                          const float *para_a_value_ptr, const float *para_b_value_ptr,
                          float *para_a_gradient_ptr, float *para_b_gradient_ptr) {

            assert(n_proc_user <= batch_n_user_);

            //compute the gradient
//#pragma omp parallel for default(none) shared(n_proc_user, start_userID, para_a_value_ptr, para_b_value_ptr, para_a_gradient_ptr, para_b_gradient_ptr)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                float *sampleIP_l = transform_array_.data() + proc_userID * n_sample_rank_;
                const int userID = proc_userID + start_userID;

                const float para_a = para_a_value_ptr[proc_userID];
                const float para_b = para_b_value_ptr[proc_userID];

                float para_a_gradient = 0;
                float para_b_gradient = 0;
                int n_scale_digit = 0;
                float scale_number = std::pow(10, n_scale_digit);
                for (int sampleID = 0; sampleID < n_sample_rank_; sampleID++) {
                    const float cdf = sampleIP_l[sampleID];
                    const float pred_rank = cdf * para_a + para_b;
                    float tmp_error = (float) (sampleID + 1) - pred_rank;
                    para_a_gradient +=
                            (-cdf) * n_power * std::pow(tmp_error, (float) n_power - 1) / scale_number;
                    para_b_gradient += (-1) * n_power * std::pow(tmp_error, (float) n_power - 1) / scale_number;

                    float tmp_error2 = (float) (sampleID) - pred_rank;
                    para_a_gradient +=
                            (-cdf) * n_power * std::pow(tmp_error2, (float) n_power - 1) / scale_number;
                    para_b_gradient += (-1) * n_power * std::pow(tmp_error2, (float) n_power - 1) / scale_number;

                    const float max_value = std::max(std::abs(para_a_gradient), std::abs(para_b_gradient));

                    if (max_value > 1000) {
                        const float add_n_scale_digit = std::floor(std::log10(max_value));

                        para_a_gradient /= std::pow(10, add_n_scale_digit);
                        para_b_gradient /= std::pow(10, add_n_scale_digit);

                        n_scale_digit += (int) add_n_scale_digit;
                        scale_number = std::pow(10, n_scale_digit);
                    }
//                    if (userID == 0) {
//                        std::printf(
//                                "userID %d, n_scale_digit %d, scale_number %.3f, para_a_gradient %f, para_b_gradient %f\n",
//                                userID, n_scale_digit, scale_number, para_a_gradient, para_b_gradient);
//                    }
                }

                para_a_gradient_ptr[proc_userID] = para_a_gradient;
                para_b_gradient_ptr[proc_userID] = para_b_gradient;

            }

        }

        void ForwardCalcError(const int start_userID, const int n_proc_user,
                              const float *predict_para_ptr, const int n_predict_parameter,
                              float *error_ptr) {
            assert(n_proc_user <= batch_n_user_);

            //compute all tmp_error
#pragma omp parallel for default(none) schedule(static) num_threads(omp_get_max_threads()) shared(n_proc_user, start_userID, predict_para_ptr, n_predict_parameter, error_ptr)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                float *sampleIP_l = transform_array_.data() + proc_userID * n_sample_rank_;
                const int userID = proc_userID + start_userID;

                const float para_a = predict_para_ptr[userID * n_predict_parameter];
                const float para_b = predict_para_ptr[userID * n_predict_parameter + 1];

                float max_error = -1;

                for (int sampleID = 0; sampleID < n_sample_rank_ - 1; sampleID++) {
                    const float cdf = sampleIP_l[sampleID];
                    const float pred_rank = cdf * para_a + para_b;
                    float tmp_error = (float) (sampleID + 1) - pred_rank;
                    tmp_error = std::abs(tmp_error);
                    max_error = std::max(max_error, tmp_error);

                    const float next_cdf = sampleIP_l[sampleID + 1];
                    const float next_pred_rank = next_cdf * para_a + para_b;
                    tmp_error = (float) (sampleID + 1) - next_pred_rank;
                    tmp_error = std::abs(tmp_error);
                    max_error = std::max(max_error, tmp_error);

                }

                {
                    const int sampleID = n_sample_rank_ - 1;
                    const float cdf = sampleIP_l[sampleID];
                    const float pred_rank = cdf * para_a + para_b;
                    float tmp_error = (float) (sampleID) - pred_rank;
                    tmp_error = std::abs(tmp_error);
                    max_error = std::max(max_error, tmp_error);
                }

                {
                    const int sampleID = 0;
                    const float cdf = sampleIP_l[sampleID];
                    const float pred_rank = cdf * para_a + para_b;
                    float tmp_error = (float) (sampleID) - pred_rank;
                    tmp_error = std::abs(tmp_error);
                    max_error = std::max(max_error, tmp_error);
                }
                assert(max_error != -1);

                max_error += 0.01f;
                if (max_error > (float) n_sample_rank_) {
                    max_error = n_sample_rank_ + 0.01f;
                }
                error_ptr[userID] = max_error;

            }

        }

    };
}
#endif //REVERSE_KRANKS_GDCPU_HPP
