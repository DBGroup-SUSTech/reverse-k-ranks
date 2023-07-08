//
// Created by bianzheng on 2023/4/14.
//

#ifndef REVERSE_KRANKS_GDUNIFORMFASTCPU_HPP
#define REVERSE_KRANKS_GDUNIFORMFASTCPU_HPP
namespace ReverseMIPS {
    class GDUniformFastCPU {
        int n_sample_rank_, batch_n_user_;

        std::vector<float> transform_array_; //batch_n_user_ * n_sample_rank_
        std::vector<float> error_arr_; //batch_n_user_ * n_sample_rank_ * 2
        std::vector<float> grad_a_arr_; //batch_n_user_ * n_sample_rank_ * 2
        std::vector<float> grad_b_arr_; //batch_n_user_ * n_sample_rank_ * 2

    public:
        inline GDUniformFastCPU() {}

        inline GDUniformFastCPU(const int n_sample_rank, const int batch_n_user) {
            this->n_sample_rank_ = n_sample_rank;
            this->batch_n_user_ = batch_n_user;
            this->transform_array_.resize(batch_n_user_ * n_sample_rank_);
            this->error_arr_.resize(batch_n_user_ * n_sample_rank_ * 2);
            this->grad_a_arr_.resize(batch_n_user_ * n_sample_rank_ * 2);
            this->grad_b_arr_.resize(batch_n_user_ * n_sample_rank_ * 2);
        }

        void Load(std::vector<const float *> sampleIP_l_l, const int start_userID, const int n_proc_user) {
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                memcpy(transform_array_.data() + proc_userID * n_sample_rank_, sampleIP_l_l[proc_userID],
                       sizeof(float) * n_sample_rank_);
            }

        }

        void CalcDistributionPara(const int start_userID, const int n_proc_user,
                                  const int n_distribution_parameter, float *distribution_para_l) {
#pragma omp parallel for default(none) schedule(static) num_threads(omp_get_max_threads()) shared(n_proc_user, start_userID, distribution_para_l, n_distribution_parameter)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const int userID = proc_userID + start_userID;
                //compute average, std
                const float *sampleIP_l = transform_array_.data() + proc_userID * n_sample_rank_;
                const float low_value = sampleIP_l[n_sample_rank_ - 1];
                const float high_value = sampleIP_l[0];

                distribution_para_l[userID * n_distribution_parameter] = low_value;
                distribution_para_l[userID * n_distribution_parameter + 1] = high_value;

            }
        }

        void Precompute(const float *distribution_para_l, const int n_distribution_parameter,
                        const int start_userID, const int n_proc_user) {

#pragma omp parallel for default(none) schedule(static) num_threads(omp_get_max_threads()) shared(n_proc_user, start_userID, distribution_para_l, n_distribution_parameter)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                const int userID = proc_userID + start_userID;
                const float low_value = distribution_para_l[userID * n_distribution_parameter];
                const float high_value = distribution_para_l[userID * n_distribution_parameter + 1];

                float *sampleIP_l = transform_array_.data() + proc_userID * n_sample_rank_;
                for (int sampleID = 0; sampleID < n_sample_rank_; sampleID++) {
                    const float sampleIP = sampleIP_l[sampleID];
                    const float cdf = (sampleIP - low_value) / (high_value - low_value);
                    sampleIP_l[sampleID] = cdf;
                }

            }
        }

        void ForwardBatch(const int start_userID, const int n_proc_user,
                          const float *para_a_value_ptr, const float *para_b_value_ptr,
                          float *para_a_gradient_ptr, float *para_b_gradient_ptr) {

            assert(n_proc_user <= batch_n_user_);

//            //compute all tmp_error
//#pragma omp parallel for default(none) schedule(static) num_threads(omp_get_max_threads()) shared(n_proc_user, start_userID, para_a_value_ptr, para_b_value_ptr)
//            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
//                float *sampleIP_l = transform_array_.data() + proc_userID * n_sample_rank_;
//                const int userID = proc_userID + start_userID;
//
//                const float para_a = para_a_value_ptr[proc_userID];
//                const float para_b = para_b_value_ptr[proc_userID];
//
//                for (int sampleID = 0; sampleID < n_sample_rank_ - 1; sampleID++) {
//                    const float cdf = sampleIP_l[sampleID];
//                    const float pred_rank = cdf * para_a + para_b;
//                    float tmp_error = (float) (sampleID + 1) - pred_rank;
//                    const bool is_error_negative2 = tmp_error < 0;
//                    tmp_error = std::abs(tmp_error);
//                    error_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2] = tmp_error;
//                    grad_a_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2] = is_error_negative2 ? cdf : -cdf;
//                    grad_b_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2] = is_error_negative2 ? 1 : -1;
//
//                    const float next_cdf = sampleIP_l[sampleID + 1];
//                    const float next_pred_rank = next_cdf * para_a + para_b;
//                    tmp_error = (float) (sampleID + 1) - next_pred_rank;
//                    const bool is_error_negative = tmp_error < 0;
//                    tmp_error = std::abs(tmp_error);
//                    error_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2 + 1] = tmp_error;
//                    grad_a_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2 + 1] = is_error_negative ? cdf : -cdf;
//                    grad_b_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2 + 1] = is_error_negative ? 1 : -1;
//
//                }
//
//                {
//                    const int sampleID = n_sample_rank_ - 1;
//                    const float cdf = sampleIP_l[sampleID];
//                    const float pred_rank = cdf * para_a + para_b;
//                    float tmp_error = (float) (sampleID) - pred_rank;
//                    const bool is_error_negative = tmp_error < 0;
//                    tmp_error = std::abs(tmp_error);
//                    error_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2] = tmp_error;
//                    grad_a_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2] = is_error_negative ? cdf : -cdf;
//                    grad_b_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2] = is_error_negative ? 1 : -1;
//                }
//
//                {
//                    const int sampleID = 0;
//                    const float cdf = sampleIP_l[sampleID];
//                    const float pred_rank = cdf * para_a + para_b;
//                    float tmp_error = (float) (sampleID) - pred_rank;
//                    const bool is_error_negative = tmp_error < 0;
//                    tmp_error = std::abs(tmp_error);
//                    error_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2 + 1] = tmp_error;
//                    grad_a_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2 + 1] = is_error_negative ? cdf : -cdf;
//                    grad_b_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2 + 1] = is_error_negative ? 1 : -1;
//                }
//
//            }

            //compute all tmp_error
#pragma omp parallel for default(none) schedule(static) num_threads(omp_get_max_threads()) shared(n_proc_user, start_userID, para_a_value_ptr, para_b_value_ptr)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                float *sampleIP_l = transform_array_.data() + proc_userID * n_sample_rank_;
                const int userID = proc_userID + start_userID;

                const float para_a = para_a_value_ptr[proc_userID];
                const float para_b = para_b_value_ptr[proc_userID];

                for (int sampleID = 0; sampleID < n_sample_rank_; sampleID++) {
                    const float cdf = sampleIP_l[sampleID];
                    const float pred_rank = cdf * para_a + para_b;
                    float tmp_error = (float) (sampleID + 1) - pred_rank;
                    const bool is_error_negative2 = tmp_error < 0;
                    tmp_error = std::abs(tmp_error);
                    error_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2] = tmp_error;
                    grad_a_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2] = is_error_negative2 ? cdf : -cdf;
                    grad_b_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2] = is_error_negative2 ? 1 : -1;

                    float tmp_error2 = (float) (sampleID) - pred_rank;
                    const bool is_error_negative = tmp_error2 < 0;
                    tmp_error2 = std::abs(tmp_error2);
                    error_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2 + 1] = tmp_error2;
                    grad_a_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2 + 1] = is_error_negative ? cdf : -cdf;
                    grad_b_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2 + 1] = is_error_negative ? 1 : -1;

                }

            }

#pragma omp parallel for default(none) schedule(static) num_threads(omp_get_max_threads()) shared(n_proc_user, start_userID, para_a_value_ptr, para_b_value_ptr, para_a_gradient_ptr, para_b_gradient_ptr)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {

                //use argmax to find the maximum error in error_arr
                float max_error = -1;
                int max_error_index = -1;
                for (int sampleID = 0; sampleID < n_sample_rank_ * 2; sampleID++) {
                    const float tmp_error = error_arr_[proc_userID * n_sample_rank_ * 2 + sampleID];
                    if (tmp_error > max_error) {
                        max_error = tmp_error;
                        max_error_index = sampleID;
                    }
                }
                assert(max_error_index != -1 && max_error_index < n_sample_rank_ * 2);

                para_a_gradient_ptr[proc_userID] = grad_a_arr_[proc_userID * n_sample_rank_ * 2 + max_error_index];
                para_b_gradient_ptr[proc_userID] = grad_b_arr_[proc_userID * n_sample_rank_ * 2 + max_error_index];

            }
        }

        void ForwardCalcError(const int start_userID, const int n_proc_user,
                              const float *predict_para_ptr, const int n_predict_parameter,
                              float *error_ptr) {
            assert(n_proc_user <= batch_n_user_);

            //compute all tmp_error
#pragma omp parallel for default(none) schedule(static) num_threads(omp_get_max_threads()) shared(n_proc_user, start_userID, predict_para_ptr, n_predict_parameter)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                float *sampleIP_l = transform_array_.data() + proc_userID * n_sample_rank_;
                const int userID = proc_userID + start_userID;

                const float para_a = predict_para_ptr[userID * n_predict_parameter];
                const float para_b = predict_para_ptr[userID * n_predict_parameter + 1];

                for (int sampleID = 0; sampleID < n_sample_rank_ - 1; sampleID++) {
                    const float cdf = sampleIP_l[sampleID];
                    const float pred_rank = cdf * para_a + para_b;
                    float tmp_error = (float) (sampleID + 1) - pred_rank;
                    tmp_error = std::abs(tmp_error);
                    error_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2] = tmp_error;

                    const float next_cdf = sampleIP_l[sampleID + 1];
                    const float next_pred_rank = next_cdf * para_a + para_b;
                    tmp_error = (float) (sampleID + 1) - next_pred_rank;
                    tmp_error = std::abs(tmp_error);
                    error_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2 + 1] = tmp_error;

                }

                {
                    const int sampleID = n_sample_rank_ - 1;
                    const float cdf = sampleIP_l[sampleID];
                    const float pred_rank = cdf * para_a + para_b;
                    float tmp_error = (float) (sampleID) - pred_rank;
                    tmp_error = std::abs(tmp_error);
                    error_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2] = tmp_error;
                }

                {
                    const int sampleID = 0;
                    const float cdf = sampleIP_l[sampleID];
                    const float pred_rank = cdf * para_a + para_b;
                    float tmp_error = (float) (sampleID) - pred_rank;
                    tmp_error = std::abs(tmp_error);
                    error_arr_[proc_userID * n_sample_rank_ * 2 + sampleID * 2 + 1] = tmp_error;
                }

            }

#pragma omp parallel for default(none) schedule(static) num_threads(omp_get_max_threads()) shared(n_proc_user, start_userID, error_ptr)
            for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
                float max_error = -1;
                for (int sampleID = 0; sampleID < n_sample_rank_ * 2; sampleID++) {
                    const float tmp_error = error_arr_[proc_userID * n_sample_rank_ * 2 + sampleID];
                    max_error = std::max(max_error, tmp_error);
                }
                max_error += 0.01f;
                if (max_error > n_sample_rank_) {
                    max_error = n_sample_rank_ + 0.01f;
                }
                const int userID = proc_userID + start_userID;
                error_ptr[userID] = max_error;
            }


        }

    };
}
#endif //REVERSE_KRANKS_GDUNIFORMFASTCPU_HPP
