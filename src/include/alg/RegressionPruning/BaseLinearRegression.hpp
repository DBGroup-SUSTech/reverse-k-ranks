//
// Created by BianZheng on 2022/11/11.
//

#ifndef REVERSE_K_RANKS_BASELINEARREGRESSION_HPP
#define REVERSE_K_RANKS_BASELINEARREGRESSION_HPP
namespace ReverseMIPS {
    class BaseLinearRegression {
    public:

        static constexpr int batch_n_user_ = 4096;

        inline BaseLinearRegression() = default;

        virtual void StartPreprocess(const int *sampleIP_l, const int &n_sample_rank) = 0;

        virtual void BatchLoopPreprocess(const std::vector<const float *> &sampleIP_l_l,
                                         const int &start_userID, const int &n_proc_user,
                                         double &assign_cache_time, double &linear_program_time,
                                         double &calc_error_time) = 0;

        virtual void FinishPreprocess() = 0;

        virtual void SaveIndex(const char *index_basic_dir, const char *dataset_name, const size_t &n_sample_query,
                               const size_t &sample_topk) = 0;

    };
}
#endif //REVERSE_K_RANKS_BASELINEARREGRESSION_HPP
