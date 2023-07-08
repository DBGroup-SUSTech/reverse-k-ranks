//
// Created by bianzheng on 2023/5/18.
//

#ifndef REVERSE_KRANKS_BASECOMPUTE_HPP
#define REVERSE_KRANKS_BASECOMPUTE_HPP
namespace ReverseMIPS {
    class BaseCompute {
    public:

        std::vector<UserRankElement> user_topk_cache_l_;

        inline BaseCompute() = default;

        virtual void GetRank(const std::vector<float> &queryIP_l,
                             const std::vector<int> &rank_lb_l, const std::vector<int> &rank_ub_l,
                             const std::vector<char> &prune_l, const std::vector<char> &result_l,
                             const int &n_remain_result, size_t &refine_ip_cost, int &n_refine_user,
                             int64_t &n_compute_item, double &refine_user_time, double &single_refine_user_time) = 0;

        virtual void FinishCompute() = 0;

    };
}
#endif //REVERSE_KRANKS_BASECOMPUTE_HPP
