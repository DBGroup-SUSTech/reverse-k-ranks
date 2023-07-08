//
// Created by BianZheng on 2022/6/29.
//

#ifndef REVERSE_K_RANKS_SCORESEARCHTOPT_HPP
#define REVERSE_K_RANKS_SCORESEARCHTOPT_HPP

#include "struct/DistancePair.hpp"

#include <cassert>
#include <vector>
#include <algorithm>
#include "spdlog/spdlog.h"

namespace ReverseMIPS {

    class ScoreSearchTopT {
    public:

        int n_interval_, topt_, n_user_, n_data_item_;
        // n_user * n_interval, the last element of an interval column must be n_data_item
        std::unique_ptr<int[]> interval_table_;
        // n_user, stores the distance of interval for each user
        std::unique_ptr<double[]> interval_dist_l_;
        // n_user, bound for column, first is lower bound, second is upper bound
        std::unique_ptr<std::pair<double, double>[]> user_ip_bound_l_;


        inline ScoreSearchTopT() = default;

        inline ScoreSearchTopT(const int &n_interval, const int topt, const int &n_user, const int &n_data_item) {
            this->n_interval_ = n_interval;
            this->topt_ = topt;
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            assert(topt_ <= n_data_item_);

            interval_table_ = std::make_unique<int[]>(n_user_ * n_interval_);
            std::memset(interval_table_.get(), 0, n_user_ * n_interval_ * sizeof(int));

            interval_dist_l_ = std::make_unique<double[]>(n_user_);
            user_ip_bound_l_ = std::make_unique<std::pair<double, double>[]>(n_user_);
            spdlog::info("interval bound: n_interval {}", n_interval);
        }

        void LoopPreprocess(const DistancePair *distance_ptr, const int &userID) {
            std::vector<double> IP_l(topt_);
            for (int candID = 0; candID < topt_; candID++) {
                IP_l[candID] = distance_ptr[candID].dist_;
            }
            LoopPreprocess(IP_l.data(), userID);
        }

        void
        LoopPreprocess(const double *distance_ptr, const int &userID) {

            double upper_bound = distance_ptr[0] + 0.01;
            double lower_bound = distance_ptr[topt_ - 1] - 0.01;
            const std::pair<double, double> &bound_pair = std::make_pair(lower_bound, upper_bound);

            double lb = bound_pair.first;
            double ub = bound_pair.second;
            user_ip_bound_l_[userID] = std::make_pair(lb, ub);
            double interval_distance = (ub - lb) / n_interval_;
            interval_dist_l_[userID] = interval_distance;

            int *interval_ptr = interval_table_.get() + userID * n_interval_;
            for (int candID = 0; candID < topt_; candID++) {
                double ip = distance_ptr[candID];
                int itv_idx = std::floor((ub - ip) / interval_distance);
                assert(0 <= itv_idx && itv_idx < n_interval_);
                interval_ptr[itv_idx]++;
            }
            for (int intervalID = 1; intervalID < n_interval_; intervalID++) {
                interval_ptr[intervalID] += interval_ptr[intervalID - 1];
            }
            assert(interval_ptr[n_interval_ - 1] == topt_);

        }

        void GetItvID(const DistancePair *distance_ptr, const int &userID,
                      std::vector<unsigned char> &itvID_l) const {
            if (n_interval_ > 256) {
                spdlog::error("the number of interval larger than 256, program exit");
                exit(-1);
            }
            assert(itvID_l.size() == n_data_item_);
            for (int candID = 0; candID < n_data_item_; candID++) {
                assert(user_ip_bound_l_[userID].first <= user_ip_bound_l_[userID].second);
                const double IP_ub = user_ip_bound_l_[userID].second;
                const double itv_dist = interval_dist_l_[userID];
                const int itemID = distance_ptr[candID].ID_;
                const double ip = distance_ptr[candID].dist_;
                assert(ip <= IP_ub);
                const unsigned char itvID = std::floor((IP_ub - ip) / itv_dist);
                if (itvID >= n_interval_) {
                    itvID_l[itemID] = n_interval_;
                } else {
                    itvID_l[itemID] = itvID;
                }
                assert(0 <= itvID);
            }

        }

        //convert ip_bound to rank_bound
        void RankBound(const std::vector<double> &queryIP_l, const std::vector<bool> &prune_l,
                       const int &topk,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l) {

            assert(prune_l.size() == n_user_);
            assert(rank_lb_l.size() == n_user_ && rank_ub_l.size() == n_user_);
            assert(queryIP_l.size() == n_user_);

            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID]) {
                    continue;
                }

                const double queryIP = queryIP_l[userID];
                std::pair<double, double> user_IPbound = user_ip_bound_l_[userID];
                const double user_IP_ub = user_IPbound.second;
                const double itv_dist = interval_dist_l_[userID];
                const int itvID = std::floor((user_IP_ub - queryIP) / itv_dist);
                if (itvID < 0) {
                    assert(rank_ub_l[userID] <= 0 && 0 <= rank_lb_l[userID]);
                    rank_ub_l[userID] = 0;
                    rank_lb_l[userID] = 0;
                    continue;
                } else if (itvID >= n_interval_) {
                    rank_ub_l[userID] = topt_;
                    rank_lb_l[userID] = n_data_item_;
                    continue;
                }

                int bkt_rank_ub = itvID == 0 ? 0 : interval_table_[userID * n_interval_ + itvID - 1];
                int bkt_rank_lb = interval_table_[userID * n_interval_ + itvID];
                rank_lb_l[userID] = bkt_rank_lb;
                rank_ub_l[userID] = bkt_rank_ub;
            }
        }

        //convert ip_bound to rank_bound
        void RankBound(const std::vector<double> &queryIP_l, const std::vector<bool> &prune_l,
                       const int &topk,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l,
                       std::vector<std::pair<double, double>> &queryIPBound_l,
                       std::vector<int> &itvID_l) {

            assert(prune_l.size() == n_user_);
            assert(rank_lb_l.size() == n_user_ && rank_ub_l.size() == n_user_);
            assert(queryIP_l.size() == n_user_);
            assert(queryIPBound_l.size() == n_user_);

            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID]) {
                    continue;
                }

                const double queryIP = queryIP_l[userID];
                std::pair<double, double> user_IPbound = user_ip_bound_l_[userID];
                const double user_IP_ub = user_IPbound.second;
                const double itv_dist = interval_dist_l_[userID];
                const int itvID = std::floor((user_IP_ub - queryIP) / itv_dist);
                itvID_l[userID] = itvID;
                if (itvID < 0) {
                    assert(rank_ub_l[userID] <= 0 && 0 <= rank_lb_l[userID]);
                    rank_ub_l[userID] = 0;
                    rank_lb_l[userID] = 0;
                    queryIPBound_l[userID] = std::make_pair(queryIP, queryIP);
                    continue;
                } else if (itvID >= n_interval_) {
                    rank_ub_l[userID] = topt_;
                    rank_lb_l[userID] = n_data_item_;
                    queryIPBound_l[userID] = std::make_pair(-DBL_MAX, user_IPbound.first);
                    continue;
                }

                int bkt_rank_ub = itvID == 0 ? 0 : interval_table_[userID * n_interval_ + itvID - 1];
                int bkt_rank_lb = interval_table_[userID * n_interval_ + itvID];
                rank_lb_l[userID] = bkt_rank_lb;
                rank_ub_l[userID] = bkt_rank_ub;

                const double itv_IP_lb = user_IP_ub - (itvID + 1) * itv_dist;
                const double itv_IP_ub = user_IP_ub - itvID * itv_dist;
                queryIPBound_l[userID] = std::make_pair(itv_IP_lb, itv_IP_ub);
            }
        }

    };
}
#endif //REVERSE_K_RANKS_SCORESEARCHTOPT_HPP
