//
// Created by BianZheng on 2022/3/18.
//

#ifndef REVERSE_KRANKS_INTERVALSEARCH_HPP
#define REVERSE_KRANKS_INTERVALSEARCH_HPP

#include "struct/DistancePair.hpp"

#include <cassert>
#include <vector>
#include <algorithm>
#include "spdlog/spdlog.h"

namespace ReverseMIPS {

    class ScoreSearch {
    public:

        size_t n_interval_, n_user_, n_data_item_;
        // n_user * n_interval, the last element of an interval column must be n_data_item
        std::unique_ptr<int[]> interval_table_;
        // n_user, stores the distance of interval for each user
        std::unique_ptr<double[]> interval_dist_l_;
        // n_user, bound for column, first is lower bound, second is upper bound
        std::unique_ptr<std::pair<double, double>[]> user_ip_bound_l_;


        inline ScoreSearch() = default;

        inline ScoreSearch(const int &n_interval, const int &n_user, const int n_data_item) {
            this->n_interval_ = n_interval;
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;

            interval_table_ = std::make_unique<int[]>(n_user_ * n_interval_);
            std::memset(interval_table_.get(), 0, n_user_ * n_interval_ * sizeof(int));

            interval_dist_l_ = std::make_unique<double[]>(n_user_);
            user_ip_bound_l_ = std::make_unique<std::pair<double, double>[]>(n_user_);
            spdlog::info("interval bound: n_interval {}", n_interval);
        }

        inline ScoreSearch(const char *index_path) {
            LoadIndex(index_path);
        }

        void LoopPreprocess(const DistancePair *distance_ptr, const int &userID) {
            const double IP_ub = distance_ptr[0].dist_ + 0.01;
            const double IP_lb = distance_ptr[n_data_item_ - 1].dist_ - 0.01;
            const std::pair<double, double> &bound_pair = std::make_pair(IP_lb, IP_ub);
            //lb == IP_lb, ub == IP_ub
            user_ip_bound_l_[userID] = bound_pair;
            const double interval_distance = (IP_ub - IP_lb) / (double) n_interval_;
            interval_dist_l_[userID] = interval_distance;

            int *interval_ptr = interval_table_.get() + userID * n_interval_;
#pragma omp parallel for default(none) shared(IP_ub, interval_distance, interval_ptr, distance_ptr)
            for (int itvID = 0; itvID < n_interval_; itvID++) {
                double itv_IP_lb = IP_ub - (itvID + 1) * interval_distance;
                const DistancePair *iter_begin = distance_ptr;
                const DistancePair *iter_end = distance_ptr + n_data_item_;

                const DistancePair *lb_ptr = std::lower_bound(iter_begin, iter_end, itv_IP_lb,
                                                              [](const DistancePair &arrIP, double queryIP) {
                                                                  return arrIP.dist_ > queryIP;
                                                              });
                const int n_item = (int) (lb_ptr - iter_begin);
                interval_ptr[itvID] = n_item;
            }
            assert(interval_ptr[n_interval_ - 1] == n_data_item_);
        }

        void
        LoopPreprocess(const double *distance_ptr, const int &userID) {
            const double IP_ub = distance_ptr[0] + 0.01;
            const double IP_lb = distance_ptr[n_data_item_ - 1] - 0.01;
            const std::pair<double, double> &bound_pair = std::make_pair(IP_lb, IP_ub);
            //lb == IP_lb, ub == IP_ub
            user_ip_bound_l_[userID] = bound_pair;
            const double interval_distance = (IP_ub - IP_lb) / (double) n_interval_;
            interval_dist_l_[userID] = interval_distance;

            int *interval_ptr = interval_table_.get() + userID * n_interval_;
#pragma omp parallel for default(none) shared(IP_ub, interval_distance, interval_ptr, distance_ptr)
            for (int itvID = 0; itvID < n_interval_; itvID++) {
                double itv_IP_lb = IP_ub - (itvID + 1) * interval_distance;
                const double *iter_begin = distance_ptr;
                const double *iter_end = distance_ptr + n_data_item_;

                const double *lb_ptr = std::lower_bound(iter_begin, iter_end, itv_IP_lb,
                                                        [](const double &arrIP, double queryIP) {
                                                            return arrIP > queryIP;
                                                        });
                const int n_item = (int) (lb_ptr - iter_begin);
                interval_ptr[itvID] = n_item;
            }
            assert(interval_ptr[n_interval_ - 1] == n_data_item_);

        }

        //convert ip_bound to rank_bound
        void RankBound(const std::vector<double> &queryIP_l,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l) {

            assert(rank_lb_l.size() == n_user_ && rank_ub_l.size() == n_user_);
            assert(queryIP_l.size() == n_user_);

            for (int userID = 0; userID < n_user_; userID++) {

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
                    assert(rank_ub_l[userID] <= n_data_item_ && n_data_item_ <= rank_lb_l[userID]);
                    rank_ub_l[userID] = n_data_item_;
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
        void RankBound(const std::vector<double> &queryIP_l,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l,
                       std::vector<std::pair<double, double>> &queryIPBound_l,
                       std::vector<int> &itvID_l) {

            assert(rank_lb_l.size() == n_user_ && rank_ub_l.size() == n_user_);
            assert(queryIP_l.size() == n_user_);
            assert(queryIPBound_l.size() == n_user_);

            for (int userID = 0; userID < n_user_; userID++) {

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
                    continue;
                } else if (itvID >= n_interval_) {
                    assert(rank_ub_l[userID] <= n_data_item_ && n_data_item_ <= rank_lb_l[userID]);
                    rank_ub_l[userID] = n_data_item_;
                    rank_lb_l[userID] = n_data_item_;
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

        void SaveIndex(const char *index_path) {
            std::ofstream out_stream_ = std::ofstream(index_path, std::ios::binary | std::ios::out);
            if (!out_stream_) {
                spdlog::error("error in write result");
                exit(-1);
            }
            out_stream_.write((char *) &n_interval_, sizeof(size_t));
            out_stream_.write((char *) &n_user_, sizeof(size_t));
            out_stream_.write((char *) &n_data_item_, sizeof(size_t));

            out_stream_.write((char *) interval_table_.get(), n_user_ * n_interval_ * sizeof(int));
            out_stream_.write((char *) interval_dist_l_.get(), n_user_ * sizeof(double));
            out_stream_.write((char *) user_ip_bound_l_.get(), n_user_ * sizeof(std::pair<double, double>));

            out_stream_.close();
        }

        void LoadIndex(const char *index_path) {
            std::ifstream index_stream = std::ifstream(index_path, std::ios::binary | std::ios::in);
            if (!index_stream) {
                spdlog::error("error in reading index");
                exit(-1);
            }

            index_stream.read((char *) &n_interval_, sizeof(size_t));
            index_stream.read((char *) &n_user_, sizeof(size_t));
            index_stream.read((char *) &n_data_item_, sizeof(size_t));

            interval_table_ = std::make_unique<int[]>(n_user_ * n_interval_);
            index_stream.read((char *) interval_table_.get(), sizeof(int) * n_user_ * n_interval_);

            interval_dist_l_ = std::make_unique<double[]>(n_user_);
            index_stream.read((char *) interval_dist_l_.get(), sizeof(double) * n_user_);

            user_ip_bound_l_ = std::make_unique<std::pair<double, double>[]>(n_user_);
            index_stream.read((char *) user_ip_bound_l_.get(), sizeof(std::pair<double, double>) * n_user_);

            index_stream.close();
        }

    };
}
#endif //REVERSE_KRANKS_INTERVALSEARCH_HPP
