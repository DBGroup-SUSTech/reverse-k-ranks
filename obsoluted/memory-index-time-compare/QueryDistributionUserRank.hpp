//
// Created by BianZheng on 2022/8/17.
//

#ifndef REVERSE_K_RANKS_KTHUSERRANK_HPP
#define REVERSE_K_RANKS_KTHUSERRANK_HPP

#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <queue>

namespace ReverseMIPS {
    void SampleItem(const int &n_data_item, const int &n_sample_item, std::vector<int> &sample_itemID_l) {
        assert(sample_itemID_l.size() == n_sample_item);
        std::vector<int> shuffle_item_idx_l(n_data_item);
        std::iota(shuffle_item_idx_l.begin(), shuffle_item_idx_l.end(), 0);

//        std::shuffle(shuffle_item_idx_l.begin(), shuffle_item_idx_l.end(), std::default_random_engine(100));

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(shuffle_item_idx_l.begin(), shuffle_item_idx_l.end(), g);

        for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
            sample_itemID_l[sampleID] = shuffle_item_idx_l[sampleID];
        }
    }

    void ComputeSortUserRank(const VectorMatrix &user, const VectorMatrix &data_item,
                             const std::vector<int> &sample_itemID_l, const int &n_sample_item,
                             std::vector<int> &sorted_user_rank_l,
                             double &compute_score_table_time) {

        compute_score_table_time = 0;
        const int n_data_item = data_item.n_vector_;
        const int n_user = user.n_vector_;

        assert(sorted_user_rank_l.size() == n_sample_item * n_user);

        ComputeItemIDScoreTable cst(user, data_item);
        std::vector<double> distance_l(n_data_item);
        std::vector<double> sample_itemIP_l(n_sample_item);

        TimeRecord record;
        record.reset();

        TimeRecord compute_record;

        for (int userID = 0; userID < n_user; userID++) {
            compute_record.reset();
            cst.ComputeItems(userID, distance_l.data());
            compute_score_table_time += compute_record.get_elapsed_time_second();

            for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
                const int sampleItemID = sample_itemID_l[sampleID];
                sample_itemIP_l[sampleID] = distance_l[sampleItemID];
            }

            compute_record.reset();
            cst.SortItems(userID, distance_l.data());
            compute_score_table_time += compute_record.get_elapsed_time_second();

            for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
                const double itemIP = sample_itemIP_l[sampleID];
                const double *distance_ptr = distance_l.data();

                const double *lb_ptr = std::lower_bound(distance_ptr, distance_ptr + n_data_item, itemIP,
                                                        [](const double &arrIP, double queryIP) {
                                                            return arrIP > queryIP;
                                                        });
                const long rank = lb_ptr - distance_ptr;
                sorted_user_rank_l[sampleID * n_user + userID] = (int) rank;
            }

            if (userID % cst.report_every_ == 0 && userID != 0) {
                spdlog::info(
                        "Compute second score table {:.2f}%, {:.2f} s/iter, Mem: {} Mb, Compute Score Time {}s, Sort Score Time {}s",
                        userID / (0.01 * n_user), record.get_elapsed_time_second(), get_current_RSS() / (1024 * 1024),
                        cst.compute_time_, cst.sort_time_);
                cst.compute_time_ = 0;
                cst.sort_time_ = 0;
                record.reset();
            }
        }

        for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
            std::sort(sorted_user_rank_l.begin() + sampleID * n_user,
                      sorted_user_rank_l.begin() + (sampleID + 1) * n_user);
        }

        for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
            for (int userID = 1; userID < n_user; userID++) {
                assert(sorted_user_rank_l[sampleID * n_user + userID] >=
                       sorted_user_rank_l[sampleID * n_user + userID - 1]);
            }
        }
    }

}
#endif //REVERSE_K_RANKS_KTHUSERRANK_HPP
