//
// Created by BianZheng on 2022/8/17.
//

#ifndef REVERSE_K_RANKS_KTHUSERRANK_HPP
#define REVERSE_K_RANKS_KTHUSERRANK_HPP

#include "FileIO.hpp"
#include "ComputeItemScore.hpp"
#include "ReadScoreTable.hpp"

#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <queue>

namespace ReverseMIPS {

    void SampleItem(const int &n_data_item, const int64_t &n_sample_item, std::vector<int> &sample_itemID_l) {
        assert(sample_itemID_l.size() == n_sample_item);
        std::vector<int> shuffle_item_idx_l(n_data_item);
        std::iota(shuffle_item_idx_l.begin(), shuffle_item_idx_l.end(), 0);

//        std::random_device rd;
//        std::mt19937 g(rd());
//        std::shuffle(shuffle_item_idx_l.begin(), shuffle_item_idx_l.end(), g);

        std::random_device rd;
        std::seed_seq seed{rd(), rd(), rd(), rd(), rd(), rd(), rd()};
        std::mt19937 eng(0);
        std::shuffle(shuffle_item_idx_l.begin(), shuffle_item_idx_l.end(), eng);

        for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
            sample_itemID_l[sampleID] = shuffle_item_idx_l[sampleID];
        }
    }

    void ComputeQueryRankBySample(const VectorMatrix &user, const VectorMatrix &data_item,
                                  const std::vector<int> &sample_itemID_l, const int64_t &n_sample_item,
                                  std::vector<int> &accu_n_user_rank_l, const char *index_path) {
        assert(sample_itemID_l.size() == n_sample_item);

        const int n_user = user.n_vector_;
        const int n_data_item = data_item.n_vector_;

        const int64_t n_sample_item_size_t = n_sample_item;
        int64_t accu_n_element = n_sample_item_size_t * (n_data_item + 1);
        assert(accu_n_user_rank_l.size() == accu_n_element);
        accu_n_user_rank_l.assign(accu_n_element, 0);

        ReadScoreTable rst(n_user, n_data_item, index_path);
        rst.ReadPreprocess();

        ComputeItemScore cis(user, data_item);

        TimeRecord record;
        record.reset();

        std::vector<float> itemIP_l(n_data_item);
        std::vector<float> sample_item_score_l(n_sample_item);
        for (int userID = 0; userID < n_user; userID++) {
            rst.ReadDisk(userID, itemIP_l);
            cis.ComputeItems(sample_itemID_l.data(), (int) n_sample_item, userID, sample_item_score_l.data());

            //compute the rank of each sampled item
#pragma omp parallel for default(none) shared(n_sample_item, sample_item_score_l, accu_n_user_rank_l, n_data_item, itemIP_l)
            for (int64_t sampleID = 0; sampleID < n_sample_item; sampleID++) {
                const float sampleIP = sample_item_score_l[sampleID];
                float *rank_ptr = std::lower_bound(itemIP_l.data(), itemIP_l.data() + n_data_item, sampleIP,
                                                   [](const float &arrIP, float queryIP) {
                                                       return arrIP > queryIP;
                                                   });
                const int64_t rank = rank_ptr - itemIP_l.data();
                assert(0 <= rank && rank <= n_data_item);

                accu_n_user_rank_l[sampleID * (n_data_item + 1) + rank]++;
            }

            if (userID != 0 && userID % 100000 == 0) {
                spdlog::info(
                        "ComputeQueryRank {:.1f}%, Mem: {} Mb, {:.2f} s/iter",
                        userID / (0.01 * n_user), get_current_RSS() / 1000000,
                        record.get_elapsed_time_second());
                record.reset();

            }

        }
        rst.FinishRead();

        for (int64_t sampleID = 0; sampleID < n_sample_item; sampleID++) {
            const int n_rank = n_data_item + 1;
            for (int rank = 1; rank < n_rank; rank++) {
                accu_n_user_rank_l[sampleID * n_rank + rank] += accu_n_user_rank_l[
                        sampleID * n_rank + rank - 1];
            }
            assert(accu_n_user_rank_l[(sampleID + 1) * n_rank - 1] == n_user);
        }

    }

    void ComputeSortKthRank(const std::vector<int> &accu_n_user_rank_l, const int &n_data_item,
                            const int64_t &n_sample_item, const int &sample_topk,
                            std::vector<int> &sort_kth_rank_l, std::vector<int> &sort_sampleID_l) {

        assert(sort_kth_rank_l.size() == n_sample_item);
        assert(sort_sampleID_l.size() == n_sample_item);
        assert(accu_n_user_rank_l.size() == n_sample_item * (n_data_item + 1));
        assert(0 <= sample_topk);

        std::vector<int> kth_rank_l(n_sample_item);
        for (uint64_t sampleID = 0; sampleID < n_sample_item; sampleID++) {
            //binary search
            const int *user_rank_ptr = accu_n_user_rank_l.data() + sampleID * (n_data_item + 1);
            const int *rank_ptr = std::lower_bound(user_rank_ptr, user_rank_ptr + (n_data_item + 1), sample_topk,
                                                   [](const int &arr_n_rank, const int &topk) {
                                                       return arr_n_rank < topk;
                                                   });
            const int64_t kth_rank = rank_ptr - user_rank_ptr;
            assert(0 <= kth_rank && kth_rank <= n_data_item + 1);
            kth_rank_l[sampleID] = (int) kth_rank;
        }

        std::iota(sort_sampleID_l.begin(), sort_sampleID_l.end(), 0);
        std::sort(sort_sampleID_l.begin(), sort_sampleID_l.end(),
                  [&kth_rank_l](int i1, int i2) { return kth_rank_l[i1] < kth_rank_l[i2]; });

        for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
            sort_kth_rank_l[sampleID] = kth_rank_l[sort_sampleID_l[sampleID]];
        }
        assert(std::is_sorted(sort_kth_rank_l.begin(), sort_kth_rank_l.end()));

    }

    void BuildIndexBySample(const VectorMatrix &user, const VectorMatrix &data_item,
                            const char *index_path, const char *dataset_name, const char *index_dir,
                            const int &n_sample_item, const int &sample_topk,
                            double &compute_rank_time, double &store_index_time) {
        TimeRecord record;
        record.reset();
        const int n_data_item = data_item.n_vector_;
        std::vector<int> sample_itemID_l(n_sample_item);
        SampleItem(n_data_item, n_sample_item, sample_itemID_l);

        const size_t size_t_n_sample_item = n_sample_item;
        std::vector<int> accu_n_user_rank_l(size_t_n_sample_item * (n_data_item + 1));
        ComputeQueryRankBySample(user, data_item,
                                 sample_itemID_l, n_sample_item,
                                 accu_n_user_rank_l, index_path);

        //sort the rank in ascending sort, should also influence sample_itemID_l
        std::vector<int> sort_kth_rank_l(n_sample_item);
        std::vector<int> sort_sampleID_l(n_sample_item);
        ComputeSortKthRank(accu_n_user_rank_l, n_data_item,
                           n_sample_item, sample_topk,
                           sort_kth_rank_l, sort_sampleID_l);
        compute_rank_time = record.get_elapsed_time_second();

//    for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
//        printf("%d ", sort_kth_rank_l[sampleID]);
//    }
//    printf("\n");
//
//    for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
//        const int sort_sampleID = sort_sampleID_l[sampleID];
//        for (int i = 0; i < 30; i++) {
//            printf("%5d ", accu_n_user_rank_l[sort_sampleID * (n_data_item + 1) + i]);
//        }
//        printf("\n");
//    }

        record.reset();
        WriteDistributionBelowTopk(sample_itemID_l, sort_kth_rank_l,
                                   sort_sampleID_l, accu_n_user_rank_l,
                                   n_data_item, n_sample_item, sample_topk,
                                   dataset_name, index_dir);
        store_index_time = record.get_elapsed_time_second();
    }

}
#endif //REVERSE_K_RANKS_KTHUSERRANK_HPP
