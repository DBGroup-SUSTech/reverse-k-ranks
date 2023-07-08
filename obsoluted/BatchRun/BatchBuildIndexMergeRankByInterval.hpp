//
// Created by BianZheng on 2022/7/25.
//

#ifndef REVERSE_KRANKS_BATCHBUILDINDEXMERGERANKBYINTERVAL_HPP
#define REVERSE_KRANKS_BATCHBUILDINDEXMERGERANKBYINTERVAL_HPP

#include "../MergeIntervalDiskCompression/MergeRankByInterval.hpp"
#include "../ScoreSample/ScoreSearch.hpp"
#include "score_computation/ComputeScoreTable.hpp"
#include "util/TimeMemory.hpp"
#include "util/VectorIO.hpp"

namespace ReverseMIPS {
    void MergeRankByIntervalParameter(const int &n_data_item, const int &n_user,
                                      const uint64_t &index_size_gb,
                                      int &n_merge_user) {
        const uint64_t index_size_byte = (uint64_t) index_size_gb * 1024 * 1024 * 1024;
        const uint64_t predict_index_size_byte = (uint64_t) (sizeof(int) * 2) * n_data_item * n_user;
        const uint64_t n_merge_user_big_size = index_size_byte / (sizeof(int) * 2) / n_data_item;
        n_merge_user = int(n_merge_user_big_size);
        if (index_size_byte >= predict_index_size_byte) {
            spdlog::info("index size larger than the whole score table, use whole table setting");
            n_merge_user = n_user - 1;
        }
    }

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    void
    BuildIndex(const char *basic_dir, const char *dataset_name, const int &index_size_gb) {
        int n_data_item, n_query_item, n_user, vec_dim;
        std::vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                                  vec_dim);
        VectorMatrix &user = data[0];
        VectorMatrix &data_item = data[1];
        VectorMatrix &query_item = data[2];
        spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user,
                     vec_dim);

        user.vectorNormalize();

        spdlog::info(
                "input parameter: n_sample {512}, index_size_gb {256}, method_name MergeRankByInterval");

        char disk_index_path[256];
        sprintf(disk_index_path, "../index/%s_MergeRankByInterval%d.index",
                dataset_name, index_size_gb);
        char disk_memory_index_path[256];
        sprintf(disk_memory_index_path, "../index/%s_MergeRankByInterval%d_memory.index",
                dataset_name, index_size_gb);

        int n_merge_user256;
        MergeRankByIntervalParameter(n_data_item, n_user, index_size_gb, n_merge_user256);
        MergeRankByInterval disk_index(user, n_data_item, disk_index_path, n_merge_user256);
        disk_index.BuildIndexPreprocess(user);
        disk_index.PreprocessData(user, data_item);
        std::vector<std::vector<int>> &eval_seq_l = disk_index.BuildIndexMergeUser();
        assert(eval_seq_l.size() == n_merge_user256);

        if (disk_index.exact_rank_ins_.method_name != "BaseIPBound") {
            spdlog::error("MergeRankByInterval, its exact_rank_ins_ should all be BaseIPBound");
            exit(-1);
        }
        //do not have PreprocessData since only applicable for BaseIPBound

        const int score_sample_n_sample = 512;
        char score_sample_path[256];
        sprintf(score_sample_path, "../index/%s_ScoreSearch%d.index", dataset_name, score_sample_n_sample);

        //rank search
        ScoreSearch score_search_ins(score_sample_n_sample, n_user, n_data_item);

        //Compute Score Table
        ComputeScoreTable cst(user, data_item);
        std::vector<DistancePair> distance_pair_l(n_data_item);

        TimeRecord record;
        record.reset();
        TimeRecord component_record;

        double memory_index_time = 0;
        double disk_index_time = 0;

        const int report_user_every = 10000;
        int report_count = 0;

        for (int labelID = 0; labelID < n_merge_user256; labelID++) {
            std::vector<int> &user_l = eval_seq_l[labelID];
            const unsigned int n_eval = user_l.size();

            for (int evalID = 0; evalID < n_eval; evalID++) {
                report_count++;
                int userID = user_l[evalID];
                assert(0 <= userID && userID < n_user);
                cst.ComputeSortItems(userID, distance_pair_l.data());

                component_record.reset();
                score_search_ins.LoopPreprocess(distance_pair_l.data(), userID);
                memory_index_time += component_record.get_elapsed_time_second();

                component_record.reset();
                disk_index.BuildIndexLoop(distance_pair_l.data(), userID);
                disk_index_time += component_record.get_elapsed_time_second();

                if (report_count != 0 && report_count % report_user_every == 0) {
                    std::cout << "preprocessed " << report_count / (0.01 * n_user) << " %, "
                              << record.get_elapsed_time_second() << " s/iter" << " Mem: "
                              << get_current_RSS() / 1000000 << " Mb \n";
                    spdlog::info(
                            "Compute Score Time {}s, Sort Score Time {}s, Memory Index Time {}s, Disk Index Time {}s",
                            cst.compute_time_, cst.sort_time_, memory_index_time, disk_index_time);
                    cst.compute_time_ = 0;
                    cst.sort_time_ = 0;
                    memory_index_time = 0;
                    disk_index_time = 0;
                    record.reset();
                }
            }
            disk_index.WriteIndex();
        }

        cst.FinishCompute();
        disk_index.FinishBuildIndex();
        disk_index.SaveMemoryIndex(disk_memory_index_path);

        score_search_ins.SaveIndex(score_sample_path);
    }
}
#endif //REVERSE_KRANKS_BATCHBUILDINDEXMERGERANKBYINTERVAL_HPP
