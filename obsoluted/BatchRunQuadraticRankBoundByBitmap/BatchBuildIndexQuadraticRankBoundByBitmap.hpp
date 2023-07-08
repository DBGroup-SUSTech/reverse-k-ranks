//
// Created by BianZheng on 2022/7/23.
//

#ifndef REVERSE_KRANKS_BATCHBUILDINDEXQUADRATICRANKBOUNDBYBITMAP_HPP
#define REVERSE_KRANKS_BATCHBUILDINDEXQUADRATICRANKBOUNDBYBITMAP_HPP

#include "../QuadraticRankBoundByBitmap/MergeQuadraticRankBoundByBitmap.hpp"
#include "alg/RankBoundRefinement/RankSearch.hpp"
#include "score_computation/ComputeScoreTable.hpp"
#include "util/TimeMemory.hpp"
#include "util/VectorIO.hpp"

namespace ReverseMIPS {
    void QuadraticRankBoundByBitmapParameter(const int &n_data_item, const int &n_user, const int &n_sample,
                                             const uint64_t &index_size_gb,
                                             int &n_merge_user) {
        const int bitmap_size_byte = (n_data_item / 8 + (n_data_item % 8 == 0 ? 0 : 1)) * sizeof(unsigned char);
        const int n_rank_bound = n_sample;

        const uint64_t index_size_byte = (uint64_t) index_size_gb * 1024 * 1024 * 1024;
        const uint64_t predict_index_size_byte = (uint64_t) bitmap_size_byte * n_rank_bound * n_user;
        const uint64_t n_merge_user_big_size = index_size_byte / (bitmap_size_byte * n_rank_bound);
        n_merge_user = int(n_merge_user_big_size);
        if (index_size_byte >= predict_index_size_byte) {
            spdlog::info("index size larger than the whole score table, use whole table setting");
            n_merge_user = n_user - 1;
//            n_merge_user = n_user / 2;
        }
    }

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    void
    BuildIndex(const char *basic_dir, const char *dataset_name, const int &index_size_gb, const int &disk_n_sample) {
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
                "input parameter: n_sample: {128, 512}, index_size_gb {256}, method_name MergeQuadraticRankBoundByBitmap");

        char bitmap256_path[256];
        sprintf(bitmap256_path, "../index/%s_QuadraticRankBoundByBitmap%d_n_sample_%d.index",
                dataset_name, index_size_gb, disk_n_sample);
        char bitmap256_memory_path[256];
        sprintf(bitmap256_memory_path, "../index/%s_QuadraticRankBoundByBitmap%d_n_sample_%d_memory.index",
                dataset_name, index_size_gb, disk_n_sample);

        int n_merge_user256;
        QuadraticRankBoundByBitmapParameter(n_data_item, n_user, disk_n_sample, index_size_gb, n_merge_user256);
        MergeQuadraticRankBoundByBitmap bitmap256_ins(user, n_data_item, bitmap256_path, disk_n_sample,
                                                      n_merge_user256);
        bitmap256_ins.BuildIndexPreprocess(user);
        bitmap256_ins.PreprocessData(user, data_item);
        std::vector<std::vector<int>> &eval_seq_l = bitmap256_ins.BuildIndexMergeUser();
        assert(eval_seq_l.size() == n_merge_user256);

        if (bitmap256_ins.exact_rank_ins_.method_name != "BaseIPBound") {
            spdlog::error("QuadraticRankBoundByBitmap, its exact_rank_ins_ should all be BaseIPBound");
            exit(-1);
        }
        //do not have PreprocessData since only applicable for BaseIPBound

        const int rank_sample_n_sample1 = 128;
        const int rank_sample_n_sample2 = 512;
        char rank_sample128_path[256];
        sprintf(rank_sample128_path, "../index/%s_RankSearch%d.index", dataset_name, rank_sample_n_sample1);
        char rank_sample512_path[256];
        sprintf(rank_sample512_path, "../index/%s_RankSearch%d.index", dataset_name, rank_sample_n_sample2);

        //rank search
        RankSearch rs_128(rank_sample_n_sample1, n_data_item, n_user, n_data_item);
        RankSearch rs_512(rank_sample_n_sample2, n_data_item, n_user, n_data_item);

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
                rs_128.LoopPreprocess(distance_pair_l.data(), userID);
                rs_512.LoopPreprocess(distance_pair_l.data(), userID);
                memory_index_time += component_record.get_elapsed_time_second();

                component_record.reset();
                bitmap256_ins.BuildIndexLoop(distance_pair_l.data(), userID);
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
            bitmap256_ins.WriteIndex();
        }

        cst.FinishCompute();
        bitmap256_ins.FinishBuildIndex();
        bitmap256_ins.SaveMemoryIndex(bitmap256_memory_path);

        rs_128.SaveIndex(rank_sample128_path);
        rs_512.SaveIndex(rank_sample512_path);
    }
}
#endif //REVERSE_KRANKS_BATCHBUILDINDEXQUADRATICRANKBOUNDBYBITMAP_HPP
