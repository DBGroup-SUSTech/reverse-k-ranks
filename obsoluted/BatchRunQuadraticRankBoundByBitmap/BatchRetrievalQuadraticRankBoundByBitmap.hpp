//
// Created by BianZheng on 2022/7/23.
//

#ifndef REVERSE_KRANKS_BATCHRETRIEVALQUADRATICRANKBOUNDBYBITMAP_HPP
#define REVERSE_KRANKS_BATCHRETRIEVALQUADRATICRANKBOUNDBYBITMAP_HPP

#include "struct/MethodBase.hpp"
#include "BatchBuildIndexQuadraticRankBoundByBitmap.hpp"
#include "../QuadraticRankBoundByBitmap/SSMergeQuadraticRankBoundByBitmap.hpp"
#include "util/FileIO.hpp"
#include "struct/UserRankElement.hpp"

#include <vector>

namespace ReverseMIPS {

    std::unique_ptr<BaseIndex> BuildIndex(
            const std::string &method_name,
            const char *disk_index_path, const char *disk_memory_index_path, const char *score_search_index_path,
            const int &memory_n_sample, const int &disk_n_sample, const uint64_t &index_size_gb,
            VectorMatrix &user, VectorMatrix &data_item) {

        const int n_user = user.n_vector_;
        const int n_data_item = data_item.n_vector_;
        const int vec_dim = user.vec_dim_;

        if (method_name != "SSMergeQuadraticRankBoundByBitmapBatchRun") {
            spdlog::error("not support batch run retrieval method, program exit");
            exit(-1);
        }

        std::unique_ptr<BaseIndex> index;

        //rank search
        ScoreSearch rank_bound_ins(score_search_index_path);

        int n_merge_user;
        QuadraticRankBoundByBitmapParameter(n_data_item, n_user, disk_n_sample, index_size_gb, n_merge_user);

        //disk index
        MergeQuadraticRankBoundByBitmap disk_ins(user, n_data_item, disk_index_path, disk_n_sample, n_merge_user);
        disk_ins.LoadMemoryIndex(disk_memory_index_path);

        index = std::make_unique<SSMergeQuadraticRankBoundByBitmap::Index>(
                //score search
                rank_bound_ins,
                //disk index
                disk_ins,
                //general retrieval
                user, data_item);

        return index;
    }

    void RunRetrieval(
            //index storage info
            const char *disk_index_path, const char *disk_memory_index_path, const char *score_search_index_path,
            const int &memory_n_sample, const int &disk_n_sample, const uint64_t &index_size_gb,
            //index result info
            const char *basic_dir, const std::string &dataset_name, const std::string &method_name) {

        //search on TopTIP
        int n_data_item, n_query_item, n_user, vec_dim;
        std::vector<VectorMatrix> data = readData(basic_dir, dataset_name.c_str(),
                                                  n_data_item, n_query_item, n_user, vec_dim);
        VectorMatrix &user = data[0];
        VectorMatrix &data_item = data[1];
        VectorMatrix &query_item = data[2];
        user.vectorNormalize();

        spdlog::info("{} dataset_name {} start", method_name, dataset_name);

        TimeRecord record;

        std::unique_ptr<BaseIndex> index = BuildIndex(
                method_name,
                disk_index_path, disk_memory_index_path, score_search_index_path,
                memory_n_sample, disk_n_sample, index_size_gb,
                user, data_item);

        std::vector<int> topk_l{70, 60, 50, 40, 30, 20, 10};
//        std::vector<int> topk_l{30, 20, 10};

        char parameter_name[256];
        sprintf(parameter_name, "n_sample_%d-index_size_gb_%ld",
                memory_n_sample, index_size_gb);

        RetrievalResult config;
        std::vector<std::vector<std::vector<UserRankElement>>> result_rank_l;
        for (int topk: topk_l) {
            record.reset();
            std::vector<std::vector<UserRankElement>> result_rk = index->Retrieval(query_item, topk, n_query_item);

            double retrieval_time = record.get_elapsed_time_second();
            double ms_per_query = retrieval_time / n_query_item * 1000;

            std::string performance_str = index->PerformanceStatistics(topk, retrieval_time, ms_per_query);
            config.AddRetrievalInfo(performance_str, topk, retrieval_time, ms_per_query);

            result_rank_l.emplace_back(result_rk);
            spdlog::info("finish top-{}", topk);
            spdlog::info("{}", performance_str);
        }

        int n_topk = (int) topk_l.size();

        for (int i = 0; i < n_topk; i++) {
            WriteRankResult(result_rank_l[i], dataset_name.c_str(), method_name.c_str(), parameter_name);
        }

        config.AddBuildIndexInfo(index->BuildIndexStatistics());
        config.WritePerformance(dataset_name.c_str(), method_name.c_str(), parameter_name);
    }
}
#endif //REVERSE_KRANKS_BATCHRETRIEVALQUADRATICRANKBOUNDBYBITMAP_HPP
