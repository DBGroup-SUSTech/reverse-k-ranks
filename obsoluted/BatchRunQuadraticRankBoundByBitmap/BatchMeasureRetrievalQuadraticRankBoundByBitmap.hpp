//
// Created by BianZheng on 2022/7/25.
//

#ifndef REVERSE_K_RANKS_BATCHMEASURERETRIEVALQUADRATICRANKBOUNDBYBITMAP_HPP
#define REVERSE_K_RANKS_BATCHMEASURERETRIEVALQUADRATICRANKBOUNDBYBITMAP_HPP

#include "BatchBuildIndexQuadraticRankBoundByBitmap.hpp"
#include "BatchRun/MeasureDiskIndex/BaseMeasureIndex.hpp"
#include "BatchBuildIndexQuadraticRankBoundByBitmap.hpp"
#include "MeasureQuadraticRankBoundByBitmap.hpp"

#include "alg/RankBoundRefinement/PruneCandidateByBound.hpp"
#include "../ScoreSample/ScoreSearch.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/MethodBase.hpp"
#include "util/TimeMemory.hpp"
#include "util/VectorIO.hpp"
#include "util/FileIO.hpp"
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <spdlog/spdlog.h>
#include <filesystem>

namespace ReverseMIPS::BatchMeasureRetrievalQuadraticRankBoundByBitmap {

    std::unique_ptr<BaseMeasureIndex> BuildIndex(
            const std::string &method_name,
            const char *disk_index_path, const char *disk_memory_index_path, const char *memory_index_path,
            const int &memory_n_sample, const int &disk_n_sample, const uint64_t &index_size_gb,
            VectorMatrix &user, VectorMatrix &data_item) {

        const int n_user = user.n_vector_;
        const int n_data_item = data_item.n_vector_;
        const int vec_dim = user.vec_dim_;

        if (method_name != "MeasureScoreSampleQuadraticRankBoundByBitmap") {
            spdlog::error("not support measure disk index method name, program exit");
            exit(-1);
        }

        std::unique_ptr<BaseMeasureIndex> index;

        //rank search
        ScoreSearch rank_bound_ins(memory_index_path);

        int n_merge_user;
        QuadraticRankBoundByBitmapParameter(n_data_item, n_user, disk_n_sample, index_size_gb, n_merge_user);

        //disk index
        MeasureQuadraticRankBoundByBitmap::MergeQuadraticRankBoundByBitmap disk_ins(user, n_data_item, disk_index_path,
                                                                                    disk_n_sample, n_merge_user);
        disk_ins.LoadMemoryIndex(disk_memory_index_path);

        index = std::make_unique<MeasureQuadraticRankBoundByBitmap::Index>(
                //score search
                rank_bound_ins,
                //disk index
                disk_ins,
                //general retrieval
                user, data_item);

        return index;
    }


    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    void MeasureQuadraticRankBoundByBitmap(const char *disk_index_path, const char *disk_memory_index_path,
                                           const char *memory_index_path,
                                           const int &memory_n_sample, const int &disk_n_sample,
                                           const uint64_t &index_size_gb,
                                           const char *basic_dir, const char *dataset_name, const char *method_name,
                                           const int& n_eval_query) {
        //search on TopTIP
        int n_data_item, n_query_item, n_user, vec_dim;
        std::vector<VectorMatrix> data = readData(basic_dir, dataset_name,
                                                  n_data_item, n_query_item, n_user, vec_dim);
        VectorMatrix &user = data[0];
        VectorMatrix &data_item = data[1];
        VectorMatrix &query_item = data[2];
        user.vectorNormalize();

        spdlog::info("{} dataset_name {} start", method_name, dataset_name);

        std::unique_ptr<BaseMeasureIndex> index = BuildIndex(
                method_name,
                disk_index_path, disk_memory_index_path, memory_index_path,
                memory_n_sample, disk_n_sample, index_size_gb,
                user, data_item);

        std::vector<int> topk_l{70, 60, 50, 40, 30, 20, 10};
//        std::vector<int> topk_l{30, 20, 10};

        char parameter_name[256];
        sprintf(parameter_name, "n_sample_%d-index_size_gb_%ld",
                memory_n_sample, index_size_gb);

        RetrievalResult config;
        TimeRecord record;
        for (int topk: topk_l) {
            record.reset();
            index->Retrieval(query_item, topk, n_eval_query);

            double retrieval_time = record.get_elapsed_time_second();
            double ms_per_query = retrieval_time / n_query_item * 1000;

            std::string performance_str = index->PerformanceStatistics(topk, retrieval_time, ms_per_query);
            config.AddRetrievalInfo(performance_str, topk, retrieval_time, ms_per_query);

            spdlog::info("finish top-{}", topk);
            spdlog::info("{}", performance_str);
        }

        config.AddQueryInfo(n_eval_query);
        config.AddBuildIndexInfo(index->BuildIndexStatistics());
        config.WritePerformance(dataset_name, method_name, parameter_name);
    }

}

#endif //REVERSE_K_RANKS_BATCHMEASURERETRIEVALQUADRATICRANKBOUNDBYBITMAP_HPP
