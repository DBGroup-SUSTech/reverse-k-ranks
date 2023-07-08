//
// Created by BianZheng on 2022/7/16.
//

#ifndef REVERSE_K_RANKS_BATCHRETRIEVALTOPT_HPP
#define REVERSE_K_RANKS_BATCHRETRIEVALTOPT_HPP

#include "struct/MethodBase.hpp"
#include "../TopT/RSTopTIP.hpp"
#include "util/FileIO.hpp"
#include "struct/UserRankElement.hpp"

#include <vector>

namespace ReverseMIPS {

    std::unique_ptr<BaseIndex> BuildIndex(
            const std::string &method_name,
            const char *disk_topt_index_path, const char *score_search_index_path,
            const int &n_sample, const uint64_t &index_size_gb,
            VectorMatrix &user, VectorMatrix &data_item) {

        const int n_user = user.n_vector_;
        const int n_data_item = data_item.n_vector_;
        const int vec_dim = user.vec_dim_;

        std::unique_ptr<BaseIndex> index;

        //rank search
        RankSearch rank_bound_ins(score_search_index_path);

        int topt;
        TopTIPParameter(n_data_item, n_user, index_size_gb, topt);
        //disk index
        TopTIP disk_ins(n_user, n_data_item, vec_dim, disk_topt_index_path, topt);

        index = std::make_unique<RSCompressTopTIPBruteForce::Index>(
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
            const char *disk_index_path, const char *memory_index_path,
            const int &n_sample, const int &index_size_gb,
            //index result info
            const char *basic_dir, const std::string &dataset_name) {

        //search on TopTIP
        int n_data_item, n_query_item, n_user, vec_dim;
        std::vector<VectorMatrix> data = readData(basic_dir, dataset_name.c_str(),
                                                  n_data_item, n_query_item, n_user, vec_dim);
        VectorMatrix &user = data[0];
        VectorMatrix &data_item = data[1];
        VectorMatrix &query_item = data[2];
        user.vectorNormalize();

        const std::string method_name = "RSCompressTopTIPBruteForceBatchRun";

        spdlog::info("{} dataset_name {} start", method_name, dataset_name);

        TimeRecord record;

        std::unique_ptr<BaseIndex> index = BuildIndex(
                method_name,
                disk_index_path, memory_index_path,
                n_sample, index_size_gb,
                user, data_item);

//        std::vector<int> warmup_topk_l{10, 10};
//        for (int topk: warmup_topk_l) {
//            record.reset();
//            index->Retrieval(query_item, topk, 10);
//
//            double retrieval_time = record.get_elapsed_time_second();
//            double ms_per_query = retrieval_time / n_query_item * 1000;
//
//            std::string performance_str = index->PerformanceStatistics(topk, retrieval_time, ms_per_query);
//            spdlog::info("finish warmup top-{}", topk);
//            spdlog::info("{}", performance_str);
//        }

        std::vector<int> topk_l{50, 40, 30, 20, 10};
//        std::vector<int> topk_l{10};

        const int n_eval_query = query_item.n_vector_;

        char parameter_name[256];
        sprintf(parameter_name, "n_sample_%d-index_size_gb_%d",
                n_sample, index_size_gb);

        RetrievalResult config;
        std::vector<std::vector<std::vector<UserRankElement>>> result_rank_l;
        for (int topk: topk_l) {
            record.reset();
            std::vector<std::vector<UserRankElement>> result_rk = index->Retrieval(query_item, topk, n_eval_query);

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
            std::cout << config.GetConfig(i) << std::endl;
            WriteRankResult(result_rank_l[i], dataset_name.c_str(), method_name.c_str(), parameter_name);
        }

        config.AddBuildIndexInfo(index->BuildIndexStatistics());
        config.AddExecuteQuery(n_eval_query);
        config.WritePerformance(dataset_name.c_str(), method_name.c_str(), parameter_name);
    }
}
#endif //REVERSE_K_RANKS_BATCHRETRIEVALTOPT_HPP
