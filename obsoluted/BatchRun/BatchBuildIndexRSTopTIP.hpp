//
// Created by BianZheng on 2022/7/15.
//

#ifndef REVERSE_KRANKS_BATBUILDINDEXTOPT_HPP
#define REVERSE_KRANKS_BATBUILDINDEXTOPT_HPP

#include "../TopT/TopTIP.hpp"
#include "alg/RankBoundRefinement/RankSearch.hpp"
#include "score_computation/ComputeScoreTable.hpp"
#include "util/TimeMemory.hpp"
#include "util/VectorIO.hpp"

namespace ReverseMIPS {

    void TopTIPParameter(const int &n_data_item, const int &n_user, const uint64_t &index_size_gb,
                         int &topt) {
        //disk index
        const uint64_t index_size_byte = (uint64_t) index_size_gb * 1024 * 1024 * 1024;
        const uint64_t predict_index_size_byte = (uint64_t) sizeof(double) * n_data_item * n_user;
        const uint64_t topt_big_size = index_size_byte / sizeof(double) / n_user;
        topt = int(topt_big_size);
        spdlog::info("TopTIP index size byte: {}, predict index size byte: {}", index_size_byte,
                     predict_index_size_byte);
        if (index_size_byte >= predict_index_size_byte) {
            spdlog::info("index size larger than the whole score table, use whole table setting");
            topt = n_data_item;
        }
    }

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    void BuildIndex(const char *basic_dir, const char *dataset_name) {
        int n_data_item, n_query_item, n_user, vec_dim;
        std::vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                                  vec_dim);
        VectorMatrix &user = data[0];
        VectorMatrix &data_item = data[1];
        VectorMatrix &query_item = data[2];
        spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user,
                     vec_dim);

        user.vectorNormalize();

        spdlog::info("input parameter: n_sample: {128, 512}, index_size_gb {256}, method_name RSTopTIP");

        const int index_size_gb = 256;
        char toptIP256_path[256];
        sprintf(toptIP256_path, "../index/%s_TopTIP256.index", dataset_name);

        user.vectorNormalize();

        //do not have PreprocessData since only applicable for BaseIPBound

        int toptIP256;
        TopTIPParameter(n_data_item, n_user, index_size_gb, toptIP256);
        TopTIP toptIP256_ins(n_user, n_data_item, vec_dim, toptIP256_path, toptIP256);
        toptIP256_ins.BuildIndexPreprocess();

        const int n_sample_128 = 128;
        const int n_sample_512 = 512;
        char rs128_path[256];
        sprintf(rs128_path, "../index/%s_RankSearch128_TopT.index", dataset_name);
        char rs512_path[256];
        sprintf(rs512_path, "../index/%s_RankSearch512_TopT.index", dataset_name);

        //rank search
        RankSearch rs_128(n_sample_128, n_data_item, n_user);
        RankSearch rs_512(n_sample_512, n_data_item, n_user);

        //Compute Score Table
        ComputeScoreTable cst(user, data_item);
        std::vector<DistancePair> distance_pair_l(n_data_item);

        TimeRecord record;
        record.reset();
        TimeRecord component_record;

        double toptIP_time = 0;
        double rank_search_time = 0;
        for (int userID = 0; userID < n_user; userID++) {
            cst.ComputeSortItems(userID, distance_pair_l.data());

            component_record.reset();
            rs_128.LoopPreprocess(distance_pair_l.data(), userID);
            rs_512.LoopPreprocess(distance_pair_l.data(), userID);
            rank_search_time += component_record.get_elapsed_time_second();

            component_record.reset();
            toptIP256_ins.BuildIndexLoop(distance_pair_l.data());
            toptIP_time += component_record.get_elapsed_time_second();

            if (userID != 0 && userID % cst.report_every_ == 0) {
                std::cout << "preprocessed " << userID / (0.01 * n_user) << " %, "
                          << record.get_elapsed_time_second() << " s/iter" << " Mem: "
                          << get_current_RSS() / 1000000 << " Mb \n";
                spdlog::info(
                        "Compute Score Time {}s, Sort Score Time {}s, Rank Search Time {}s, TopTIP Time {}s",
                        cst.compute_time_, cst.sort_time_, rank_search_time, toptIP_time);
                cst.compute_time_ = 0;
                cst.sort_time_ = 0;
                rank_search_time = 0;
                toptIP_time = 0;
                record.reset();
            }
        }
        cst.FinishCompute();
        toptIP256_ins.FinishBuildIndex();

        rs_128.SaveIndex(rs128_path);
        rs_512.SaveIndex(rs512_path);
    }
}
#endif //REVERSE_KRANKS_BATBUILDINDEXTOPT_HPP
