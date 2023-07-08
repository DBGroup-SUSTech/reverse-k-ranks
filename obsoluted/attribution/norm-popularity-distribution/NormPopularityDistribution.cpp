//
// Created by BianZheng on 2022/6/1.
//


#include "struct/VectorMatrix.hpp"
#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"

#include "ExactUserRank.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

void TopPercRankCount(const std::unique_ptr<int[]> &query_rank_l, const std::vector<double> &query_norm_l,
                      const int &n_query, const int &n_user, const int n_data_item, const int &top_perc,
                      std::vector<std::pair<double, int>> &norm_n_rank_l) {
    assert(norm_n_rank_l.size() == query_norm_l.size() && query_norm_l.size() == n_query);
    const int topk = int(n_data_item * 1.0 / 100 * top_perc);
    for (int queryID = 0; queryID < n_query; queryID++) {
        //count how many user rank in the topk
        int topk_count = 0;
        int basic_offset = queryID * n_user;
        for (int userID = 0; userID < n_user; userID++) {
            if (query_rank_l[basic_offset + userID] <= topk) {
                topk_count++;
            }
        }
        const double norm = query_norm_l[queryID];
        norm_n_rank_l[queryID] = std::make_pair(norm, topk_count);
    }

}

void
WriteRank(const std::vector<std::pair<double, int>> &norm_n_rank_l, const int &top_perc, const char *dataset_name) {

    char resPath[256];
    std::sprintf(resPath, "../../result/attribution/norm-popularity-distribution-%s-perc-%d.csv",
                 dataset_name, top_perc);
    std::ofstream file(resPath);
    if (!file) {
        std::printf("error in write result\n");
    }

    const int n_query_item = norm_n_rank_l.size();

    for (int qID = 0; qID < n_query_item; qID++) {
        file << norm_n_rank_l[qID].first << "," << norm_n_rank_l[qID].second << std::endl;
    }

    file.close();
}

using namespace std;
using namespace ReverseMIPS;


int main(int argc, char **argv) {
    if (!(argc == 2 or argc == 3)) {
        cout << argv[0] << " dataset_name [basic_dir]" << endl;
        return 0;
    }
    const char *dataset_name = argv[1];
    const char *basic_dir = "/home/bianzheng/Dataset/ReverseMIPS";
    if (argc == 3) {
        basic_dir = argv[2];
    }
    spdlog::info("NormPopularityDistribution dataset_name {}, basic_dir {}", dataset_name, basic_dir);

    std::vector<int> top_rank_perc_l = {1, 2, 5, 8, 10, 20, 40};

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    char index_path[256];
    sprintf(index_path, "../../index/index");
    const int node_size = 512;

    TimeRecord record;
    record.reset();
    unique_ptr<ExactUserRank::Index> index = ExactUserRank::BuildIndex(data_item, user, index_path, node_size);

    double build_index_time = record.get_elapsed_time_second();
    spdlog::info("finish preprocess and save the index");

    record.reset();
    //query_rank_l: n_query_item * n_user, store the rank of each user, sorted in ascending order
    unique_ptr<int[]> query_rank_l = index->GetAllRank(query_item);

    vector<double> query_norm_l(n_query_item);
    ExactUserRank::CalcNorm(query_item, query_norm_l);

    double retrieval_time = record.get_elapsed_time_second();
    spdlog::info("build index time: total {}s, retrieval time: total {}s", build_index_time, retrieval_time);

    for (const int &top_perc: top_rank_perc_l) {
        std::vector<std::pair<double, int>> norm_n_rank_l(n_query_item);
        TopPercRankCount(query_rank_l, query_norm_l,
                         n_query_item, n_user, n_data_item, top_perc,
                         norm_n_rank_l);

        WriteRank(norm_n_rank_l, top_perc, dataset_name);
    }

    return 0;
}