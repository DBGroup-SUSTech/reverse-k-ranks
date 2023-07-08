//
// Created by BianZheng on 2022/5/17.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "struct/VectorMatrix.hpp"

#include "UserRank.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

void WriteRank(const std::vector<std::vector<int>> &all_rank_l, const char *dataset_name, const int &node_size,
               const int &n_data_item) {

    char resPath[256];
    std::sprintf(resPath, "../../result/attribution/PrintUserRank/print-user-rank-%s.csv", dataset_name);
    std::ofstream file(resPath);
    if (!file) {
        std::printf("error in write result\n");
    }

    const int n_query_item = all_rank_l.size();
    const int n_user = all_rank_l[0].size();

    const int n_bin = std::ceil(1.0 * n_data_item / node_size);

    for (int qID = 0; qID < n_query_item; qID++) {
        std::vector<int> bin_l(n_bin, 0);
        const std::vector<int> user_rank_l = all_rank_l[qID];
        assert(n_user == user_rank_l.size());

        for (int userID = 0; userID < n_user; userID++) {
            const int rank = all_rank_l[qID][userID];
            int binID;
            if (rank == 1) {
                binID = 0;
            } else {
                binID = (rank - 1) / node_size + 1;
                if (binID == n_bin) {
                    binID = n_bin - 1;
                }
            }
            assert(0 <= binID && binID < n_bin);
            bin_l[binID]++;
        }

        for (int binID = 0; binID < n_bin - 1; binID++) {
            file << bin_l[binID] << ",";
        }
        file << bin_l[n_bin - 1] << std::endl;
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
    spdlog::info("PrintUserRank dataset_name {}, basic_dir {}", dataset_name, basic_dir);

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
    unique_ptr<UserRank::Index> index = UserRank::BuildIndex(data_item, user, index_path, node_size);

    double build_index_time = record.get_elapsed_time_second();
    spdlog::info("finish preprocess and save the index");

    record.reset();
    std::vector<double> prune_ratio_l(n_query_item);
    vector<vector<int>> result_rk = index->GetAllRank(query_item, prune_ratio_l);

    double retrieval_time = record.get_elapsed_time_second();
    spdlog::info("build index time: total {}s, retrieval time: total {}s", build_index_time, retrieval_time);

    WriteRank(result_rk, dataset_name, node_size, n_data_item);
    return 0;
}