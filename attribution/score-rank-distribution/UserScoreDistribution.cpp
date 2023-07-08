//
// Created by BianZheng on 2022/5/23.
//

#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"
#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

using namespace ReverseMIPS;

std::vector<std::vector<float>> &
ScoreDistribution(const VectorMatrix &user, const VectorMatrix &data_item, const int n_sample_user) {
    //generate the number sequentially, then shuffle
    const int n_user = user.n_vector_;
    std::vector<int> user_idx_l(n_user);
    std::iota(user_idx_l.begin(), user_idx_l.end(), 0);

    std::random_device rd;
    std::mt19937 random_gen(rd());
    std::shuffle(user_idx_l.begin(), user_idx_l.end(), random_gen);

    const int n_data_item = data_item.n_vector_;
    const int vec_dim = user.vec_dim_;
    static std::vector<std::vector<float>> score_dist_l(n_sample_user, std::vector<float>(n_data_item));

#pragma omp parallel for default(none) shared(data_item, user, n_sample_user, n_data_item, user_idx_l, score_dist_l, vec_dim)
    for (int sampleID = 0; sampleID < n_sample_user; sampleID++) {
        const int userID = user_idx_l[sampleID];
        const float *user_vecs = user.getVector(userID);

        for (int itemID = 0; itemID < n_data_item; itemID++) {
            const float *item_vecs = data_item.getVector(itemID);
            const float IP = InnerProduct(user_vecs, item_vecs, vec_dim);
            score_dist_l[sampleID][itemID] = IP;
        }

        std::sort(score_dist_l[sampleID].begin(),
                  score_dist_l[sampleID].end(), std::greater());
    }

    return score_dist_l;
}

void WriteRank(const std::vector<std::vector<float>> &score_list_l, const char *dataset_name) {

    char resPath[256];
    std::sprintf(resPath, "../../result/attribution/user-score-distribution-%s.csv", dataset_name);
    std::ofstream file(resPath);
    if (!file) {
        std::printf("error in write result\n");
    }

    const int n_sample_user = score_list_l.size();
    const int n_data_item = score_list_l[0].size();

    for (int sampleID = 0; sampleID < n_sample_user; sampleID++) {
        const std::vector<float> user_distribution = score_list_l[sampleID];
        assert(n_data_item == user_distribution.size());
        for (int itemID = 0; itemID < n_data_item - 1; itemID++) {
            file << user_distribution[itemID] << ",";
        }
        file << user_distribution[n_data_item - 1] << std::endl;
    }

    file.close();

}

using namespace std;

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
    spdlog::info("UserScoreDistribution dataset_name {}, basic_dir {}", dataset_name, basic_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    TimeRecord record;
    record.reset();
    const int n_sample_user = 1000;
    std::vector<std::vector<float>> score_list = ScoreDistribution(user, data_item, n_sample_user);
    double score_distribution_time = record.get_elapsed_time_second();
    spdlog::info("build score distribution time: total {}s", score_distribution_time);

    WriteRank(score_list, dataset_name);
    return 0;
}