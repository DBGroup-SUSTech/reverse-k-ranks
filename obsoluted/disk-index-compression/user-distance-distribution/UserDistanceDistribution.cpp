//
// Created by BianZheng on 2022/6/23.
//

#include "CandidatesIO.hpp"
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

std::vector<int>
UserDistanceDistribution(const VectorMatrix &user, const std::vector<int> &user_cand_l) {
    const int n_user = user.n_vector_;
    const int vec_dim = user.vec_dim_;
    const int n_user_cand = (int) user_cand_l.size();

    const double min_val = -1;
    const double max_val = 1;
    const double itv_dist = 0.01;
    const int n_itv = int((max_val - min_val) / itv_dist);
    std::vector<int> frequency_l(n_itv);
    for (int itvID = 0; itvID < n_itv; itvID++) {
        frequency_l[itvID] = 0;
    }

#pragma omp parallel for default(none) shared(n_user_cand, user, user_cand_l, n_user, vec_dim, min_val, max_val, itv_dist, frequency_l)
    for (int sampleID = 0; sampleID < n_user_cand; sampleID++) {
        const int userID = user_cand_l[sampleID];
        assert(userID < n_user);
        const double *user_vecs = user.getVector(userID);

        for (int other_sampleID = sampleID + 1; other_sampleID < n_user_cand; other_sampleID++) {
            const int other_userID = user_cand_l[other_sampleID];
            const double *other_user_vecs = user.getVector(other_userID);

            const double IP = InnerProduct(user_vecs, other_user_vecs, vec_dim);
            const int itvID = int((IP - min_val) / itv_dist);
            frequency_l[itvID]++;
        }

    }

    return frequency_l;
}

void WriteScoreDistribution(const std::vector<int> &distance_distribution_l,
                            const char *dataset_name, const int topk, const char *other_name) {

    char resPath[256];
    if (strcmp(other_name, "") == 0) {
        std::sprintf(resPath, "../../result/attribution/user-score-distribution-%s-top%d.csv",
                     dataset_name, topk);
    } else {
        std::sprintf(resPath, "../../result/attribution/user-score-distribution-%s-top%d-%s.csv",
                     dataset_name, topk, other_name);
    }
    std::ofstream file(resPath);
    if (!file) {
        spdlog::error("error in write result\n");
    }

    const double min_val = -1;
    const double max_val = 1;
    const double itv_dist = 0.01;
    const int n_itv = int((max_val - min_val) / itv_dist);
    std::vector<double> score_val_l(n_itv);
    for (int itvID = 0; itvID < n_itv; itvID++) {
        score_val_l[itvID] = min_val + itv_dist * itvID;
    }

    assert(score_val_l.size() == distance_distribution_l.size());
    for (int itvID = 0; itvID < n_itv; itvID++) {
        file << score_val_l[itvID] << "," << distance_distribution_l[itvID] << std::endl;
    }

    file.close();
}

using namespace std;

int main(int argc, char **argv) {
    if (!(argc == 3 or argc == 4)) {
        cout << argv[0] << " dataset_name topk [basic_dir]" << endl;
        return 0;
    }
    const char *dataset_name = argv[1];
    const int topk = std::atoi(argv[2]);
    const char *basic_dir = "/home/bianzheng/Dataset/ReverseMIPS";
    if (argc == 4) {
        basic_dir = argv[3];
    }
    spdlog::info("UserDistanceDistribution dataset_name {}, basic_dir {}", dataset_name, basic_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    user.vectorNormalize();

    TimeRecord record;
    record.reset();

    {
        const int n_sample_query = 5;
        //generate the number sequentially, then shuffle
        std::vector<int> query_idx_l(n_query_item);
        std::iota(query_idx_l.begin(), query_idx_l.end(), 0);

        std::random_device rd;
        std::mt19937 random_gen(rd());
        std::shuffle(query_idx_l.begin(), query_idx_l.end(), random_gen);

        for (int sampleID = 0; sampleID < n_sample_query; sampleID++) {
            const int queryID = query_idx_l[sampleID];
            char cand_path[256];
            sprintf(cand_path, "../../result/attribution/Candidate-%s/%s-top%d-qID-%d.txt",
                    dataset_name, dataset_name, topk, queryID);
            std::vector<int> user_cand_l;
            ReadUserCandidates(cand_path, user_cand_l);

            std::vector<int> user_distance_distribution = UserDistanceDistribution(user, user_cand_l);

            char other_name[256];
            sprintf(other_name, "queryID-%d", queryID);
            WriteScoreDistribution(user_distance_distribution, dataset_name, topk, other_name);
        }
    }

    std::vector<int> user_full_l(n_user);
    for (int userID = 0; userID < n_user; userID++) {
        user_full_l[userID] = userID;
    }
    std::vector<int> user_full_distance_distribution = UserDistanceDistribution(user, user_full_l);
    WriteScoreDistribution(user_full_distance_distribution, dataset_name, topk, "");

    double score_distribution_time = record.get_elapsed_time_second();
    spdlog::info("build score distribution time: total {}s", score_distribution_time);


    return 0;
}