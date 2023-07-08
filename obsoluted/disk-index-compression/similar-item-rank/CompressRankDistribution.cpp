//
// Created by BianZheng on 2022/6/27.
//

#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/DistancePair.hpp"
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

void CalcItemRankSingleRankDistribution(const VectorMatrix &data_item, const double *user_vecs,
                                        std::vector<int> &rank_l) {
    const int n_data_item = data_item.n_vector_;
    const int vec_dim = data_item.vec_dim_;

    std::vector<DistancePair> distance_l(n_data_item);
    assert(n_data_item == rank_l.size() && n_data_item == distance_l.size());
    for (int itemID = 0; itemID < n_data_item; itemID++) {
        const double ip = InnerProduct(user_vecs, data_item.getVector(itemID), vec_dim);
        distance_l[itemID] = DistancePair(ip, itemID);
    }
    std::sort(distance_l.begin(), distance_l.end(), std::greater());

    for (int candID = 0; candID < n_data_item; candID++) {
        rank_l[candID] = -1;
    }
    for (int rankID = 0; rankID < n_data_item; rankID++) {
        const int itemID = distance_l[rankID].ID_;
        rank_l[itemID] = rankID;
    }
    for (int candID = 0; candID < n_data_item; candID++) {
        assert(rank_l[candID] != -1);
    }
}

void CalcItemRankScoreInterval(const VectorMatrix &data_item, const double *user_vecs,
                               std::vector<int> &rank_l) {
    const int n_interval = 64;
    const int n_data_item = data_item.n_vector_;
    const int vec_dim = data_item.vec_dim_;

    std::vector<DistancePair> distance_l(n_data_item);
    assert(n_data_item == rank_l.size() && n_data_item == distance_l.size());
    for (int itemID = 0; itemID < n_data_item; itemID++) {
        const double ip = InnerProduct(user_vecs, data_item.getVector(itemID), vec_dim);
        distance_l[itemID] = DistancePair(ip, itemID);
    }
    std::sort(distance_l.begin(), distance_l.end(), std::greater());

    for (int candID = 0; candID < n_data_item; candID++) {
        rank_l[candID] = -1;
    }

    double upper_bound = distance_l[0].dist_ + 0.01;
    double lower_bound = distance_l[n_data_item - 1].dist_ - 0.01;
    double interval_distance = (upper_bound - lower_bound) / n_interval;

    for (int candID = 0; candID < n_data_item; candID++) {
        int itemID = distance_l[candID].ID_;
        double ip = distance_l[candID].dist_;
        int itv_idx = std::floor((upper_bound - ip) / interval_distance);
        assert(0 <= itv_idx && itv_idx < n_interval);
        rank_l[itemID] = itv_idx;
    }

    for (int candID = 0; candID < n_data_item; candID++) {
        assert(rank_l[candID] != -1);
    }
}

void CalcItemRankCompressRankDistribution(const VectorMatrix &data_item, const double *user_vecs,
                                          std::vector<int> &rank_l) {
    const int n_sample = 64;
    const int n_data_item = data_item.n_vector_;
    const int vec_dim = data_item.vec_dim_;

    std::vector<DistancePair> distance_l(n_data_item);
    assert(n_data_item == rank_l.size() && n_data_item == distance_l.size());
    for (int itemID = 0; itemID < n_data_item; itemID++) {
        const double ip = InnerProduct(user_vecs, data_item.getVector(itemID), vec_dim);
        distance_l[itemID] = DistancePair(ip, itemID);
    }
    std::sort(distance_l.begin(), distance_l.end(), std::greater());

    for (int candID = 0; candID < n_data_item; candID++) {
        rank_l[candID] = -1;
    }

    int n_cache_every = n_data_item / n_sample;
    if (n_data_item % n_sample != 0) {
        n_cache_every++;
    }

    for (int candID = 0; candID < n_data_item; candID++) {
        int itemID = distance_l[candID].ID_;
        double ip = distance_l[candID].dist_;
        int compressID = candID / n_cache_every;
        assert(0 <= compressID && compressID < n_sample);
        rank_l[itemID] = compressID;
    }

    for (int candID = 0; candID < n_data_item; candID++) {
        assert(rank_l[candID] != -1);
    }
}

void WriteRankRelation(const std::vector<std::pair<int, int>> &rank_l,
                       const char *method_name, const char *dataset_name, const char *user_relation_name,
                       const int sampleID) {
    char resPath[256];
    std::sprintf(resPath, "../../result/attribution/SimilarItemRank/%s-%s-%s-sampleID-%d.csv",
                 method_name, dataset_name, user_relation_name, sampleID);
    std::ofstream file(resPath);
    if (!file) {
        spdlog::error("error in write result\n");
    }

    const int n_rank = (int) rank_l.size();

    for (int rankID = 0; rankID < n_rank; rankID++) {
        file << rank_l[rankID].first << "," << rank_l[rankID].second << std::endl;
    }

    file.close();
}

void CalcSaveRankRelation(const VectorMatrix &user, const VectorMatrix &data_item,
                          const char *user_relation_name, const char *dataset_name,
                          const int &sample_userID, const int &cand_userID, const int sampleID) {
    const int n_data_item = data_item.n_vector_;

    {
        std::vector<int> sample_rank_l(n_data_item);
        CalcItemRankCompressRankDistribution(data_item, user.getVector(sample_userID), sample_rank_l);

        std::vector<int> cand_rank_l(n_data_item);
        CalcItemRankCompressRankDistribution(data_item, user.getVector(cand_userID), cand_rank_l);

        std::vector<std::pair<int, int>> item_rank_l(n_data_item);
        item_rank_l.assign(n_data_item, std::make_pair(-1, -1));
        for (int itemID = 0; itemID < n_data_item; itemID++) {
            int sample_rank = sample_rank_l[itemID];
            int cand_rank = cand_rank_l[itemID];
            item_rank_l[itemID] = std::make_pair(sample_rank, cand_rank);
        }
        for (int rankID = 0; rankID < n_data_item; rankID++) {
            assert(item_rank_l[rankID].first != -1 && item_rank_l[rankID].second != -1);
        }

        WriteRankRelation(item_rank_l, "compress-rank-relation", dataset_name, user_relation_name, sampleID);
    }

    {
        std::vector<int> sample_rank_l(n_data_item);
        CalcItemRankScoreInterval(data_item, user.getVector(sample_userID), sample_rank_l);

        std::vector<int> cand_rank_l(n_data_item);
        CalcItemRankScoreInterval(data_item, user.getVector(cand_userID), cand_rank_l);

        std::vector<std::pair<int, int>> item_rank_l(n_data_item);
        item_rank_l.assign(n_data_item, std::make_pair(-1, -1));
        for (int itemID = 0; itemID < n_data_item; itemID++) {
            int sample_rank = sample_rank_l[itemID];
            int cand_rank = cand_rank_l[itemID];
            item_rank_l[itemID] = std::make_pair(sample_rank, cand_rank);
        }
        for (int rankID = 0; rankID < n_data_item; rankID++) {
            assert(item_rank_l[rankID].first != -1 && item_rank_l[rankID].second != -1);
        }

        WriteRankRelation(item_rank_l, "score-interval", dataset_name, user_relation_name, sampleID);
    }

    {
        std::vector<int> sample_rank_l(n_data_item);
        CalcItemRankSingleRankDistribution(data_item, user.getVector(sample_userID), sample_rank_l);

        std::vector<int> cand_rank_l(n_data_item);
        CalcItemRankSingleRankDistribution(data_item, user.getVector(cand_userID), cand_rank_l);

        std::vector<std::pair<int, int>> item_rank_l(n_data_item);
        item_rank_l.assign(n_data_item, std::make_pair(-1, -1));
        for (int itemID = 0; itemID < n_data_item; itemID++) {
            int sample_rank = sample_rank_l[itemID];
            int cand_rank = cand_rank_l[itemID];
            item_rank_l[itemID] = std::make_pair(sample_rank, cand_rank);
        }
        for (int rankID = 0; rankID < n_data_item; rankID++) {
            assert(item_rank_l[rankID].first != -1 && item_rank_l[rankID].second != -1);
        }

        WriteRankRelation(item_rank_l, "single-rank-distribution", dataset_name, user_relation_name, sampleID);
    }

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
    spdlog::info("SimilarItemRank dataset_name {}, basic_dir {}", dataset_name, basic_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    TimeRecord record;
    record.reset();

    user.vectorNormalize();

    const int n_sample_user = 15;
    //generate the number sequentially, then shuffle
    std::vector<int> user_idx_l(n_user);
    std::iota(user_idx_l.begin(), user_idx_l.end(), 0);

    std::random_device rd;
    std::mt19937 random_gen(rd());
    std::shuffle(user_idx_l.begin(), user_idx_l.end(), random_gen);

    for (int sampleID = 0; sampleID < n_sample_user; sampleID++) {
        const int sample_userID = user_idx_l[sampleID];
        std::vector<DistancePair> distance_l(n_user);

        //calc cosine, get the maximum as the basic information
#pragma omp parallel for default(none) shared(sample_userID, n_user, user, vec_dim, distance_l)
        for (int cand_userID = 0; cand_userID < n_user; cand_userID++) {
            double ip = InnerProduct(user.getVector(sample_userID), user.getVector(cand_userID), vec_dim);
            distance_l[cand_userID] = DistancePair(ip, cand_userID);
        }
        std::sort(distance_l.begin(), distance_l.end(), std::greater());

        const int near_userID = distance_l[1].ID_;

        CalcSaveRankRelation(user, data_item,
                             "nearest-user", dataset_name,
                             sample_userID, near_userID, sampleID);

        int far_userID = distance_l[n_user - 1].ID_;
        CalcSaveRankRelation(user, data_item,
                             "far-user", dataset_name,
                             sample_userID, far_userID, sampleID);
    }

    double score_distribution_time = record.get_elapsed_time_second();
    spdlog::info("build score distribution time: total {}s", score_distribution_time);


    return 0;
}