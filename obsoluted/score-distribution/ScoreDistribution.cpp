//
// Created by BianZheng on 2022/6/13.
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
#include <iomanip>

using namespace ReverseMIPS;

void ScoreDistributionSample(const VectorMatrix &iter_vm, const double *sample_vecs,
                             const std::pair<double, double> &ip_bound_pair,
                             const double &hist_distance, const int &n_interval,
                             std::vector<int> &frequency_l) {
    const int n_vecs = iter_vm.n_vector_;
    const int vec_dim = iter_vm.vec_dim_;
    assert(frequency_l.size() == n_interval);

#pragma omp parallel for default(none) shared(n_vecs, iter_vm, sample_vecs, vec_dim, ip_bound_pair, hist_distance, n_interval, frequency_l)
    for (int vecsID = 0; vecsID < n_vecs; vecsID++) {
        const double *iter_vecs = iter_vm.getVector(vecsID);
        const double IP = InnerProduct(sample_vecs, iter_vecs, vec_dim);

        const int itvID = std::ceil((IP - ip_bound_pair.first) / hist_distance);
        assert(0 <= itvID && itvID < n_interval);
        frequency_l[itvID]++;
    }

}

void
WriteRank(const std::vector<double> &interval_val_l, const std::vector<int> &frequency_l,
          const char *dataset_name, const int &userID, const int &itemID) {

    char resPath[256];
    if (userID != -1) {
        std::sprintf(resPath, "../../result/attribution/score-distribution-%s-userID-%d.csv", dataset_name, userID);
    } else if (itemID != -1) {
        std::sprintf(resPath, "../../result/attribution/score-distribution-%s-itemID-%d.csv", dataset_name, itemID);
    } else {
        std::sprintf(resPath, "../../result/attribution/score-distribution-%s.csv", dataset_name);
    }
    std::ofstream file(resPath);
    if (!file) {
        std::printf("error in write result\n");
    }
    assert(interval_val_l.size() == frequency_l.size());

    const int n_interval = (int) interval_val_l.size();

    for (int itvID = 0; itvID < n_interval; itvID++) {
        file << std::setw(5) << interval_val_l[itvID] << ","
             << std::setw(7) << frequency_l[itvID]
             << std::endl;

//        file << interval_val_l[itvID] << "," << frequency_l[itvID] << std::endl;
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
    spdlog::info("ScoreDistribution dataset_name {}, basic_dir {}", dataset_name, basic_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    TimeRecord record;
    record.reset();

    const std::pair<double, double> ip_bound_pair = std::make_pair(-5, 10);
    const double hist_distance = 0.01;

    const int n_interval = std::ceil((ip_bound_pair.second - ip_bound_pair.first) / hist_distance);
    std::vector<double> interval_val_l(n_interval);
    for (int itvID = 0; itvID < n_interval; itvID++) {
        interval_val_l[itvID] = ip_bound_pair.first + itvID * hist_distance;
    }

    {
        const int n_sample_user = 30;
        //generate the number sequentially, then shuffle
        std::vector<int> user_idx_l(n_user);
        std::iota(user_idx_l.begin(), user_idx_l.end(), 0);

        std::random_device rd;
        std::mt19937 random_gen(rd());
        std::shuffle(user_idx_l.begin(), user_idx_l.end(), random_gen);

        std::vector<int> frequency_l(n_data_item);
        for (int sampleID = 0; sampleID < n_sample_user; sampleID++) {
            const int &userID = user_idx_l[sampleID];
            frequency_l.assign(n_interval, 0);
            const double *user_vecs = user.getVector(userID);
            ScoreDistributionSample(data_item, user_vecs,
                                    ip_bound_pair, hist_distance, n_interval,
                                    frequency_l);
            WriteRank(interval_val_l, frequency_l, dataset_name, userID, -1);
        }

    }

    double sample_user_time = record.get_elapsed_time_second();
    record.reset();

    {
        const int n_sample_item = 30;
        //generate the number sequentially, then shuffle
        std::vector<int> item_idx_l(n_data_item);
        std::iota(item_idx_l.begin(), item_idx_l.end(), 0);

        std::random_device rd;
        std::mt19937 random_gen(rd());
        std::shuffle(item_idx_l.begin(), item_idx_l.end(), random_gen);

        std::vector<int> frequency_l(n_data_item);
        for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
            const int &itemID = item_idx_l[sampleID];
            const double *item_vecs = data_item.getVector(itemID);

            frequency_l.assign(n_interval, 0);
            ScoreDistributionSample(user, item_vecs,
                                    ip_bound_pair, hist_distance, n_interval,
                                    frequency_l);
            WriteRank(interval_val_l, frequency_l, dataset_name, -1, itemID);
        }

    }

    double sample_item_time = record.get_elapsed_time_second();
    record.reset();

    {
        std::vector<int> frequency_l(n_data_item);
        frequency_l.assign(n_interval, 0);
        for (int userID = 0; userID < n_user; userID++) {
            const double *user_vecs = user.getVector(userID);
            ScoreDistributionSample(data_item, user_vecs,
                                    ip_bound_pair, hist_distance, n_interval,
                                    frequency_l);
        }
        WriteRank(interval_val_l, frequency_l, dataset_name, -1, -1);
    }

    double total_time = record.get_elapsed_time_second();
    spdlog::info("sample user time: {}s", sample_user_time);
    spdlog::info("sample item time: {}s", sample_item_time);
    spdlog::info("total time: {}s", total_time);

    return 0;
}