//
// Created by BianZheng on 2022/2/27.
//

#include "alg/SpaceInnerProduct.hpp"
#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "struct/VectorMatrix.hpp"
#include <algorithm>
#include <iostream>
#include <vector>
#include <spdlog/spdlog.h>

using namespace std;
using namespace ReverseMIPS;

void WriteDistribution(vector<double> &distribution, const char *dataset_name) {
    char resPath[256];
    std::sprintf(resPath, "../../result/attribution/%s-config.txt", dataset_name);
    std::ofstream file(resPath);
    if (!file) {
        std::printf("error in write result\n");
    }

    int n_data_item = distribution.size();
    for (int i = 0; i < n_data_item; i++) {
        file << distribution[i] << endl;
    }

    printf("finish write distribution\n");
    file.close();
}

vector<double> CalculateBoundDistribution(VectorMatrix &user, VectorMatrix &data_item) {
    const int report_every = 50000;
    TimeRecord record;
    const int n_user = user.n_vector_;
    const int vec_dim = user.vec_dim_;
    const int n_data_item = data_item.n_vector_;
    //calculate the IP of all the element

    vector<double> distribution(n_data_item);
    for (int i = 0; i < n_data_item; i++) {
        distribution[i] = 0;
    }

    record.reset();
#pragma omp parallel for default(none) shared(n_user, n_data_item, vec_dim, data_item, user, distribution, std::cout, record)
    for (int userID = 0; userID < n_user; userID++) {
        vector<double> IPcache(n_data_item);
        double highIP, lowIP;

        for (int itemID = 0; itemID < n_data_item; itemID++) {
            double query_dist = InnerProduct(data_item.getVector(itemID), user.getVector(userID), vec_dim);
            IPcache[itemID] = query_dist;
        }
        std::sort(IPcache.begin(), IPcache.end(), std::greater());
        highIP = IPcache[0];
        lowIP = IPcache[n_data_item - 1];

        double IPdiff = highIP - lowIP;
        for (int itemID = 0; itemID < n_data_item; itemID++) {
            distribution[itemID] = distribution[itemID] + (IPcache[itemID] - lowIP) / IPdiff;
        }

        if (userID % report_every == 0) {
            std::cout << "preprocessed " << userID / (0.01 * n_user) << " %, "
                      << record.get_elapsed_time_second() << " s/iter" << " Mem: "
                      << get_current_RSS() / 1000000 << " Mb \n";
            record.reset();
        }

    }

    for (int i = 0; i < n_data_item; i++) {
        distribution[i] /= n_user;
    }
    std::reverse(distribution.begin(), distribution.end());
    printf("finish calculate distribution\n");
    return distribution;
}

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
    printf("CalculateBoundDistribution dataset_name %s, basic_dir %s\n", dataset_name, basic_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    vector<double> distribution = CalculateBoundDistribution(user, data_item);
    WriteDistribution(distribution, dataset_name);
    return 0;
}