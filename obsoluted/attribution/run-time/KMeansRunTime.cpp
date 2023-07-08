//
// Created by bianzheng on 2022/5/11.
//

#include "alg/Cluster/KMeansParallel.hpp"
#include "util/TimeMemory.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <fstream>
#include <spdlog/spdlog.h>

using namespace std;
using namespace ReverseMIPS;

vector<vector<double>> GenRandom(const int &n_eval, const int &n_dim) {
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(1.0, 1000.0);

    vector<vector<double>> matrix(n_eval, std::vector<double>(n_dim));

    for (int itemID = 0; itemID < n_eval; itemID++) {
        for (int dim = 0; dim < n_dim; dim++) {
            double random = dis(gen);
            matrix[itemID][dim] = random;
        }
    }
    return matrix;
}

void AttributionWrite(const std::vector<double> &result_l, const std::vector<int> &k_l) {

    char resPath[256];
    std::sprintf(resPath, "../../result/attribution/KMeansRunTime/result.txt");
    std::ofstream file(resPath);
    if (!file) {
        std::printf("error in write result\n");
    }

    assert(result_l.size() == k_l.size());
    int size = (int) result_l.size();

    for (int i = 0; i < size; i++) {
        file << "k=" << k_l[i] << ", running time " << std::to_string(result_l[i]) << "s" << std::endl;
    }

    file.close();
}

int main(int argc, char **argv) {
    const int dimension = 150;
    vector<int> k_l{1200, 2500, 5000, 10000, 20000, 40000, 80000};
    const int n_eval = 100000;
    spdlog::info("KMeansRunTime");

    vector<double> result_l;

    vector<vector<double>> matrix = GenRandom(n_eval, dimension);

    for (const int &k: k_l) {

        //compute the inner product
        TimeRecord record;
        record.reset();
        //perform kmeans
        double comp_res = 0;

        ReverseMIPS::clustering_parameters<double> para(k);
        para.set_random_seed(0);
        para.set_max_iteration(20);
//        std::tuple<std::vector<std::vector<double>>, std::vector<uint32_t>> cluster_data = kmeans_lloyd(matrix, para);
        std::tuple<std::vector<std::vector<double>>, std::vector<uint32_t>> cluster_data = kmeans_lloyd_parallel(matrix, para);
        std::vector<std::vector<double>> centroid_l = std::get<0>(cluster_data);
        double run_time = record.get_elapsed_time_second();

        for (int centerID = 0; centerID < k; centerID++) {
            for (int dimID = 0; dimID < dimension; dimID++) {
                comp_res += centroid_l[centerID][dimID];
            }
        }

        spdlog::info("k {}, running time {}s", k, run_time);
        spdlog::info("compute res {}", comp_res);
//        assert(grid_lb <= comp_res && comp_res <= grid_ub);
        result_l.emplace_back(run_time);
    }
    AttributionWrite(result_l, k_l);

    return 0;
}
