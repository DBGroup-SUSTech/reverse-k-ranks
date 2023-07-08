//
// Created by BianZheng on 2022/11/30.
//

#ifndef REVERSE_KRANKS_COMPLEXITYCOMPAREUTIL_HPP
#define REVERSE_KRANKS_COMPLEXITYCOMPAREUTIL_HPP

#include <random>
#include <fstream>
#include <cassert>
#include <memory>

namespace ReverseMIPS {
    std::unique_ptr<double[]> GenRandom(const size_t &n_user, const size_t &n_dim) {
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(1.0, 1000.0);

        std::unique_ptr<double[]> random_l = std::make_unique<double[]>(n_user * n_dim);

        for (size_t itemID = 0; itemID < n_user; itemID++) {
            for (size_t dim = 0; dim < n_dim; dim++) {
                size_t id = itemID * n_dim + dim;
                double random = dis(gen);
                random_l[id] = random;
            }
        }
        return random_l;
    }

    void SortArray(double *ip_vecs_l, const size_t &n_user, const size_t &tau) {
#pragma omp parallel for default(none) shared(n_user, ip_vecs_l, tau)
        for (size_t userID = 0; userID < n_user; userID++) {
            std::sort(ip_vecs_l + userID * tau, ip_vecs_l + (userID + 1) * tau, std::greater());
        }
    }

    void WritePerformance(const std::vector<std::pair<size_t, double>> &time_use_l,
                          const char *method_name, const size_t& n_user, const size_t& n_query) {

        char resPath[256];
        std::sprintf(resPath, "../../result/attribution/complexity-compare-%s-n_user_%ld-n_query_%ld.txt",
                     method_name, n_user, n_query);
        std::ofstream file(resPath);
        if (!file) {
            std::printf("error in write result\n");
        }

        const int size = (int) time_use_l.size();
        for (int i = 0; i < size; i++) {
            const size_t try_dim = time_use_l[i].first;
            const double time = time_use_l[i].second;
            file << "try_dim: " << try_dim << ", time: " << time << "s" << std::endl;
        }

        file.close();

    }
}
#endif //REVERSE_KRANKS_COMPLEXITYCOMPAREUTIL_HPP
