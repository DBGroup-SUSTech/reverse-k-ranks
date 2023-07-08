//
// Created by BianZheng on 2022/4/13.
//

#include "alg/KMeans/KMeans.hpp"
#include "alg/KMeans/KMeansParallel.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wmissing-braces"
#endif

int main() {
    std::vector<std::vector<double>> data{{1,    1},
                                          {2,    2},
                                          {1200, 1200},
                                          {2,    2}};

    //guarantee the dimension alignment issue before the implementation
    ReverseMIPS::clustering_parameters<double> para(2);
    para.set_random_seed(0);
    para.set_max_iteration(100);
    auto cluster_data = ReverseMIPS::kmeans_lloyd(data, para);

    std::cout << "Means:" << std::endl;
    for (const auto &mean: std::get<0>(cluster_data)) {
        std::cout << "\t(" << mean[0] << "," << mean[1] << ")" << std::endl;
    }
    std::cout << "\nCluster labels:" << std::endl;
    std::cout << "\tPoint:";
    for (const auto &point: data) {
        std::stringstream value;
        value << "(" << point[0] << "," << point[1] << ")";
        std::cout << std::setw(14) << value.str();
    }
    std::cout << std::endl;
    std::cout << "\tLabel:";
    for (const auto &label: std::get<1>(cluster_data)) {
        std::cout << std::setw(14) << label;
    }
    std::cout << std::endl;


    cluster_data = ReverseMIPS::kmeans_lloyd_parallel(data, 2);

    std::cout << "Means:" << std::endl;
    for (const auto &mean: std::get<0>(cluster_data)) {
        std::cout << "\t(" << mean[0] << "," << mean[1] << ")" << std::endl;
    }
    std::cout << "\nCluster labels:" << std::endl;
    std::cout << "\tPoint:";
    for (const auto &point: data) {
        std::stringstream value;
        value << "(" << point[0] << "," << point[1] << ")";
        std::cout << std::setw(14) << value.str();
    }
    std::cout << std::endl;
    std::cout << "\tLabel:";
    for (const auto &label: std::get<1>(cluster_data)) {
        std::cout << std::setw(14) << label;
    }
    std::cout << std::endl;
}
