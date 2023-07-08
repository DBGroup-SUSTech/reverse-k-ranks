//
// Created by bianzheng on 2023/5/30.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"

#include "RunGridIndexDouble.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <random>

class Parameter {
public:
    size_t n_user, n_data_item, vec_dim;
};

void LoadOptions(int argc, char **argv, Parameter &para) {
    namespace po = boost::program_options;

    po::options_description opts("Allowed options");
    opts.add_options()
            ("help,h", "help info")
            ("n_user,nu",
             po::value<std::size_t>(&para.n_user)->default_value(1000),
             "# user")
            ("n_data_item, ndi", po::value<std::size_t>(&para.n_data_item)->default_value(5000),
             "# data item")
            ("vec_dim, vd", po::value<size_t>(&para.vec_dim)->default_value(30),
             "dimensionality");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << opts << std::endl;
        exit(0);
    }
}

using namespace std;
using namespace ReverseMIPS;

std::unique_ptr<double[]> GenerateData(const size_t &n_vector, const size_t &vec_dim) {
    std::random_device rand;
    std::mt19937 gen(rand());
    std::uniform_real_distribution<double> uniform_dist(0, 1);

    std::unique_ptr<double[]> vm_ptr = std::make_unique<double[]>(n_vector * vec_dim);
    const size_t n_ele = n_vector * vec_dim;
    for (size_t eleID = 0; eleID < n_ele; eleID++) {
        vm_ptr[eleID] = uniform_dist(gen);
    }

    return std::move(vm_ptr);
}


int main(int argc, char **argv) {
    Parameter para;
    LoadOptions(argc, argv, para);
    const size_t n_user = para.n_user;
    const size_t n_data_item = para.n_data_item;
    const size_t vec_dim = para.vec_dim;
    spdlog::info("DataTypeCompare n_user {}, n_data_item {}, vec_dim {}",
                 n_user, n_data_item, vec_dim);

    std::unique_ptr<double[]> user_ptr = GenerateData(n_user, vec_dim);
    std::unique_ptr<double[]> data_item_ptr = GenerateData(n_data_item, vec_dim);

    double grid_index_time = 0;
    RunGridIndex(user_ptr, data_item_ptr,
                 n_user, n_data_item, vec_dim,
                 grid_index_time);

    double ip_time = 0;
    RunInnerProduct(user_ptr, data_item_ptr,
                    n_user, n_data_item, vec_dim,
                    ip_time);

    double grid_index_no_negative_time = 0;
    RunGridIndexNoNegative(user_ptr, data_item_ptr,
                           n_user, n_data_item, vec_dim,
                           grid_index_no_negative_time);

    spdlog::info("finish double type, grid index time {}s, inner product time {}s, grid index no negative time {}s",
                 grid_index_time, ip_time, grid_index_no_negative_time);
    return 0;
}