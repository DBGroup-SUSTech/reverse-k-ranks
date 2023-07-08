//
// Created by bianzheng on 2023/5/23.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"

#include "RunGridIndex.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string dataset_dir, dataset_name;
    int n_sample_user;
};

void LoadOptions(int argc, char **argv, Parameter &para) {
    namespace po = boost::program_options;

    po::options_description opts("Allowed options");
    opts.add_options()
            ("help,h", "help info")
            ("dataset_dir,dd",
             po::value<std::string>(&para.dataset_dir)->default_value("/home/bianzheng/Dataset/ReverseMIPS"),
             "the basic directory of dataset")
            ("dataset_name, dn", po::value<std::string>(&para.dataset_name)->default_value("fake-normal"),
             "dataset_name")
            ("n_sample_user, nsu", po::value<int>(&para.n_sample_user)->default_value(50),
             "number of user to sample");

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


int main(int argc, char **argv) {
    Parameter para;
    LoadOptions(argc, argv, para);
    const char *dataset_name = para.dataset_name.c_str();
    const char *dataset_dir = para.dataset_dir.c_str();
    spdlog::info("GridBoundCompare dataset_name {}", dataset_name);
    spdlog::info("dataset_dir {}", dataset_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(dataset_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    const int n_sample_user = para.n_sample_user;
    spdlog::info("n_sample_user {}", n_sample_user);

    double grid_index_time = 0;
    RunGridIndex(user, data_item, n_sample_user, grid_index_time);

    double ip_time = 0;
    RunInnerProduct(user, data_item, n_sample_user, ip_time);

    double grid_index_no_negative_time = 0;
    RunGridIndexNoNegative(user, data_item, n_sample_user, grid_index_no_negative_time);

    spdlog::info(
            "finish GridBoundCompare dataset_name {}, grid index time {}s, inner product time {}s, grid index no negative time {}s",
            dataset_name, grid_index_time, ip_time, grid_index_no_negative_time);
    return 0;
}