//
// Created by BianZheng on 2022/6/3.
//

#include "alg/SpaceInnerProduct.hpp"
#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "struct/VectorMatrix.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

void WritePerformance(const char *dataset_name, const double build_index_time) {
    char resPath[256];
    std::sprintf(resPath, "../../result/attribution/%s-score-run-time.txt", dataset_name);
    std::ofstream file(resPath);
    if (!file) {
        spdlog::error("error in write result");
    }
    file << "build index time: " << build_index_time << "s" << std::endl;
    file.close();
}

class Parameter {
public:
    std::string basic_dir, dataset_name;
};

void LoadOptions(int argc, char **argv, Parameter &para) {
    namespace po = boost::program_options;

    po::options_description opts("Allowed options");
    opts.add_options()
            ("help,h", "help info")
            ("basic_dir,bd",
             po::value<std::string>(&para.basic_dir)->default_value("/home/bianzheng/Dataset/ReverseMIPS"),
             "basic directory")
            ("dataset_name, ds", po::value<std::string>(&para.dataset_name)->default_value("fake-normal"),
             "dataset_name");

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
    const char *basic_dir = para.basic_dir.c_str();
    spdlog::info("dataset_name {}, basic_dir {}", dataset_name, basic_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    TimeRecord record;
    record.reset();

    vector<double> distance_cache(n_data_item);
    double total_sum = 0;
    for (int userID = 0; userID < n_user; userID++) {
        for (int itemID = 0; itemID < n_data_item; itemID++) {
            double ip = InnerProduct(data_item.getVector(itemID), user.getVector(userID), vec_dim);
            distance_cache[itemID] = ip;
            total_sum += distance_cache[itemID];
        }
        std::sort(distance_cache.begin(), distance_cache.end(), std::greater());
    }

    const double build_index_time = record.get_elapsed_time_second();

    spdlog::info("build index time: total {}s", build_index_time);

    WritePerformance(dataset_name, build_index_time);
    return 0;
}