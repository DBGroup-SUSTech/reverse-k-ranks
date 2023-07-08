//
// Created by BianZheng on 2022/9/3.
//

#include "util/VectorIO.hpp"
#include "FileIO.hpp"
#include "struct/VectorMatrix.hpp"

#include "BuildRankSample.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string basic_dir, dataset_name;
    int n_sample;
    uint64_t index_size_gb;
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
             "dataset_name")

            ("index_size_gb, tt", po::value<uint64_t>(&para.index_size_gb)->default_value(50),
             "index size, in unit of GB")
            ("n_sample, ns", po::value<int>(&para.n_sample)->default_value(20),
             "the numer of sample");

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
    const string method_name = "RankSampleTopTBuildIndex";
    spdlog::info("{} dataset_name {}, basic_dir {}", method_name, dataset_name, basic_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    char index_path[256];
    sprintf(index_path, "../../index/index");

    char parameter_name[256] = "";

    const int n_sample = para.n_sample;
    const uint64_t index_size_gb = para.index_size_gb;
    spdlog::info("input parameter: n_sample {} index_size_gb {}", n_sample, index_size_gb);
    double total_compute_time = 0;
    double total_sort_time = 0;
    double process_index_time = 0;

    TimeRecord record;
    record.reset();
    BuildRankSample::BuildIndexMultipleThread(data_item, user, index_path,
                                              n_sample,index_size_gb,
                                              total_compute_time, total_sort_time, process_index_time);
    const double build_index_time = record.get_elapsed_time_second();
    sprintf(parameter_name, "n_sample_%d-index_size_gb_%lu-multiple_thread", n_sample, index_size_gb);

    spdlog::info("finish preprocess and save the index");

    RetrievalResult2 config;

    spdlog::info("build index time: multiple thread {}s, compute time {}s, sort time {}s, process index time {}s",
                 build_index_time, total_compute_time, total_sort_time, process_index_time);

    config.AddBuildIndexTime("Multiple Thread", build_index_time);
    config.AddBuildIndexTime("Compute IP", total_compute_time);
    config.AddBuildIndexTime("Sort IP", total_sort_time);
    config.AddBuildIndexTime("Process Index", process_index_time);
    config.WritePerformance(dataset_name, method_name.c_str(), parameter_name);
    return 0;
}