//
// Created by BianZheng on 2022/11/11.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"

#include "ReverseMIPS.hpp"
#include "AttributionFileIO.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string dataset_dir, dataset_name;
    int simpfer_k_max;
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
            // reverse top-k adaption
            ("simpfer_k_max, skm", po::value<int>(&para.simpfer_k_max)->default_value(300),
             "k_max in simpfer");

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
    spdlog::info("ReverseMIPS dataset_name {}, dataset_dir {}", dataset_name, dataset_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(dataset_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    TimeRecord record;
    record.reset();
    char parameter_name[256] = "";
    const int simpfer_k_max = para.simpfer_k_max;
    spdlog::info("input parameter: simpfer_k_max {}", simpfer_k_max);
    unique_ptr<RMIPS::Index> index = RMIPS::BuildIndex(data_item, user, simpfer_k_max);
    sprintf(parameter_name, "simpfer_k_max_%d", simpfer_k_max);

    double build_index_time = record.get_elapsed_time_second();
    spdlog::info("finish preprocess and save the index");

    vector<int> topk_l;
    topk_l = {200};
//    topk_l = {10};

    vector<vector<int>> result_rank_l;
    const int n_execute_query = n_query_item;
    for (int topk: topk_l) {
        vector<int> result_rk = index->Retrieval(query_item, topk, n_execute_query);

        string performance_str = index->PerformanceStatistics(topk);

        result_rank_l.emplace_back(result_rk);
        spdlog::info("finish top-{}", topk);
        spdlog::info("{}", performance_str);
    }

    spdlog::info("build index time: total {}s", build_index_time);
    int n_topk = (int) topk_l.size();

    for (int i = 0; i < n_topk; i++) {
        const int topk = topk_l[i];
        AttributionWriteRankResult(result_rank_l[i], topk, dataset_name, parameter_name);
    }
    return 0;
}