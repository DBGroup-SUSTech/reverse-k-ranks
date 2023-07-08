//
// Created by BianZheng on 2022/5/20.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"

#include "ComputeAll.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string basic_dir, dataset_name, bound_name;
    int scale, n_codebook, n_codeword;
};

void LoadOptions(int argc, char **argv, Parameter &para) {
    namespace po = boost::program_options;

    po::options_description opts("Allowed options");
    opts.add_options()
            ("help,h", "help info")
            ("basic_dir,bd",
             po::value<std::string>(&para.basic_dir)->default_value("/home/bianzheng/Dataset/ReverseMIPS"),
             "basic directory")
            ("dataset_name, ds", po::value<std::string>(&para.dataset_name)->default_value("fake-small"),
             "dataset_name")
            ("bound_name, bn", po::value<std::string>(&para.bound_name)->default_value("OnlineGrid"),
             "bound_name")
            ("scale, s", po::value<int>(&para.scale)->default_value(100),
             "scale for integer bound")
            ("n_codebook, ncb", po::value<int>(&para.n_codebook)->default_value(8),
             "number of codebook")
            ("n_codeword, ncw", po::value<int>(&para.n_codeword)->default_value(32),
             "number of codeword");

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
    string bound_name = para.bound_name;
    spdlog::info("bound_name {} dataset_name {}, basic_dir {}", bound_name, dataset_name, basic_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    char index_path[256];
    sprintf(index_path, "../index/index");

    TimeRecord record;
    record.reset();
    unique_ptr<BaseIndex> index;
    char parameter_name[256] = "";

    spdlog::info("input parameter: none");
    index = ComputeAll::BuildIndex(data_item, user, bound_name, para.scale, para.n_codebook, para.n_codeword,
                                   parameter_name);

    double build_index_time = record.get_elapsed_time_second();
    spdlog::info("finish preprocess and save the index");

    vector<int> topk_l{20, 10};
    RetrievalResult config;
    vector<vector<vector<UserRankElement>>> result_rank_l;
    for (int topk: topk_l) {
        record.reset();
        vector<vector<UserRankElement>> result_rk = index->Retrieval(query_item, topk);

        double retrieval_time = record.get_elapsed_time_second();
        double ms_per_query = retrieval_time / n_query_item * 1000;

        string performance_str = index->PerformanceStatistics(topk, retrieval_time, ms_per_query);
        config.AddRetrievalInfo(performance_str, topk, retrieval_time, ms_per_query);

        result_rank_l.emplace_back(result_rk);
        spdlog::info("finish top-{}", topk);
    }

    spdlog::info("build index time: total {}s", build_index_time);
    int n_topk = (int) topk_l.size();

    for (int i = 0; i < n_topk; i++) {
        cout << config.GetConfig(i) << endl;
        WriteRankResult(result_rank_l[i], dataset_name, bound_name.c_str(), parameter_name);
    }

    config.AddBuildIndexInfo(index->BuildIndexStatistics());
    config.AddBuildIndexTime(build_index_time);
    config.WritePerformance(dataset_name, bound_name.c_str(), parameter_name);
    return 0;
}