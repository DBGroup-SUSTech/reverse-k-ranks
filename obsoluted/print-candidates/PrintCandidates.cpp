//
// Created by BianZheng on 2022/6/20.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"

#include "PruneUser.hpp"
#include "CandidatesIO.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string basic_dir, dataset_name;
    int n_sample, index_size_gb;
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

            ("n_sample, ns", po::value<int>(&para.n_sample)->default_value(20),
             "the numer of sample")
            ("index_size_gb, tt", po::value<int>(&para.index_size_gb)->default_value(50),
             "index size, in unit of GB");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << opts << std::endl;
        exit(0);
    }
}

using namespace ReverseMIPS;
using namespace std;


int main(int argc, char **argv) {
    Parameter para;
    LoadOptions(argc, argv, para);
    const char *dataset_name = para.dataset_name.c_str();
    const char *basic_dir = para.basic_dir.c_str();
    string method_name = "FullID";
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

    TimeRecord record;
    record.reset();

    const int n_sample = para.n_sample;
    spdlog::info("input parameter: n_sample {}",
                 n_sample);
    unique_ptr<CompressTopTIDBruteForce::Index> index = CompressTopTIDBruteForce::BuildIndex(data_item, user,
                                                                                             index_path,
                                                                                             n_sample);
    char other_name[256];
    sprintf(other_name, "n_sample_%d", n_sample);

    double build_index_time = record.get_elapsed_time_second();
    spdlog::info("finish preprocess and save the index, time {}s", build_index_time);

    vector<int> topk_l{50, 10};
//    vector<int> topk_l{10000, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8};
//    vector<int> topk_l{20};
    CreateCandidateFile(dataset_name);

    for (int topk: topk_l) {
        record.reset();
        std::vector<std::vector<ItemCandidates>> result_cand;
        result_cand.clear();
        vector<vector<UserRankElement>> result_rk = index->Retrieval(query_item, topk, result_cand);

        double retrieval_time = record.get_elapsed_time_second();
        double ms_per_query = retrieval_time / n_query_item * 1000;

        string performance_str = index->PerformanceStatistics(topk, retrieval_time, ms_per_query);
        cout << performance_str << endl;

        printf("topk %d\n", topk);
        WriteRankResult(result_rk, dataset_name, method_name.c_str(), other_name);
        WriteCandidateResult(result_cand, topk, dataset_name);
        spdlog::info("finish top-{}", topk);
    }
    return 0;
}