//
// Created by BianZheng on 2022/12/8.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/MethodBase.hpp"

#include "SimpferPPAboveKMax.hpp"
#include "SimpferPPBelowKMax.hpp"
#include "SimpferPPEstIPCost.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string dataset_dir, dataset_name, method_name, index_dir;
    int topk;
    int simpfer_k_max, min_cache_topk, max_cache_topk;
    size_t stop_time;
    int n_bit;
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
            ("index_dir, id",
             po::value<std::string>(&para.index_dir)->default_value("/home/bianzheng/reverse-k-ranks/index"),
             "the directory of the index")
            ("method_name, mn", po::value<std::string>(&para.method_name)->default_value("BatchDiskBruteForce"),
             "method_name")
            ("topk, tk",
             po::value<int>(&para.topk)->default_value(10),
             "value of topk")
            // reverse top-k adaption
            ("simpfer_k_max, skm", po::value<int>(&para.simpfer_k_max)->default_value(25),
             "k_max in simpfer")
            ("stop_time, st", po::value<size_t>(&para.stop_time)->default_value(60),
             "stop time, in unit of second")
            ("min_cache_topk, mct", po::value<int>(&para.min_cache_topk)->default_value(64),
             "minimum number topk for caching")
            ("max_cache_topk, mct", po::value<int>(&para.max_cache_topk)->default_value(256),
             "maximum number topk for caching")
            // score distribution parameter
            ("n_bit, nb", po::value<int>(&para.n_bit)->default_value(8),
             "number of bit");

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
    const char *index_dir = para.index_dir.c_str();
    string method_name = para.method_name;
    const int topk = para.topk;
    spdlog::info("{} dataset_name {}, topk {}", method_name, dataset_name, topk);
    spdlog::info("dataset_dir {}", dataset_dir);
    spdlog::info("index_dir {}", index_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(dataset_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    char index_path[256];
    sprintf(index_path, "%s/%s.index", index_dir, dataset_name);

    TimeRecord record;
    record.reset();
    unique_ptr<BaseIndex> index;
    char parameter_name[256];
    if (method_name == "SimpferPPAboveKMax") {
        const int simpfer_k_max = para.simpfer_k_max;
        const size_t stop_time = para.stop_time;
        const int min_cache_topk = para.min_cache_topk;
        const int max_cache_topk = para.max_cache_topk;
        spdlog::info("input parameter: simpfer_k_max {}, stop_time {}s, min_cache_topk {}, max_cache_topk {}",
                     simpfer_k_max, stop_time, min_cache_topk, max_cache_topk);
        index = SimpferPPAboveKMax::BuildIndex(data_item, user, simpfer_k_max, stop_time,
                                               min_cache_topk, max_cache_topk);
        sprintf(parameter_name, "top%d-simpfer_k_max_%d", topk, simpfer_k_max);

    } else if (method_name == "SimpferPPBelowKMax") {
        const int simpfer_k_max = para.simpfer_k_max;
        const size_t stop_time = para.stop_time;
        spdlog::info("input parameter: simpfer_k_max {}, stop_time {}s",
                     simpfer_k_max, stop_time);
        index = SimpferPPBelowKMax::BuildIndex(data_item, user, simpfer_k_max, stop_time);
        sprintf(parameter_name, "top%d-simpfer_k_max_%d", topk, simpfer_k_max);

    } else if (method_name == "SimpferPPEstIPCost") {
        const int simpfer_k_max = para.simpfer_k_max;
        const size_t stop_time = para.stop_time;
        const int min_cache_topk = para.min_cache_topk;
        const int max_cache_topk = para.max_cache_topk;
        spdlog::info("input parameter: simpfer_k_max {}, stop_time {}s, min_cache_topk {}, max_cache_topk {}",
                     simpfer_k_max, stop_time, min_cache_topk, max_cache_topk);
        index = SimpferPPEstIPCost::BuildIndex(data_item, user, simpfer_k_max, stop_time,
                                               min_cache_topk, max_cache_topk);
        sprintf(parameter_name, "top%d-simpfer_k_max_%d", topk, simpfer_k_max);

    } else {
        spdlog::error("not such method");
    }

    double build_index_time = record.get_elapsed_time_second();
    spdlog::info("finish preprocess and save the index");

    int n_execute_query = n_query_item;
//    if (method_name == "Simpfer" || method_name == "SimpferOnly" ||
//        method_name == "SimpferFEXIPROTopkOnly") {
//        n_execute_query = 100;
//    }

    RetrievalResult config;

    vector<SingleQueryPerformance> query_performance_l(n_execute_query);
    vector<vector<UserRankElement>> result_rk = index->Retrieval(query_item, topk, n_execute_query,
                                                                 query_performance_l);
    index->FinishCompute();
    string performance_str = index->PerformanceStatistics(topk);
    config.AddRetrievalInfo(performance_str);

    spdlog::info("finish top-{}", topk);
    spdlog::info("{}", performance_str);

    spdlog::info("build index time: total {}s", build_index_time);

    cout << config.GetConfig(0) << endl;
    WriteRankResult(result_rk, dataset_name, method_name.c_str(), parameter_name);
    WriteQueryPerformance(query_performance_l, dataset_name, method_name.c_str(),
                          parameter_name);

    config.AddMemoryInfo(index->IndexSizeByte());
    config.AddBuildIndexTime(build_index_time);
    config.AddExecuteQuery(n_execute_query);
    config.WritePerformance(dataset_name, method_name.c_str(), parameter_name);
    return 0;
}