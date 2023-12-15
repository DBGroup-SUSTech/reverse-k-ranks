//
// Created by bianzheng on 2022/4/29.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"

#include "GridIndex.hpp"
#include "QS.hpp"
#include "QSRPNormalLP.hpp"
#include "QSRPUniformLP.hpp"
#include "Rtree.hpp"
#include "SimpferPP.hpp"
#include "US.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string dataset_dir, dataset_name, method_name, index_dir;
    int topk;
    int n_sample, n_sample_query, sample_topk;
    int n_thread;
    int simpfer_k_max;
    size_t stop_time;
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
             "the value of topk")
            // memory index parameter
            ("n_sample, ns", po::value<int>(&para.n_sample)->default_value(20),
             "number of sample of a rank bound")
            ("n_sample_query, nsq", po::value<int>(&para.n_sample_query)->default_value(150),
             "the numer of sample query in training query distribution")
            ("sample_topk, st", po::value<int>(&para.sample_topk)->default_value(60),
             "topk in training query distribution")
            ("n_thread, nt", po::value<int>(&para.n_thread)->default_value(-1),
             "number of thread for processing")
            // reverse top-k adaption
            ("simpfer_k_max, skm", po::value<int>(&para.simpfer_k_max)->default_value(25),
             "k_max in simpfer")
            ("stop_time, st", po::value<size_t>(&para.stop_time)->default_value(60),
             "stop time, in unit of second");

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
    spdlog::info("Retrieval method_name {}, dataset_name {}, topk {}", method_name, dataset_name, topk);
    spdlog::info("dataset_dir {}", dataset_dir);
    spdlog::info("index_dir {}", index_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(dataset_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    TimeRecord record;
    record.reset();
    unique_ptr<BaseIndex> index;
    char parameter_name[256] = "";
    if (method_name == "GridIndex") {
        const size_t stop_time = para.stop_time;
        spdlog::info("input parameter: stop_time {}s", stop_time);
        index = GridIndex::BuildIndex(data_item, user, stop_time);
        sprintf(parameter_name, "top%d", topk);

    } else if (method_name == "QS") {
        const int n_sample = para.n_sample;
        const int n_sample_query = para.n_sample_query;
        const int sample_topk = para.sample_topk;
        const int n_thread = para.n_thread == -1 ? omp_get_num_procs() : para.n_thread;
        spdlog::info("input parameter: n_sample {} n_sample_query {} sample_topk {} n_thread {}",
                     n_sample, n_sample_query, sample_topk, n_thread);
        index = QS::BuildIndex(data_item, user, dataset_name,
                               n_sample, n_sample_query, sample_topk, n_thread, index_dir);
        sprintf(parameter_name, "top%d-n_sample_%d-n_sample_query_%d-sample_topk_%d-n_thread_%d",
                topk, n_sample, n_sample_query, sample_topk, n_thread);

    } else if (method_name == "QSRPNormalLP") {
        const int n_sample = para.n_sample;
        const int n_sample_query = para.n_sample_query;
        const int sample_topk = para.sample_topk;
        const int n_thread = para.n_thread == -1 ? omp_get_num_procs() : para.n_thread;
        spdlog::info("input parameter: n_sample {} n_sample_query {} sample_topk {} n_thread {}",
                     n_sample, n_sample_query, sample_topk, n_thread);
        index = QSRPNormalLP::BuildIndex(data_item, user, dataset_name,
                                         n_sample, n_sample_query, sample_topk, n_thread, index_dir);
        sprintf(parameter_name, "top%d-n_sample_%d-n_sample_query_%d-sample_topk_%d-n_thread_%d",
                topk, n_sample, n_sample_query, sample_topk, n_thread);

    }  else if (method_name == "QSRPUniformLP") {
        const int n_sample = para.n_sample;
        const int n_sample_query = para.n_sample_query;
        const int sample_topk = para.sample_topk;
        spdlog::info("input parameter: n_sample {} n_sample_query {} sample_topk {}",
                     n_sample, n_sample_query, sample_topk);
        index = QSRPUniformLP::BuildIndex(data_item, user, dataset_name,
                                          n_sample, n_sample_query, sample_topk, index_dir);
        sprintf(parameter_name, "top%d-n_sample_%d-n_sample_query_%d-sample_topk_%d",
                topk, n_sample, n_sample_query, sample_topk);

    } else if (method_name == "Rtree") {
        const size_t stop_time = para.stop_time;
        spdlog::info("input parameter: stop_time {}s", stop_time);
        index = Rtree::BuildIndex(data_item, user, stop_time);
        sprintf(parameter_name, "top%d", topk);

    } else if (method_name == "SimpferPP") {
        const int simpfer_k_max = para.simpfer_k_max;
        const size_t stop_time = para.stop_time;
        spdlog::info("input parameter: simpfer_k_max {}, stop_time {}s",
                     simpfer_k_max, stop_time);
        index = SimpferPP::BuildIndex(data_item, user, simpfer_k_max, stop_time);
        sprintf(parameter_name, "top%d-simpfer_k_max_%d", topk, simpfer_k_max);

    } else if (method_name == "US") {
        const int n_sample = para.n_sample;
        const int n_thread = para.n_thread == -1 ? omp_get_num_procs() : para.n_thread;
        spdlog::info("input parameter: n_sample {} n_thread {}", n_sample, n_thread);
        index = US::BuildIndex(data_item, user, dataset_name, index_dir, n_sample, n_thread);
        sprintf(parameter_name, "top%d-n_sample_%d-n_thread_%d", topk, n_sample, n_thread);

    } else {
        spdlog::error("not such method");
        exit(-1);
    }

    double build_index_time = record.get_elapsed_time_second();
    spdlog::info("finish preprocess and save the index");

    int n_execute_query = n_query_item;
//    if (method_name == "Simpfer" || method_name == "SimpferOnly") {
//        n_execute_query = 100;
//    }

    RetrievalResult config;

//    {
//        vector<SingleQueryPerformance> query_performance_l(n_execute_query);
//        vector<vector<UserRankElement>> result_rk = index->Retrieval(query_item, 1, n_execute_query,
//                                                                     query_performance_l);
//    }

    vector<SingleQueryPerformance> query_performance_l(n_execute_query);
    vector<vector<UserRankElement>> result_rk = index->Retrieval(query_item, topk, n_execute_query,
                                                                 query_performance_l);
    string performance_str = index->PerformanceStatistics(topk);
    config.AddRetrievalInfo(performance_str);

//    {
//        vector<SingleQueryPerformance> query_performance_l2(n_execute_query);
//        vector<vector<UserRankElement>> result_rk2 = index->Retrieval(query_item, 1, n_execute_query,
//                                                                      query_performance_l2);
//    }

    index->FinishCompute();

    spdlog::info("finish top-{}", topk);
    spdlog::info("{}", performance_str);

    spdlog::info("build index time: total {}s", build_index_time);

    cout << config.GetConfig(0) << endl;
    WriteRankResult(result_rk, dataset_name, method_name.c_str(), parameter_name);
    WriteQueryPerformance(query_performance_l, dataset_name, method_name.c_str(), parameter_name);

    config.AddMemoryInfo(index->IndexSizeByte());
    config.AddBuildIndexTime(build_index_time);
    config.AddExecuteQuery(n_execute_query);
    char modified_method_name[256];
    sprintf(modified_method_name, "retrieval-%s", method_name.c_str());
    config.WritePerformance(dataset_name, modified_method_name, parameter_name);
    return 0;
}