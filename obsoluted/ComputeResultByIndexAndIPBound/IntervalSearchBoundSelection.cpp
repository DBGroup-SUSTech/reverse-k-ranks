//
// Created by bianzheng on 2022/5/3.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"

#include "impl/IntervalSearch/IRBBallPrune.hpp"
#include "impl/IntervalSearch/IRBFullDimPrune.hpp"
#include "impl/IntervalSearch/IRBFullIntPrune.hpp"
#include "impl/IntervalSearch/IRBFullNormPrune.hpp"
#include "impl/IntervalSearch/IRBPartDimPartIntPrune.hpp"
#include "impl/IntervalSearch/IRBPartDimPartNormPrune.hpp"
#include "impl/IntervalSearch/IRBPartIntPartNormPrune.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string basic_dir, dataset_name, method_name;
    int cache_bound_every, n_interval;
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
            ("method_name, mn", po::value<std::string>(&para.method_name)->default_value("IRBBallPrune"),
             "method_name")

            ("cache_bound_every, cbe", po::value<int>(&para.cache_bound_every)->default_value(512),
             "how many numbers would cache a value")
            ("n_interval, nitv", po::value<int>(&para.n_interval)->default_value(1024),
             "the numer of interval");

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
    string method_name = para.method_name;
    spdlog::info("{} dataset_name {}, basic_dir {}", method_name, dataset_name, basic_dir);

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
    if (method_name == "IRBBallPrune") {
        const int cache_bound_every = para.cache_bound_every;
        const int n_interval = para.n_interval;
        spdlog::info("input parameter: cache_bound_every {}, n_interval {}",
                     cache_bound_every, n_interval);
        index = IRBBallPrune::BuildIndex(data_item, user, index_path, cache_bound_every, n_interval);
        sprintf(parameter_name, "cache_bound_every_%d-n_interval_%d", cache_bound_every, n_interval);

    } else if (method_name == "IRBFullDimPrune") {
        const int cache_bound_every = para.cache_bound_every;
        const int n_interval = para.n_interval;
        spdlog::info("input parameter: cache_bound_every {}, n_interval {}",
                     cache_bound_every, n_interval);
        index = IRBFullDimPrune::BuildIndex(data_item, user, index_path, cache_bound_every, n_interval);
        sprintf(parameter_name, "cache_bound_every_%d-n_interval_%d", cache_bound_every, n_interval);

    } else if (method_name == "IRBFullIntPrune") {
        const int cache_bound_every = para.cache_bound_every;
        const int n_interval = para.n_interval;
        spdlog::info("input parameter: cache_bound_every {}, n_interval {}",
                     cache_bound_every, n_interval);
        index = IRBFullIntPrune::BuildIndex(data_item, user, index_path, cache_bound_every, n_interval);
        sprintf(parameter_name, "cache_bound_every_%d-n_interval_%d", cache_bound_every, n_interval);

    } else if (method_name == "IRBFullNormPrune") {
        const int cache_bound_every = para.cache_bound_every;
        const int n_interval = para.n_interval;
        spdlog::info("input parameter: cache_bound_every {}, n_interval {}",
                     cache_bound_every, n_interval);
        index = IRBFullNormPrune::BuildIndex(data_item, user, index_path, cache_bound_every, n_interval);
        sprintf(parameter_name, "cache_bound_every_%d-n_interval_%d", cache_bound_every, n_interval);

    } else if (method_name == "IRBPartDimPartIntPrune") {
        const int cache_bound_every = para.cache_bound_every;
        const int n_interval = para.n_interval;
        spdlog::info("input parameter: cache_bound_every {}, n_interval {}",
                     cache_bound_every, n_interval);
        index = IRBPartDimPartIntPrune::BuildIndex(data_item, user, index_path, cache_bound_every, n_interval);
        sprintf(parameter_name, "cache_bound_every_%d-n_interval_%d", cache_bound_every, n_interval);

    } else if (method_name == "IRBPartDimPartNormPrune") {
        const int cache_bound_every = para.cache_bound_every;
        const int n_interval = para.n_interval;
        spdlog::info("input parameter: cache_bound_every {}, n_interval {}",
                     cache_bound_every, n_interval);
        index = IRBPartDimPartNormPrune::BuildIndex(data_item, user, index_path, cache_bound_every, n_interval);
        sprintf(parameter_name, "cache_bound_every_%d-n_interval_%d", cache_bound_every, n_interval);

    } else if (method_name == "IRBPartIntPartNormPrune") {
        const int cache_bound_every = para.cache_bound_every;
        const int n_interval = para.n_interval;
        spdlog::info("input parameter: cache_bound_every {}, n_interval {}",
                     cache_bound_every, n_interval);
        index = IRBPartIntPartNormPrune::BuildIndex(data_item, user, index_path, cache_bound_every, n_interval);
        sprintf(parameter_name, "cache_bound_every_%d-n_interval_%d", cache_bound_every, n_interval);

    }else{
        spdlog::error("not such method");
    }

    double build_index_time = record.get_elapsed_time_second();
    spdlog::info("finish preprocess and save the index");

    vector<int> topk_l{70, 60, 50, 40, 30, 20, 10};
//    vector<int> topk_l{20};
    RetrievalResult config;
    vector<vector<vector<UserRankElement>>> result_rank_l;
    for (int topk: topk_l) {
        record.reset();
        vector<vector<UserRankElement>> result_rk = index->Retrieval(query_item, topk);

        double retrieval_time = record.get_elapsed_time_second();
        double second_per_query = retrieval_time / n_query_item;

        string performance_str = index->PerformanceStatistics(topk, retrieval_time, second_per_query);
        config.config_l.push_back(performance_str);

        result_rank_l.emplace_back(result_rk);
        spdlog::info("finish top-{}", topk);
    }

    spdlog::info("build index time: total {}s", build_index_time);
    int n_topk = (int) topk_l.size();

    for (int i = 0; i < n_topk; i++) {
        cout << config.config_l[i] << endl;
        WriteRankResult(result_rank_l[i], dataset_name, method_name.c_str(), parameter_name);
    }

    config.AddBuildIndexInfo(index->BuildIndexStatistics());
    config.AddBuildIndexTime(build_index_time);
    config.WritePerformance(dataset_name, method_name.c_str(), parameter_name);
    return 0;
}