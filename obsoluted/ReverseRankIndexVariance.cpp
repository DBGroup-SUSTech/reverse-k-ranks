//
// Created by BianZheng on 2022/6/1.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"

#include "BruteForce/BatchDiskBruteForce.hpp"
#include "DiskIndexCompress/CompressTopTIDIPBruteForce.hpp"
#include "ScoreSample/CompressTopTIPBruteForce.hpp"
#include "BruteForce/DiskBruteForce.hpp"
#include "BruteForce/MemoryBruteForce.hpp"
#include "OnlineBruteForce.hpp"

#include "GridIndex.hpp"

#include "BPlusTree/BPlusTree.hpp"
#include "ScoreSample/SSMergeRankByInterval.hpp"
#include "ScoreSample/ScoreSample.hpp"
#include "RankSample.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

void
BuildRedundantQuery(const ReverseMIPS::VectorMatrix &query_item, const int &n_repeat,
                    std::vector<ReverseMIPS::VectorMatrix> &add_query_l) {
    assert(add_query_l.size() == query_item.n_vector_);
    const int n_query = query_item.n_vector_;
    const int vec_dim = query_item.vec_dim_;
    for (int queryID = 0; queryID < n_query; queryID++) {
        std::unique_ptr<double[]> repeat_query_ptr = std::make_unique<double[]>(n_repeat * vec_dim);
        const double *query_vecs = query_item.getVector(queryID);
        for (int repeatID = 0; repeatID < n_repeat; repeatID++) {
            double *tmp_query_vecs = repeat_query_ptr.get() + repeatID * vec_dim;
            memcpy(tmp_query_vecs, query_vecs, vec_dim * sizeof(double));
        }
        add_query_l[queryID].init(repeat_query_ptr, n_repeat, vec_dim);
    }
}

class Parameter {
public:
    std::string basic_dir, dataset_name, method_name;
    int cache_bound_every, n_interval, topt_perc, n_repeat;
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
            ("method_name, mn", po::value<std::string>(&para.method_name)->default_value("BatchDiskBruteForce"),
             "method_name")

            ("cache_bound_every, cbe", po::value<int>(&para.cache_bound_every)->default_value(512),
             "how many numbers would cache a value")
            ("n_interval, nitv", po::value<int>(&para.n_interval)->default_value(1024),
             "the numer of interval")
            ("topt_perc, ttp", po::value<int>(&para.topt_perc)->default_value(50),
             "store percent of top-t inner product as index")
            ("n_repeat, nr", po::value<int>(&para.n_repeat)->default_value(1000),
             "the number of repeat time for query");

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

    vector<VectorMatrix> redundant_query_l(n_query_item);
    BuildRedundantQuery(query_item, para.n_repeat, redundant_query_l);

    char index_path[256];
    sprintf(index_path, "../index/index");

    TimeRecord record;
    record.reset();
    unique_ptr<BaseIndex> index;
    char parameter_name[256] = "";
    if (method_name == "BatchDiskBruteForce") {
        ///BruteForce
        spdlog::info("input parameter: none");
        index = BatchDiskBruteForce::BuildIndex(data_item, user, index_path);

    } else if (method_name == "CompressTopTIDIPBruteForce") {
        const int cache_bound_every = para.cache_bound_every;
        const int n_interval = para.n_interval;
        const int topt_perc = para.topt_perc;
        spdlog::info("input parameter: cache_bound_every {}, n_interval {}, topt_perc {}",
                     cache_bound_every, n_interval, topt_perc);
        index = CompressTopTIDIPBruteForce::BuildIndex(data_item, user, index_path, cache_bound_every, n_interval,
                                                       topt_perc);
        sprintf(parameter_name, "cache_bound_every_%d-n_interval_%d-topt_perc_%d", cache_bound_every, n_interval,
                topt_perc);

    } else if (method_name == "CompressTopTIPBruteForce") {
        const int cache_bound_every = para.cache_bound_every;
        const int n_interval = para.n_interval;
        const int topt_perc = para.topt_perc;
        spdlog::info("input parameter: cache_bound_every {}, n_interval {}, topt_perc {}",
                     cache_bound_every, n_interval, topt_perc);
        index = CompressTopTIPBruteForce::BuildIndex(data_item, user, index_path, cache_bound_every, n_interval,
                                                     topt_perc);
        sprintf(parameter_name, "cache_bound_every_%d-n_interval_%d-topt_perc_%d", cache_bound_every, n_interval,
                topt_perc);

    } else if (method_name == "DiskBruteForce") {
        spdlog::info("input parameter: none");
        index = DiskBruteForce::BuildIndex(data_item, user, index_path);

    } else if (method_name == "MemoryBruteForce") {
        spdlog::info("input parameter: none");
        index = MemoryBruteForce::BuildIndex(data_item, user);

    } else if (method_name == "OnlineBruteForce") {
        spdlog::info("input parameter: none");
        index = OnlineBruteForce::BuildIndex(data_item, user);

    } else if (method_name == "GridIndex") {
        ///Online
        spdlog::info("input parameter: none");
        index = GridIndex::BuildIndex(data_item, user);

    } else if (method_name == "BPlusTree") {
        ///Proposed method
        const int cache_bound_every = para.cache_bound_every;
        spdlog::info("input parameter: node_size {}", cache_bound_every);
        index = BPlusTree::BuildIndex(data_item, user, index_path, cache_bound_every);
        sprintf(parameter_name, "node_size_%d", cache_bound_every);

    } else if (method_name == "HRBMergeRankBound") {
        const int cache_bound_every = para.cache_bound_every;
        const int n_interval = para.n_interval;
        const int topt_perc = para.topt_perc;
        spdlog::info("input parameter: cache_bound_every {}, n_interval {}, topt_perc {}",
                     cache_bound_every, n_interval, topt_perc);
        index = HRBMergeRankBound::BuildIndex(data_item, user, index_path, cache_bound_every, n_interval, topt_perc);
        sprintf(parameter_name, "cache_bound_every_%d-n_interval_%d-topt_perc_%d", cache_bound_every, n_interval,
                topt_perc);

    } else if (method_name == "RankSample") {
        const int cache_bound_every = para.cache_bound_every;
        spdlog::info("input parameter: cache_bound_every {}", cache_bound_every);
        index = RankSample::BuildIndex(data_item, user, index_path, cache_bound_every);
        sprintf(parameter_name, "cache_bound_every_%d", cache_bound_every);

    } else if (method_name == "ScoreSample") {
        const int n_interval = para.n_interval;
        spdlog::info("input parameter: n_interval {}", n_interval);
        index = ScoreSample::BuildIndex(data_item, user, index_path, n_interval);
        sprintf(parameter_name, "n_interval_%d", n_interval);

    } else {
        spdlog::error("not such method");
    }

    double build_index_time = record.get_elapsed_time_second();
    spdlog::info("finish preprocess and save the index");

    vector<int> topk_l{70, 60, 50, 40, 30, 20, 10};
//    vector<int> topk_l{10000, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8};
    const int n_topk = (int) topk_l.size();
    RedundantRetrievalResult config;
    config.config_l_l.resize(n_topk);
    config.performance_metric_name = index->VariancePerformanceMetricName();

    for (int topkID = 0; topkID < n_topk; topkID++) {
        const int topk = topk_l[topkID];
        for (int queryID = 0; queryID < n_query_item; queryID++) {
            const VectorMatrix &tmp_query = redundant_query_l[queryID];
            record.reset();

            vector<vector<UserRankElement>> result_rk = index->Retrieval(tmp_query, topk);
            double retrieval_time = record.get_elapsed_time_second();
            double second_per_query = retrieval_time / n_query_item;

            string performance_str = index->VariancePerformanceStatistics(retrieval_time, second_per_query, queryID);
            config.config_l_l[topkID].push_back(performance_str);
        }

        spdlog::info("finish top-{}", topk);
    }

    spdlog::info("build index time: total {}s", build_index_time);

    config.WritePerformance(dataset_name, method_name.c_str(), parameter_name, topk_l);
    return 0;
}