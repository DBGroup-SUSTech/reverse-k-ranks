//
// Created by BianZheng on 2022/10/21.
//

///given the sampled rank, build the QueryRankSample index

#include "util/NameTranslation.hpp"

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/VectorMatrix.hpp"

#include "alg/DiskIndex/ReadAll.hpp"
#include "alg/RankBoundRefinement/SampleSearch.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string dataset_dir, dataset_name, method_name, index_dir;
    int n_sample, n_sample_query, sample_topk;
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
            ("method_name, mn",
             po::value<std::string>(&para.method_name)->default_value("QueryRankSampleSearchKthRank"),
             "method_name")

            ("n_sample, ns", po::value<int>(&para.n_sample)->default_value(-1),
             "number of sample of a rank bound")
            ("n_sample_query, nsq", po::value<int>(&para.n_sample_query)->default_value(9000),
             "number of sampled query")
            ("sample_topk, st", po::value<int>(&para.sample_topk)->default_value(600),
             "topk of sampled query");

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

void BuildIndex(const VectorMatrix &data_item, const VectorMatrix &user,
                const int64_t &n_sample, const int &n_sample_query, const int &sample_topk,
                const char *score_table_path, const char *dataset_name, const string &index_name,
                const char *index_basic_dir) {
    const int n_user = user.n_vector_;
    const int n_data_item = data_item.n_vector_;

    //rank search
    const bool load_sample_score = false;
    const bool is_query_distribution = index_name != "US";
    SampleSearch rank_search(
            index_basic_dir, dataset_name, index_name.c_str(),
            n_sample, load_sample_score, is_query_distribution, n_sample_query, sample_topk);

    ReadAll read_ins(n_user, n_data_item, score_table_path);
    read_ins.RetrievalPreprocess();

    const int report_every = 10000;
    TimeRecord record;
    record.reset();
    std::vector<float> distance_l(n_data_item);
    for (int userID = 0; userID < n_user; userID++) {
        read_ins.ReadDiskNoCache(userID, distance_l);
        rank_search.LoopPreprocess(distance_l.data(), userID);

        if (userID % report_every == 0) {
            std::cout << "preprocessed " << userID / (0.01 * n_user) << " %, "
                      << record.get_elapsed_time_second() << " s/iter" << " Mem: "
                      << get_current_RSS() / 1000000 << " Mb \n";
            record.reset();
        }
    }

    const bool save_sample_score = true;
    rank_search.SaveIndex(index_basic_dir, dataset_name, index_name.c_str(),
                          save_sample_score, is_query_distribution,
                          n_sample_query, sample_topk);
    read_ins.FinishRetrieval();
}

double ComputeMemoryByNumSample(const int &n_user, const int &n_sample) {
    return 1.0 * n_user * n_sample * sizeof(float) / 1024 / 1024 / 1024;
}

int main(int argc, char **argv) {
    Parameter para;
    LoadOptions(argc, argv, para);
    const char *dataset_name = para.dataset_name.c_str();
    const char *dataset_dir = para.dataset_dir.c_str();
    string index_dir = para.index_dir;
    const char *method_name = para.method_name.c_str();
    const std::string index_name = SampleSearchIndexName(method_name);
    spdlog::info("BuildSampleIndexBySample dataset_name {}, method_name {}, dataset_dir {}",
                 dataset_name, method_name, dataset_dir);
    spdlog::info("index_name {}, index_dir {}", index_name, index_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(dataset_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    const int n_sample = para.n_sample;
    const double memory_capacity = ComputeMemoryByNumSample(n_user, n_sample);
    spdlog::info("n_sample {}, memory_capacity {:.2f}GB",
                 n_sample, memory_capacity);

    char score_table_path[256];
    sprintf(score_table_path, "%s/%s.index", index_dir.c_str(), dataset_name);

    TimeRecord record;
    record.reset();

    BuildIndex(data_item, user,
               n_sample, para.n_sample_query, para.sample_topk,
               score_table_path, dataset_name, index_name, index_dir.c_str());

    double build_index_time = record.get_elapsed_time_second();
    spdlog::info("finish preprocess and save the index");

    RetrievalResult config;

    spdlog::info("BuildSampleIndexBySample build index time: total {:.2f}s", build_index_time);

    char parameter_name[128];
    sprintf(parameter_name, "%s-n_sample_%d", index_name.c_str(), n_sample);
    config.AddBuildIndexTime(build_index_time);
    config.WritePerformance(dataset_name, "BuildSampleIndexBySample", parameter_name);
    return 0;
}