//
// Created by BianZheng on 2022/11/5.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/VectorMatrix.hpp"
#include "NameTranslation.hpp"

#include "alg/RegressionPruning/BaseLinearRegression.hpp"
#include "alg/RegressionPruning/DirectLinearRegression.hpp"
#include "GlobalLinearRegression.hpp"
#include "../LinearProgrammingRegressionIndex/MinMaxLinearRegression.hpp"
#include "alg/RankBoundRefinement/SampleSearch.hpp"
#include "../LinearProgrammingRegressionIndex/UniformLinearRegression.hpp"

#include "score_computation/ComputeScoreTable.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string dataset_dir, dataset_name, index_dir, method_name;
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
            ("method_name, tm",
             po::value<std::string>(&para.method_name)->default_value("QueryRankSampleDirectIntLR"),
             "method name")

            ("n_sample, ns", po::value<int>(&para.n_sample)->default_value(-1),
             "number of sample of a rank bound")
            ("n_sample_query, nsq", po::value<int>(&para.n_sample_query)->default_value(150),
             "number of sample query")
            ("sample_topk, st", po::value<int>(&para.sample_topk)->default_value(60),
             "sample topk");

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

void BuildLocalIndex(const VectorMatrix &data_item, const VectorMatrix &user,
                     const std::vector<int64_t> &n_sample_l, const int &n_sample_query, const int &sample_topk,
                     const char *dataset_name, const char *basic_index_dir, const std::string &method_name) {
    const int n_user = user.n_vector_;
    const int n_data_item = data_item.n_vector_;

    //rank search
    const int n_rs_ins = (int) n_sample_l.size();
    std::vector<std::unique_ptr<BaseLinearRegression>> lr_l;

    std::vector<SampleSearch> rs_ins_l(n_rs_ins);
    const bool load_sample_score = true;
    const bool is_query_distribution = true;
    for (int rsID = 0; rsID < n_rs_ins; rsID++) {
        const int n_sample = (int) n_sample_l[rsID];
        const string index_name = IndexName(method_name);

        if (index_name == "QueryRankSampleSearchUniformRank") {
            rs_ins_l[rsID] = SampleSearch(basic_index_dir, dataset_name, "QueryRankSampleSearchUniformRank",
                                          n_sample, load_sample_score, is_query_distribution,
                                          n_sample_query, sample_topk);
        } else if (index_name == "QueryRankSampleSearchKthRank") {
            rs_ins_l[rsID] = SampleSearch(basic_index_dir, dataset_name, "QueryRankSampleSearchKthRank",
                                          n_sample, load_sample_score, is_query_distribution,
                                          n_sample_query, sample_topk);
        }else{
            spdlog::error("not such index, program exit");
            exit(-1);
        }

        if (method_name == "QueryRankSampleDirectIntLR") {
            lr_l.push_back(std::make_unique<DirectLinearRegression>(n_data_item, n_user));

        } else if (method_name == "QueryRankSampleMinMaxIntLR" || method_name == "QueryRankSampleSearchUniformRankMinMaxIntLR") {
            lr_l.push_back(std::make_unique<MinMaxLinearRegression>(n_data_item, n_user, method_name));
//            lr_l.emplace_back(MinMaxLinearRegression(n_data_item, n_user));

        } else if (method_name == "QueryRankSampleUniformIntLR" || method_name == "QueryRankSampleSearchUniformRankUniformIntLR") {
            lr_l.push_back(std::make_unique<UniformLinearRegression>(n_data_item, n_user, method_name));
//            lr_l.emplace_back(MinMaxLinearRegression(n_data_item, n_user));

        } else {
            spdlog::error("no such training method, program exit");
            exit(-1);
        }
        lr_l[rsID]->StartPreprocess(rs_ins_l[rsID].known_rank_idx_l_.get(), n_sample);

    }

    TimeRecord record;
    record.reset();
    const int report_every = 4000;

    for (int userID = 0; userID < n_user; userID++) {
        for (int rsID = 0; rsID < n_rs_ins; rsID++) {
            lr_l[rsID]->LoopPreprocess(rs_ins_l[rsID].SampleData(userID), userID);
        }
        if (userID % report_every == 0) {
            std::cout << "preprocessed " << userID / (0.01 * n_user) << " %, "
                      << record.get_elapsed_time_second() << " s/iter" << " Mem: "
                      << get_current_RSS() / 1000000 << " Mb \n";
            record.reset();
        }
    }

    for (int rsID = 0; rsID < n_rs_ins; rsID++) {
        lr_l[rsID]->FinishPreprocess();
        lr_l[rsID]->SaveIndex(basic_index_dir, dataset_name, n_sample_query, sample_topk);
    }

}

void BuildGlobalIndex(const VectorMatrix &data_item, const VectorMatrix &user,
                      const std::vector<int64_t> &n_sample_l, const int &n_sample_query, const int &sample_topk,
                      const char *dataset_name, const char *basic_index_dir, const std::string &method_name) {
    const int n_user = user.n_vector_;
    const int n_data_item = data_item.n_vector_;

    //rank search
    const int n_rs_ins = (int) n_sample_l.size();
    std::vector<GlobalLinearRegression> lr_l;

    const bool load_sample_score = true;
    const bool is_query_distribution = true;
    lr_l.emplace_back(GlobalLinearRegression(n_data_item, n_user));
    lr_l[0].StartPreprocess();

    TimeRecord record;
    record.reset();
    const int report_every = 4000;

    //Compute Score Table
    ComputeScoreTable cst(user, data_item);
    std::vector<double> distance_l(n_data_item);

    for (int userID = 0; userID < n_user; userID++) {
        cst.ComputeSortItems(userID, distance_l.data());

        lr_l[0].LoopPreprocess(distance_l.data(), userID);
        if (userID % report_every == 0) {
            std::cout << "preprocessed " << userID / (0.01 * n_user) << " %, "
                      << record.get_elapsed_time_second() << " s/iter" << " Mem: "
                      << get_current_RSS() / 1000000 << " Mb \n";
            record.reset();
        }
    }
    cst.FinishCompute();

    lr_l[0].FinishPreprocess();
    lr_l[0].SaveIndex(basic_index_dir, dataset_name);

}

int main(int argc, char **argv) {
    Parameter para;
    LoadOptions(argc, argv, para);
    const char *dataset_name = para.dataset_name.c_str();
    const char *dataset_dir = para.dataset_dir.c_str();
    string index_dir = para.index_dir;
    const string method_name = para.method_name;
    spdlog::info("BuildIntLRByComputation dataset_name {}, dataset_dir {}",
                 dataset_name, dataset_dir);
    spdlog::info("index_dir {}, method_name {}", index_dir, method_name);
    spdlog::info("n_sample_query {}, sample_topk {}", para.n_sample_query, para.sample_topk);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(dataset_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];

    user.vectorNormalize();
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    std::vector<int> memory_capacity_l = {2, 4, 8, 16, 32};
    std::vector<int64_t> n_sample_l(memory_capacity_l.size());
    int n_capacity = (int) memory_capacity_l.size();
    for (int capacityID = 0; capacityID < n_capacity; capacityID++) {
        const int64_t memory_capacity = memory_capacity_l[capacityID];

        //n_sample is computed as the int linear regression
        const int64_t n_sample = (memory_capacity * 1024 * 1024 * 1024 -
                                  n_user * 4 * sizeof(double) -
                                  n_user * vec_dim * sizeof(int) - n_data_item * vec_dim * sizeof(int)) /
                                 sizeof(double) / n_user;
        n_sample_l[capacityID] = n_sample;
    }

    if (para.n_sample != -1) {
        n_sample_l.clear();
        n_sample_l.push_back(para.n_sample);
        n_capacity = (int) memory_capacity_l.size();
    } else {
        for (int capacityID = 0; capacityID < n_capacity; capacityID++) {
            spdlog::info("memory_capacity {}, n_sample {}",
                         memory_capacity_l[capacityID], n_sample_l[capacityID]);
        }
    }

    std::string n_sample_info = "n_sample: ";
    const int sample_length = (int) n_sample_l.size();
    for (int sampleID = 0; sampleID < sample_length; sampleID++) {
        n_sample_info += std::to_string(n_sample_l[sampleID]) + " ";
    }
    spdlog::info("{}", n_sample_info);

    TimeRecord record;
    record.reset();

    if (method_name == "QueryRankSampleGlobalIntLR") {
        BuildGlobalIndex(data_item, user,
                         n_sample_l, para.n_sample_query, para.sample_topk,
                         dataset_name, index_dir.c_str(), method_name);
    } else {
        BuildLocalIndex(data_item, user,
                        n_sample_l, para.n_sample_query, para.sample_topk,
                        dataset_name, index_dir.c_str(), method_name);
    }

    double build_index_time = record.get_elapsed_time_second();
    spdlog::info("finish preprocess and save the index");

    RetrievalResult config;

    spdlog::info("BuildIntLRByComputation build index time: total {}s", build_index_time);

    char parameter_name[128];
    sprintf(parameter_name, "QueryRankSearchIntLR");
    config.AddInfo(n_sample_info);
    config.AddBuildIndexTime(build_index_time);
    config.WritePerformance(dataset_name, "BuildIntLRByComputation", parameter_name);
    return 0;
}