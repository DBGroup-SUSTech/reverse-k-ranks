//
// Created by BianZheng on 2022/5/20.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"

#include "BuildIndexIPBound/BaseIPBound.hpp"
#include "BuildIndexIPBound/FullDim.hpp"
#include "BuildIndexIPBound/FullInt.hpp"
#include "BuildIndexIPBound/FullNorm.hpp"
#include "BuildIndexIPBound/Grid.hpp"
#include "BuildIndexIPBound/PartDimPartInt.hpp"
#include "BuildIndexIPBound/PartDimPartNorm.hpp"
#include "BuildIndexIPBound/PartIntPartNorm.hpp"

#include "BuildIndexIPBound.hpp"
#include "FileIO.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string basic_dir, dataset_name, bound_name;
    int scale, n_codebook, n_codeword, n_sample_item;
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
            ("bound_name, bn", po::value<std::string>(&para.bound_name)->default_value("PartIntPartNorm"),
             "bound_name")
            ("scale, s", po::value<int>(&para.scale)->default_value(1000),
             "scale for integer bound")
            ("n_codebook, ncb", po::value<int>(&para.n_codebook)->default_value(8),
             "number of codebook")
            ("n_codeword, ncw", po::value<int>(&para.n_codeword)->default_value(64),
             "number of codeword")
            ("n_sample_item, nsi", po::value<int>(&para.n_sample_item)->default_value(128),
             "number of sampled item");

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
    spdlog::info("BuildIndexIPBound bound_name {} dataset_name {}, basic_dir {}", bound_name, dataset_name, basic_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    char parameter_name[256] = "";
    std::unique_ptr<BaseIPBound> IPbound_ptr;

    if (bound_name == "Grid") {
        const int n_codeword = para.n_codeword;
        const int min_codeword = std::floor(std::sqrt(1.0 * 80 * std::sqrt(3 * user.vec_dim_)));
        int n_cur_codeword = 1;
        while (n_cur_codeword < min_codeword) {
            n_cur_codeword = n_cur_codeword << 1;
        }
        spdlog::info("input parameter: min_codework {}, n_cur_codeword {}, n_codeword {}",
                     min_codeword, n_cur_codeword, n_codeword);
        IPbound_ptr = std::make_unique<Grid>(n_user, n_data_item, vec_dim, n_codeword);
        sprintf(parameter_name, "codeword_%d", n_codeword);

    } else if (bound_name == "FullDim") {
        spdlog::info("input parameter: none");
        IPbound_ptr = std::make_unique<FullDim>(n_user, n_data_item, vec_dim);

    } else if (bound_name == "FullNorm") {
        spdlog::info("input parameter: none");
        IPbound_ptr = std::make_unique<FullNorm>(n_user, n_data_item, vec_dim);

    } else if (bound_name == "FullInt") {
        const int scale = para.scale;
        spdlog::info("input parameter: scale {}", scale);
        IPbound_ptr = std::make_unique<FullInt>(n_user, n_data_item, vec_dim, scale);
        sprintf(parameter_name, "scale_%d", scale);

    } else if (bound_name == "PartDimPartInt") {
        const int scale = para.scale;
        spdlog::info("CAPartDimPartInt scale {}", scale);
        IPbound_ptr = std::make_unique<PartDimPartInt>(n_user, n_data_item, vec_dim, scale);
        sprintf(parameter_name, "scale_%d", scale);

    } else if (bound_name == "PartDimPartNorm") {
        spdlog::info("input parameter: none");
        IPbound_ptr = std::make_unique<PartDimPartNorm>(n_user, n_data_item, vec_dim);

    } else if (bound_name == "PartIntPartNorm") {
        const int scale = para.scale;
        spdlog::info("input parameter: scale {}", scale);
        IPbound_ptr = std::make_unique<PartIntPartNorm>(n_user, n_data_item, vec_dim, scale);
        sprintf(parameter_name, "scale_%d", scale);

    } else {
        spdlog::error("not found IPBound name, program exit");
        exit(-1);
    }

    uint64_t bucket_size_var = 0;
    TimeRecord record;
    record.reset();
    unique_ptr<BaseIndex> index = BuildIndexIPBound::BuildIndex(data_item, user, IPbound_ptr, bucket_size_var);
    const double time_used = record.get_elapsed_time_second();
    spdlog::info("bound_name {}, build index time {}s, bucket_size_var {}", bound_name, time_used, bucket_size_var);

    double build_index_time = record.get_elapsed_time_second();
    spdlog::info("finish preprocess and save the index");

    vector<int> topk_l{30, 20, 10};
    RetrievalResult2 config;
    vector<vector<vector<UserRankElement>>> result_rank_l;
    for (int topk: topk_l) {
        record.reset();
        std::vector<SingleQueryPerformance> query_performance_l(n_query_item);
        vector<vector<UserRankElement>> result_rk = index->Retrieval(query_item, topk,
                                                                     n_query_item, query_performance_l);

        double retrieval_time = record.get_elapsed_time_second();

        string performance_str = index->PerformanceStatistics(topk);
        config.AddRetrievalInfo(performance_str);

        result_rank_l.emplace_back(result_rk);
        spdlog::info("finish top-{}", topk);
    }

    spdlog::info("build index time: total {}s", build_index_time);
    int n_topk = (int) topk_l.size();

    for (int i = 0; i < n_topk; i++) {
        cout << config.GetConfig(i) << endl;
        const int topk = topk_l[i];
        WriteRankResult2(result_rank_l[i], topk, dataset_name, bound_name.c_str(), parameter_name);
    }

    config.AddBuildIndexInfo(index->BuildIndexStatistics());
    config.AddBuildIndexTime(build_index_time);
    config.WritePerformance(dataset_name, bound_name.c_str(), parameter_name);
    return 0;
}