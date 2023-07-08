//
// Created by BianZheng on 2022/8/1.
//

#include "ScoreSamplePrintRankBound.hpp"
#include "RankSamplePrintRankBound.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/UserRankElement.hpp"
#include "util/TimeMemory.hpp"
#include "util/VectorIO.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <random>

class Parameter {
public:
    std::string basic_dir, dataset_name;
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
             "dataset_name");

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
    spdlog::info("PrintItemScoreScoreSampleIndex dataset_name {}, basic_dir {}", dataset_name, basic_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    std::vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                              vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user,
                 vec_dim);

    user.vectorNormalize();

    std::vector<int> queryID_l = {0};
    if (para.dataset_name == "amazon") {
        //line 429, 477, 480, 486, 491
        queryID_l = {428, 476, 479, 485, 490};
    } else if (para.dataset_name == "goodreads") {
        //line 1, 2, 3, 4, 5
        queryID_l = {0, 1, 2, 3, 4};
    }

    char index_basic_dir[128];
//    sprintf(index_basic_dir, "../../index/%s_constructed_index", dataset_name);
    sprintf(index_basic_dir, "../../index");

    {
        char score_sample_128_path[256];
        sprintf(score_sample_128_path, "%s/%s_ScoreSearch%d.index", index_basic_dir, dataset_name, 128);
        ScoreSamplePrintRankBound::MeasurePruneRatio(dataset_name, basic_dir, score_sample_128_path, 128, queryID_l);
    }

    {
        char rank_sample_128_path[256];
        sprintf(rank_sample_128_path, "%s/%s_RankSearch%d.index", index_basic_dir, dataset_name, 128);
        RankSamplePrintRankBound::MeasurePruneRatio(dataset_name, basic_dir, rank_sample_128_path, 128, queryID_l);
    }

    return 0;
}