//
// Created by BianZheng on 2022/8/11.
//

//对item进行采样, 计算每一个item, 返回reverse k-rank结果所在的userID, 以及返回这个item的topk userID

#include "ComputeItemIDScoreTable.hpp"
#include "FileIO.hpp"
#include "QueryDistributionUserRank.hpp"
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
    int n_sample_item;
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

            ("n_sample_item, ns", po::value<int>(&para.n_sample_item)->default_value(500),
             "number of sample of a rank bound");

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
    spdlog::info("QueryDistributionUserRank dataset_name {}, basic_dir {}", dataset_name, basic_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    std::vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                              vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user,
                 vec_dim);

    user.vectorNormalize();

    const int n_sample_item = para.n_sample_item;
    assert(n_sample_item <= n_data_item);
    spdlog::info("n_sample_item {}", n_sample_item);

    std::vector<int> sample_itemID_l(n_sample_item);
    SampleItem(n_data_item, n_sample_item, sample_itemID_l);

    double compute_score_table_time = 0;
    std::vector<int> sorted_user_rank_l(n_sample_item * n_user);

    ReverseMIPS::TimeRecord record;
    record.reset();
    ComputeSortUserRank(user, data_item,
                        sample_itemID_l, n_sample_item,
                        sorted_user_rank_l, compute_score_table_time);
    const double compute_query_distribution_time = record.get_elapsed_time_second();

    spdlog::info("QueryDistributionUserRank finish, Compute Score Table Time {}s, Compute Query Distribution Time {}s",
                 compute_score_table_time, compute_query_distribution_time);

    WriteSortedUserRank(sample_itemID_l, sorted_user_rank_l, n_sample_item, n_user, dataset_name);

    return 0;
}