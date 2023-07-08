//
// Created by BianZheng on 2022/8/11.
//

//算出每一个query的topk结果, 对score table的时间进行计数
//通过动态规划算出最优的采样方式, 作为自己的时间计算
//通过暴力算出最优的采样方式, 作为baseline的时间计算

//对item进行采样, 计算每一个item, 返回reverse k-rank结果所在的userID, 以及返回这个item的topk userID

#include "FileIO.hpp"
#include "BruteForceChooseSample.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"
#include "util/TimeMemory.hpp"
#include "util/VectorIO.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <string>

class Parameter {
public:
    std::string basic_dir, dataset_name;
    int n_sample_item, n_sample_rank, topk;
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

            ("n_sample_item, nsi", po::value<int>(&para.n_sample_item)->default_value(500),
             "number of sampled query workload")
            ("n_sample_rank, nsr", po::value<int>(&para.n_sample_rank)->default_value(20),
             "number of rank sampled in memory index")
            ("topk, tk", po::value<int>(&para.topk)->default_value(50),
             "top-k");

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
    spdlog::info("BruteForceChooseSample dataset_name {}, basic_dir {}", dataset_name, basic_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    std::vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                              vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}",
                 n_data_item, n_query_item, n_user, vec_dim);

    user.vectorNormalize();

    const int n_sample_item = para.n_sample_item;
    const int n_sample_rank = para.n_sample_rank;
    const int topk = para.topk;
    assert(n_sample_item <= n_data_item);
    spdlog::info("n_sample_item {}, n_sample_rank {}, topk {}", n_sample_item, n_sample_rank, topk);

    std::vector<int> sorted_user_rank_l(n_sample_item * n_user);
    ReadSortedUserRank(n_sample_item, n_user, dataset_name, sorted_user_rank_l);

    std::vector<int> best_sample_rank_l(n_sample_rank);
    int64_t min_cost = -1;

    spdlog::info("start compute all combination");
    TimeRecord record;
    record.reset();
    ReverseMIPS::CalcAllCombination(sorted_user_rank_l,
                                    n_sample_item, n_user, n_data_item,
                                    n_sample_rank, topk,
                                    best_sample_rank_l, min_cost);
    const double &time = record.get_elapsed_time_second();


    spdlog::info("BruteForceChooseSample finish, Find Cost Time {}s, Min Cost {}",
                 time, min_cost);
    std::copy(best_sample_rank_l.begin(), best_sample_rank_l.end(), std::ostream_iterator<int>(std::cout, " "));

    return 0;
}