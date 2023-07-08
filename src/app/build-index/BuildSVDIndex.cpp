//
// Created by bianzheng on 2023/5/16.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "alg/SVD.hpp"
#include "struct/VectorMatrix.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string dataset_dir, dataset_name, index_dir;
    float SIGMA;
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

            ("SIGMA, sigma",
             po::value<float>(&para.SIGMA)->default_value(0.7),
             "SIGMA value for SVD");

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

void BuildSVDIndex(const VectorMatrix &data_item, const VectorMatrix &user, const float &SIGMA,
                   const char *index_basic_dir, const char *dataset_name,
                   double &build_index_time) {
    TimeRecord record;
    record.reset();
    SVD svd_ins(user, data_item, SIGMA);
    build_index_time = record.get_elapsed_time_second();

    svd_ins.SaveIndex(index_basic_dir, dataset_name);

}

int main(int argc, char **argv) {
    Parameter para;
    LoadOptions(argc, argv, para);
    const char *dataset_name = para.dataset_name.c_str();
    const char *dataset_dir = para.dataset_dir.c_str();
    string index_dir = para.index_dir;
    const string program_name = "BuildSVDIndex";
    spdlog::info("{} dataset_name {}, dataset_dir {}",
                 program_name, dataset_name, dataset_dir);
    spdlog::info("index_dir {}", index_dir);
    spdlog::info("SIGMA {:.2f}", para.SIGMA);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(dataset_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    double build_index_time = 0;
    BuildSVDIndex(data_item, user, para.SIGMA, index_dir.c_str(), dataset_name, build_index_time);

    spdlog::info("finish preprocess and save the index");

    RetrievalResult config;

    spdlog::info("{} build index time: total {}s", program_name, build_index_time);

    config.AddBuildIndexTime(build_index_time);
    config.WritePerformance(dataset_name, program_name.c_str(), "");
    return 0;
}