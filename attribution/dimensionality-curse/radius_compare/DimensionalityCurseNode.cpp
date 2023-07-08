//
// Created by bianzheng on 2023/4/26.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "struct/VectorMatrix.hpp"

#include "BalltreeNode.hpp"
#include "RtreeNode.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <ios>
#include <vector>
#include <string>
#include <iomanip>

class Parameter {
public:
    std::string dataset_dir, dataset_name, method_name;
};

void LoadOptions(int argc, char **argv, Parameter &para) {
    namespace po = boost::program_options;

    po::options_description opts("Allowed options");
    opts.add_options()
            ("help,h", "help info")
            ("dataset_dir,dd",
             po::value<std::string>(&para.dataset_dir)->default_value("/home/bianzheng/Dataset/ReverseMIPS"),
             "the basic directory of dataset")
            ("dataset_name, dn", po::value<std::string>(&para.dataset_name)->default_value("fake-normal-30d"),
             "dataset_name")
            ("method_name, mn", po::value<std::string>(&para.method_name)->default_value("Balltree"),
             "method_name");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << opts << std::endl;
        exit(0);
    }
}

void
WritePerformance(const std::vector<std::pair<int, double>> &height_size_l,
                 const char *dataset_name, const char *method_name, const char *index_name) {
    char resPath[256];
    std::sprintf(resPath, "../../result/attribution/DimensionalityCurse-%s-%s-%s-performance.txt",
                 dataset_name, method_name, index_name);
    std::ofstream file(resPath);
    if (!file) {
        spdlog::error("error in write result");
    }
    const size_t arr_size = height_size_l.size();
    file << "num_descendant,size" << std::endl;
    for (int i = 0; i < arr_size; i++) {
        const int height = height_size_l[i].first;
        const double size = height_size_l[i].second;
        file << setiosflags(std::ios::fixed) << std::setprecision(3) << height << "," << size << std::endl;
    }
    file.close();

}

using namespace std;
using namespace ReverseMIPS;


int main(int argc, char **argv) {
    Parameter para;
    LoadOptions(argc, argv, para);
    const char *dataset_name = para.dataset_name.c_str();
    const char *dataset_dir = para.dataset_dir.c_str();
    string method_name = para.method_name;
    spdlog::info("DimensionalityCurse method_name {}, dataset_name {}", method_name, dataset_name);
    spdlog::info("dataset_dir {}", dataset_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(dataset_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    TimeRecord record;
    record.reset();
    std::vector<std::pair<int, double>> data_item_node_size_l;
//    std::vector<std::pair<int, double>> user_height_size_l;
    if (method_name == "Balltree") {
        data_item_node_size_l = ReverseMIPS::BalltreeNode::HeightSizeCurve(data_item);

    } else if (method_name == "Rtree") {
        data_item_node_size_l = ReverseMIPS::RtreeNode::HeightSizeCurve(data_item);
//        user_height_size_l = ReverseMIPS::Rtree::HeightSizeCurve(user);

    } else {
        spdlog::error("not such method");
    }

    const int n_output = 100;
    std::cout << "data_item_node_size_l" << std::endl;
    const size_t node_size = data_item_node_size_l.size();
    const size_t n_skip = node_size / n_output;
    for (size_t i = 0; i < node_size; i += n_skip) {
        const std::pair<int, double> p = data_item_node_size_l[i];
        cout << "sample output " << i << ", # element in a node: " << p.first << ", radius: " << p.second << endl;
    }
//    std::cout << "user_height_size_l" << std::endl;
//    std::for_each(user_height_size_l.begin(), user_height_size_l.end(),
//                  [](std::pair<int, double> &p) { cout << "height: " << p.first << " size: " << p.second << endl; });
    WritePerformance(data_item_node_size_l, dataset_name, method_name.c_str(), "data_item");
//    WritePerformance(user_height_size_l, dataset_name, method_name.c_str(), "user");
    return 0;
}