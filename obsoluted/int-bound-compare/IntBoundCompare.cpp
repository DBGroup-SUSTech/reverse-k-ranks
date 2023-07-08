//
// Created by BianZheng on 2022/10/24.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"
#include "alg/QueryIPBound/FullInt.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string dataset_dir, dataset_name;
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
    const char *dataset_dir = para.dataset_dir.c_str();
    spdlog::info("IntBoundCompare dataset_name {}, dataset_dir {}", dataset_name, dataset_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(dataset_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    user.vectorNormalize();

    const int report_every = 10000;
    TimeRecord batch_record;
    batch_record.reset();
    TimeRecord record;
    record.reset();
    FullInt ip_bound_ins(n_user, vec_dim, 1000);
    ip_bound_ins.Preprocess(user, data_item);
    std::vector<std::pair<double, double>> ip_bound_l(n_user);

    double ip_bound_val = 0;
    for (int itemID = 0; itemID < n_data_item; itemID++) {
        const double *item_vecs = data_item.getVector(itemID);
        ip_bound_ins.IPBound(item_vecs, user, ip_bound_l);
        for (int userID = 0; userID < n_user; userID++) {
            ip_bound_val += ip_bound_l[userID].first;
        }
        if (itemID != 0 && itemID % report_every == 0) {
            std::cout << "preprocessed " << itemID / (0.01 * n_data_item) << " %, "
                      << batch_record.get_elapsed_time_second() << " s/iter" << " Mem: "
                      << get_current_RSS() / 1000000 << " Mb \n";
            batch_record.reset();
        }
    }

    const double ip_bound_time = record.get_elapsed_time_second();

    record.reset();
    batch_record.reset();

    double ip_val = 0;
    for (int itemID = 0; itemID < n_data_item; itemID++) {
        const double *item_vecs = data_item.getVector(itemID);
        for (int userID = 0; userID < n_user; userID++) {
            const double *user_vecs = user.getVector(userID);
            const double ip = InnerProduct(item_vecs, user_vecs, vec_dim);
            ip_val += ip;
        }
        if (itemID != 0 && itemID % report_every == 0) {
            std::cout << "preprocessed " << itemID / (0.01 * n_data_item) << " %, "
                      << batch_record.get_elapsed_time_second() << " s/iter" << " Mem: "
                      << get_current_RSS() / 1000000 << " Mb \n";
            batch_record.reset();
        }
    }
    const double ip_time = record.get_elapsed_time_second();

    assert(ip_val > ip_bound_val);
    printf("ip_val %.3f, ip_bound_val %.3f\n", ip_val, ip_bound_val);
    spdlog::info("finish preprocess and save the index");

    spdlog::info("ip_bound_time {}s, ip_time {}s", ip_bound_time, ip_time);

    char resPath[256];
    std::sprintf(resPath, "../../result/attribution/%s-%s-config.txt",
                 "IntBoundCompare", dataset_name);
    std::ofstream file(resPath);
    if (!file) {
        spdlog::error("error in write result");
    }
    char str_info[256];
    sprintf(str_info, "ip_bound_time %.3fs, ip_time %.3fs", ip_bound_time, ip_time);
    file << str_info << std::endl;
    file.close();
    return 0;
}
