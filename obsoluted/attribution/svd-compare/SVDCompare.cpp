//
// Created by BianZheng on 2022/3/17.
//

#include <vector>
#include "alg/SVD.hpp"
#include "util/VectorIO.hpp"
#include "struct/VectorMatrix.hpp"
#include <string>
#include <spdlog/spdlog.h>

void WriteDistribution(const char *dataset_name, const char *component_name,
                       const std::vector<std::pair<double, double>> &distribution_l) {

    char resPath[256];
    std::sprintf(resPath, "../../result/attribution/SVDCompare/%s-%s-distribution.txt", dataset_name, component_name);
    std::ofstream out(resPath);
    if (!out) {
        spdlog::error("error in write result");
    }
    const int size = (int) distribution_l.size();
    for (int i = 0; i < size; i++) {
        std::pair<double, double> tmp_pair = distribution_l[i];
        out << tmp_pair.first << " " << tmp_pair.second << std::endl;
    }
    spdlog::info("write success {} {}", dataset_name, component_name);
}

using namespace std;
using namespace ReverseMIPS;

int main(int argc, char **argv) {
    if (!(argc == 2 or argc == 3)) {
        cout << argv[0] << " dataset_name [basic_dir]" << endl;
        return 0;
    }
    const char *dataset_name = argv[1];
    const char *basic_dir = "/home/bianzheng/Dataset/ReverseMIPS";
    if (argc == 3) {
        basic_dir = argv[2];
    }
    printf("attribution SVD compare dataset_name %s, basic_dir %s\n", dataset_name, basic_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    spdlog::info("before SVD transformation");

    vector<std::pair<double, double>> user_distribution_l(vec_dim);
    for (int dim = 0; dim < vec_dim; dim++) {
        user_distribution_l[dim].first = 1.0 * dim / vec_dim;
        user_distribution_l[dim].second = 0;
    }
    for (int dim = 0; dim < vec_dim; dim++) {
        for (int userID = 0; userID < n_user; userID++) {
            user_distribution_l[dim].second += user.getVector(userID)[dim];
        }
        user_distribution_l[dim].second /= n_user;
    }
    WriteDistribution(dataset_name, "before_user", user_distribution_l);

    vector<std::pair<double, double>> item_distribution_l(vec_dim);
    for (int dim = 0; dim < vec_dim; dim++) {
        item_distribution_l[dim].first = 1.0 * dim / vec_dim;
        item_distribution_l[dim].second = 0;
    }
    for (int dim = 0; dim < vec_dim; dim++) {
        for (int itemID = 0; itemID < n_data_item; itemID++) {
            item_distribution_l[dim].second += data_item.getVector(itemID)[dim];
        }
        item_distribution_l[dim].second /= n_data_item;
    }
    WriteDistribution(dataset_name, "before_item", item_distribution_l);


    VectorMatrix transfer_item;
    const double SIGMA = 0.7;
    SVD svd;
    svd.Preprocess(user, data_item, SIGMA);

    for (int dim = 0; dim < vec_dim; dim++) {
        user_distribution_l[dim].first = 1.0 * dim / vec_dim;
        user_distribution_l[dim].second = 0;
    }
    for (int dim = 0; dim < vec_dim; dim++) {
        for (int userID = 0; userID < n_user; userID++) {
            user_distribution_l[dim].second += user.getVector(userID)[dim];
        }
        user_distribution_l[dim].second /= n_user;
    }
    WriteDistribution(dataset_name, "after_user", user_distribution_l);

    for (int dim = 0; dim < vec_dim; dim++) {
        item_distribution_l[dim].first = 1.0 * dim / vec_dim;
        item_distribution_l[dim].second = 0;
    }
    for (int dim = 0; dim < vec_dim; dim++) {
        for (int itemID = 0; itemID < n_data_item; itemID++) {
            item_distribution_l[dim].second += data_item.getVector(itemID)[dim];
        }
        item_distribution_l[dim].second /= n_data_item;
    }
    WriteDistribution(dataset_name, "after_item", item_distribution_l);


    return 0;
}