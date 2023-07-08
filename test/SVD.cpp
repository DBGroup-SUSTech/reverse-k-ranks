#include <vector>
#include "alg/SVD.hpp"
#include "util/VectorIO.hpp"
#include "struct/VectorMatrix.hpp"
#include <string>
#include <spdlog/spdlog.h>

using namespace std;
using namespace ReverseMIPS;

inline bool DoubleEqual(double a, double b) {
    return std::abs(a - b) < 0.0001;
}

int main(int argc, char **argv) {
//    if (!(argc == 2 or argc == 3)) {
//        cout << argv[0] << " dataset_name [basic_dir]" << endl;
//        return 0;
//    }
//    const char *dataset_name = argv[1];
    const char *dataset_name = "fake";
    const char *basic_dir = "/home/bianzheng/Dataset/ReverseMIPS";
    if (argc == 3) {
        basic_dir = argv[2];
    }
    printf("test dataset_name %s, basic_dir %s\n", dataset_name, basic_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    vector<double> before_data_IP_l(n_data_item * n_user);
    vector<double> before_query_IP_l(n_query_item * n_user);

    for (int itemID = 0; itemID < n_data_item; itemID++) {
        for (int userID = 0; userID < n_user; userID++) {
            double *item_vecs = data_item.getVector(itemID);
            double *user_vecs = user.getVector(userID);
            double ip = InnerProduct(user_vecs, item_vecs, vec_dim);
            before_data_IP_l[itemID * n_user + userID] = ip;
        }
    }
    for (int queryID = 0; queryID < n_query_item; queryID++) {
        for (int userID = 0; userID < n_user; userID++) {
            double *query_vecs = query_item.getVector(queryID);
            double *user_vecs = user.getVector(userID);
            double ip = InnerProduct(query_vecs, user_vecs, vec_dim);
            before_query_IP_l[queryID * n_user + userID] = ip;
        }
    }

    VectorMatrix transfer_item;
    const double SIGMA = 0.7;
    SVD svd;
    svd.Preprocess(user, data_item, SIGMA);

    for (int queryID = 0; queryID < n_query_item; queryID++) {
        svd.TransferItem(query_item.getVector(queryID), vec_dim);
    }

    for (int itemID = 0; itemID < n_data_item; itemID++) {
        for (int userID = 0; userID < n_user; userID++) {
            double *item_vecs = data_item.getVector(itemID);
            double *user_vecs = user.getVector(userID);
            double ip = InnerProduct(item_vecs, user_vecs, vec_dim);
            assert(DoubleEqual(before_data_IP_l[itemID * n_user + userID], ip));
        }
    }
    for (int queryID = 0; queryID < n_query_item; queryID++) {
        for (int userID = 0; userID < n_user; userID++) {
            double *query_vecs = query_item.getVector(queryID);
            double *user_vecs = user.getVector(userID);
            double ip = InnerProduct(query_vecs, user_vecs, vec_dim);
            assert(DoubleEqual(before_query_IP_l[queryID * n_user + userID], ip));
        }
    }


    return 0;
}