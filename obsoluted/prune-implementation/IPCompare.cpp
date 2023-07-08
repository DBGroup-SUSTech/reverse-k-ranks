//
// Created by BianZheng on 2022/3/20.
//
#include <iostream>
#include <spdlog/spdlog.h>
#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/RankBoundElement.hpp"
#include "util/TimeMemory.hpp"
#include "util/VectorIO.hpp"

using namespace ReverseMIPS;

double IPSingle(const VectorMatrix &user, const VectorMatrix &item, std::vector<double> &IP_l) {
    const int n_user = user.n_vector_;
    const int n_item = item.n_vector_;
    const int vec_dim = item.vec_dim_;

    IP_l.resize(n_user * n_item);

    TimeRecord record;
    record.reset();
    double sum = 0;
    for (int userID = 0; userID < n_user; userID++) {
        const double *user_vecs = user.getVector(userID);
        for (int itemID = 0; itemID < n_item; itemID++) {
            const double *item_vecs = item.getVector(itemID);
            double IP = InnerProduct(user_vecs, item_vecs, vec_dim);
            IP_l[userID * n_item + itemID] = IP;
            sum += IP_l[userID * n_item + itemID];
        }
    }
    double time_used = record.get_elapsed_time_second();
    printf("single sum %.3f\n", sum);
    return time_used;
}

double IPIntegrate(const VectorMatrix &user, const VectorMatrix &item, std::vector<RankBoundElement> &IP_l) {
    const int n_user = user.n_vector_;
    const int n_item = item.n_vector_;
    const int vec_dim = item.vec_dim_;

    IP_l.resize(n_user * n_item);

    TimeRecord record;
    record.reset();
    double sum = 0;
    for (int userID = 0; userID < n_user; userID++) {
        const double *user_vecs = user.getVector(userID);
        for (int itemID = 0; itemID < n_item; itemID++) {
            const double *item_vecs = item.getVector(itemID);
            double IP = InnerProduct(user_vecs, item_vecs, vec_dim);
            IP_l[userID * n_item + itemID].lower_bound_ = IP;
            sum += IP_l[userID * n_item + itemID].lower_bound_;
        }
    }
    double time_used = record.get_elapsed_time_second();
    printf("integrate sum %.3f\n", sum);
    return time_used;
}

void AttributionWrite(const double &single_used, const double &integrate_used, const char *dataset_name) {

    char resPath[256];
    std::sprintf(resPath, "../../result/attribution/Prune/IPCompare-%s.txt", dataset_name);
    std::ofstream file(resPath);
    if (!file) {
        std::printf("error in write result\n");
    }

    file << "dataset name:" << dataset_name << ", single used " << single_used << "s, integrate used " << integrate_used
         << std::endl;

    file.close();
}

using namespace std;

//output bound tightness and time consumption
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
    spdlog::info("IPCompare dataset_name {}, basic_dir {}", dataset_name, basic_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user,
                 vec_dim);

    std::vector<double> single_l;
    double single_used = IPSingle(user, query_item, single_l);

    std::vector<RankBoundElement> integrate_l;
    double integrate_used = IPIntegrate(user, query_item, integrate_l);

    spdlog::info("IPCompare single {}s, integrate {}s", single_used, integrate_used);

    AttributionWrite(single_used, integrate_used, dataset_name);
    return 0;
}