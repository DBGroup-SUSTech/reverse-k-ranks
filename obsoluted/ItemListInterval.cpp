//
// Created by BianZheng on 2022/1/25.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"
#include "ItemListInterval.hpp"
#include <iostream>
#include <vector>
#include <string>

//预处理时不做任何动作, 在线计算全部的向量, 然后返回最大的k个rank

using namespace std;
using namespace ReverseMIPS;

/*
 * 首先进行merge用户, 然后建立索引, 根据指定的方向进行merge
 */
//deprecated
int main(int argc, char **argv) {
    if (!(argc == 3 or argc == 4)) {
        cout << argv[0] << " dataset_name top-k [basic_dir]" << endl;
        return 0;
    }
    const char *dataset_name = argv[1];
    int topk = atoi(argv[2]);
    const char *basic_dir = "/home/bianzheng/Dataset/ReverseMIPS";
    if (argc == 4) {
        basic_dir = argv[3];
    }
    printf("ItemListInterval dataset_name %s, topk %d, basic_dir %s\n", dataset_name, topk, basic_dir);
    int n_merge_user = 100;

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<unique_ptr<double[]>>
            data = readData(basic_dir, dataset_name, n_data_item, n_query_item, n_user, vec_dim);
    double *data_item_ptr = data[0].get();
    double *user_ptr = data[1].get();
    double *query_item_ptr = data[2].get();
    printf("n_data_item %d, n_query_item %d, n_user %d, vec_dim %d\n", n_data_item, n_query_item, n_user, vec_dim);

    VectorMatrix data_item, user, query_item;
    data_item.init(data_item_ptr, n_data_item, vec_dim);
    user.init(user_ptr, n_user, vec_dim);
    user.vectorNormalize();
    query_item.init(query_item_ptr, n_query_item, vec_dim);
    n_merge_user = std::min(n_merge_user, n_user / 2);

    vector<double> component_time_l;
    TimeRecord timeRecord;
    timeRecord.reset();
    ItemListIntervalIndex itemListIntervalIndex = BuildIndex(user, data_item, n_merge_user, dataset_name, component_time_l);
    double build_index_time = timeRecord.get_elapsed_time_second();
    double bf_index_time = component_time_l[0];
    printf("finish building index\n");

    timeRecord.reset();
    vector<vector<RankElement>> result = itemListIntervalIndex.Retrieval(query_item, topk);
    double retrieval_time = timeRecord.get_elapsed_time_second();

    printf("build index: bruteforce index time %.3fs\n", bf_index_time);
    printf("build index time %.3fs, retrieval time %.3fs\n",
           build_index_time, retrieval_time);
    writeRank(result, dataset_name, "ItemListInterval");

    map<string, string> performance_m;
    performance_m.emplace("build bruteforce index time", double2string(bf_index_time));
    performance_m.emplace("build index time", double2string(build_index_time));
    performance_m.emplace("retrieval time", double2string(retrieval_time));
    writePerformance(dataset_name, "ItemListInterval", performance_m);

    return 0;
}
