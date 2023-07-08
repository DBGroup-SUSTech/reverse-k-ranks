//
// Created by BianZheng on 2022/1/25.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"
#include "IntervalVector.hpp"
#include <iostream>
#include <vector>
#include <string>

//预处理时不做任何动作, 在线计算全部的向量, 然后返回最大的k个rank

using namespace std;
using namespace ReverseMIPS;

/*
 * 首先进行merge用户, 然后建立索引, 根据指定的方向进行merge
 */

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
    printf("IntervalVector dataset_name %s, basic_dir %s\n", dataset_name, basic_dir);
    int n_merge_user = 1024;

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
    TimeRecord record;
    record.reset();
    IntervalVector::Index index = IntervalVector::BuildIndex(user, data_item, n_merge_user, dataset_name,
                                                             component_time_l);
    double build_index_time = record.get_elapsed_time_second();
    double bf_index_time = component_time_l[0];
    printf("finish building index\n");

    vector<int> topk_l{10, 20, 30, 40, 50};
    vector<IntervalVector::RetrievalResult> retrieval_res_l;
    vector<vector<vector<UserRankElement>>> result_rank_l;
    for (int topk: topk_l) {
        record.reset();
        vector<vector<UserRankElement>> result_rk = index.Retrieval(query_item, topk);

        double retrieval_time = record.get_elapsed_time_second();
        double self_inner_product_time = index.self_inner_product_time_;
        double brute_force_search_time = index.brute_force_search_time_;
        double second_per_query = retrieval_time / n_query_item;

        result_rank_l.emplace_back(result_rk);
        retrieval_res_l.emplace_back(retrieval_time, self_inner_product_time, brute_force_search_time, second_per_query,
                                     topk);
    }

    printf("build index: total time %.3fs, bruteforce index time %.3fs\n", build_index_time, bf_index_time);
    int n_topk = (int) topk_l.size();
    for (int i = 0; i < n_topk; i++) {
        cout << retrieval_res_l[i].ToString() << endl;
        writeRank(result_rank_l[i], dataset_name, "IntervalVector");
    }

    map<string, string> performance_m;
    performance_m.emplace("build bruteforce index time", double2string(bf_index_time));
    performance_m.emplace("build index time", double2string(build_index_time));
    for (int i = 0; i < n_topk; i++) {
        retrieval_res_l[i].AddMap(performance_m);
    }
    writePerformance(dataset_name, "IntervalVector", performance_m);

    return 0;
}