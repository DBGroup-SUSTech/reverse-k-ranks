//
// Created by bianzheng on 2023/5/29.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/VectorMatrix.hpp"

#include "simpfer_plus_plus/SimpferPlusPlusRetrieval.hpp"
#include "score_computation/ComputeScoreTableBatch.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string dataset_dir, dataset_name, index_dir;
};

void LoadOptions(int argc, char **argv, Parameter &para) {
    namespace po = boost::program_options;

    po::options_description opts("Allowed options");
    opts.add_options()
            ("help,h", "help info")
            ("dataset_dir,dd",
             po::value<std::string>(&para.dataset_dir)->default_value("/home/bianzheng/Dataset/ReverseMIPS"),
             "the basic directory of dataset")
            ("dataset_name, ds", po::value<std::string>(&para.dataset_name)->default_value("fake-normal"),
             "dataset_name")
            ("index_dir, id",
             po::value<std::string>(&para.index_dir)->default_value("/home/bianzheng/reverse-k-ranks/index"),
             "the directory of the index");

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

void BuildScoreTable(VectorMatrix &user, VectorMatrix &data_item,
                     const char *index_path,
                     double &total_init_time, double &total_compute_sort_time,
                     double &total_save_index_time,
                     double &estimate_compute_sort_time,
                     double &estimate_save_index_time) {
    const int n_user = user.n_vector_;
    const int n_data_item = data_item.n_vector_;
    const int vec_dim = user.vec_dim_;

    std::vector<int> cache_topk_l;
    for (int topk = 1; topk < n_data_item; topk *= 2) {
        cache_topk_l.emplace_back(topk);
    }
    if (std::find(cache_topk_l.begin(), cache_topk_l.end(), n_user) == cache_topk_l.end()) {
        cache_topk_l.emplace_back(n_data_item);
    }

    SimpferPlusPlusCacheIndex simpfer_ins(cache_topk_l, n_user, n_data_item, vec_dim);

    total_compute_sort_time = 0;
    total_save_index_time = 0;

    double batch_compute_sort_time = 0;
    double batch_save_index_time = 0;

    //Compute Score Table
    total_init_time = 0;
    TimeRecord record;
    record.reset();
    ComputeScoreTableBatch cstb(user, data_item);
    total_init_time += record.get_elapsed_time_second();

    const uint64_t batch_n_user = cstb.ComputeBatchUser();;
    const int remainder = n_user % batch_n_user == 0 ? 0 : 1;
    const int n_batch = n_user / (int) batch_n_user + remainder;
    spdlog::info("{} user per batch, n_batch {}", batch_n_user, n_batch);

    std::vector<float> batch_distance_float_l(batch_n_user * n_data_item);

    record.reset();
    cstb.init((int) batch_n_user, batch_distance_float_l.data());
    total_init_time += record.get_elapsed_time_second();

    const uint32_t report_every = 30;

    TimeRecord batch_record, operation_record;
    batch_record.reset();

    int n_complete_user = 0;
    for (int batchID = 0; batchID < n_batch; batchID++) {
        const int start_userID = (int) batch_n_user * batchID;

        operation_record.reset();
        const int n_user_batch = n_user - start_userID > batch_n_user ? (int) batch_n_user : n_user - start_userID;
        cstb.ComputeSortItemsBatch(start_userID, n_user_batch, batch_distance_float_l.data());
        const double tmp_compute_sort_time = operation_record.get_elapsed_time_second();

        operation_record.reset();
        simpfer_ins.BuildIndexLoop(batch_distance_float_l.data(), n_user_batch);
        const double tmp_save_index_time = operation_record.get_elapsed_time_second();

        total_compute_sort_time += tmp_compute_sort_time;
        total_save_index_time += tmp_save_index_time;

        batch_compute_sort_time += tmp_compute_sort_time;
        batch_save_index_time += tmp_save_index_time;

        n_complete_user += (int) batch_n_user;

        if (batchID % report_every == 0) {
            spdlog::info(
                    "preprocessed {:.1f}%, Mem: {} Mb, {} s/iter, compute sort time {}s, process index time {}s",
                    batchID / (0.01 * n_batch), get_current_RSS() / 1000000,
                    batch_record.get_elapsed_time_second(),
                    batch_compute_sort_time, batch_save_index_time);
            batch_compute_sort_time = 0;
            batch_save_index_time = 0;
            batch_record.reset();
        }
    }
    estimate_compute_sort_time = total_compute_sort_time / n_complete_user * n_user;
    estimate_save_index_time = total_save_index_time / n_complete_user * n_user;
    cstb.FinishCompute(batch_distance_float_l.data());
    simpfer_ins.SaveIndex(index_path);
}


int main(int argc, char **argv) {
    Parameter para;
    LoadOptions(argc, argv, para);
    const char *dataset_name = para.dataset_name.c_str();
    const char *dataset_dir = para.dataset_dir.c_str();
    string index_dir = para.index_dir;
    spdlog::info("BuildSimpferPPCacheIndex dataset_name {}, dataset_dir {}", dataset_name, dataset_dir);
    spdlog::info("index_dir {}", index_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(dataset_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    char index_path[256];
    sprintf(index_path, "%s/rmips_index/%s.index", index_dir.c_str(), dataset_name);

    TimeRecord record;
    record.reset();

    double total_init_time, total_compute_sort_time, total_save_index_time;
    double estimate_compute_sort_time, estimate_save_index_time;

    BuildScoreTable(user, data_item,
                    index_path,
                    total_init_time, total_compute_sort_time, total_save_index_time,
                    estimate_compute_sort_time, estimate_save_index_time);

    double build_index_time = record.get_elapsed_time_second();
    spdlog::info("finish preprocess and save the index");

    RetrievalResult config;

    spdlog::info(
            "build index time: total {}s, init time {}s, compute sort time {}s, save index time {}s",
            build_index_time, total_init_time, total_compute_sort_time, total_save_index_time);
    spdlog::info(
            "init time {}s, estimate compute sort time {}s, estimate save index time {}s",
            total_init_time, estimate_compute_sort_time, estimate_save_index_time);
    const double estimate_gpu_process_time = total_init_time + estimate_compute_sort_time;
    spdlog::info("estimate gpu processing time {}s",
                 estimate_gpu_process_time);

    char build_index_info[256];
    sprintf(build_index_info,
            "compute sort time %.3f s, save index time %.3f s, estimate gpu process time %.3fs, estimate compute sort time %.3f s, estimate save index time %.3f s",
            total_compute_sort_time, total_save_index_time,
            estimate_gpu_process_time,
            estimate_compute_sort_time, estimate_save_index_time);

    config.AddInfo(build_index_info);
    config.AddBuildIndexTime(build_index_time);

#ifdef USE_GPU
    config.WritePerformance(dataset_name, "BuildScoreTableBatchGPU", "GPU");
#else
    config.WritePerformance(dataset_name, "BuildScoreTableBatchCPU", "CPU");
#endif
    return 0;
}