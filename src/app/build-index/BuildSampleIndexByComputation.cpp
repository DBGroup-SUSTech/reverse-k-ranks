//
// Created by BianZheng on 2022/11/5.
//

#include "util/NameTranslation.hpp"

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/VectorMatrix.hpp"

#include "score_computation/ComputeScoreTableBatch.hpp"

#include "alg/RankBoundRefinement/SampleSearch.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string dataset_dir, dataset_name, method_name, index_dir;
    int n_sample, n_sample_query, sample_topk;
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
            ("method_name, mn",
             po::value<std::string>(&para.method_name)->default_value("QueryRankSampleSearchKthRank"),
             "method_name")

            ("n_sample, ns", po::value<int>(&para.n_sample)->default_value(-1),
             "number of sample of a rank bound")
            ("n_sample_query, nsq", po::value<int>(&para.n_sample_query)->default_value(9000),
             "number of sampled query")
            ("sample_topk, st", po::value<int>(&para.sample_topk)->default_value(600),
             "topk of sampled query");

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

void BuildIndex(const VectorMatrix &data_item, const VectorMatrix &user,
                const int64_t &n_sample, const int &n_sample_query, const int &sample_topk,
                const char *dataset_name, const string &index_name, const char *index_basic_dir,
                double &total_init_time, double &total_compute_sort_time,
                double &total_build_memory_index_time, double &total_save_index_time) {
    const int n_user = user.n_vector_;
    const int n_data_item = data_item.n_vector_;

    //rank search
    const bool load_sample_score = false;
    const bool is_query_distribution = index_name != "US";
    SampleSearch rank_search(
            index_basic_dir, dataset_name, index_name.c_str(),
            n_sample, load_sample_score, is_query_distribution, n_sample_query, sample_topk);

    total_init_time = 0;
    total_compute_sort_time = 0;
    total_build_memory_index_time = 0;
    total_save_index_time = 0;

    double batch_compute_sort_time = 0;
    double batch_build_memory_index_time = 0;

    //Compute Score Table
    TimeRecord record;
    record.reset();
    ComputeScoreTableBatch cstb(user, data_item);
    total_init_time = record.get_elapsed_time_second();

    const uint64_t batch_n_user = cstb.ComputeBatchUser();;
    const int remainder = n_user % batch_n_user == 0 ? 0 : 1;
    const int n_batch = n_user / (int) batch_n_user + remainder;
    spdlog::info("{} user per batch, n_batch {}", batch_n_user, n_batch);

    std::vector<float> batch_distance_l(batch_n_user * n_data_item);

    record.reset();
    cstb.init((int) batch_n_user, batch_distance_l.data());
    total_init_time += record.get_elapsed_time_second();

    const uint32_t report_every = 30;

    TimeRecord batch_record, operation_record;
    batch_record.reset();

    for (int batchID = 0; batchID < n_batch; batchID++) {
        const int start_userID = (int) batch_n_user * batchID;

        operation_record.reset();
        const int n_user_batch = n_user - start_userID > batch_n_user ? (int) batch_n_user : n_user - start_userID;
        cstb.ComputeSortItemsBatch(start_userID, n_user_batch, batch_distance_l.data());
        const double tmp_compute_sort_time = operation_record.get_elapsed_time_second();

        operation_record.reset();
#pragma omp parallel for default(none) shared(n_user_batch, batch_distance_l, n_data_item, start_userID, rank_search) num_threads(omp_get_max_threads())
        for (int batch_userID = 0; batch_userID < n_user_batch; batch_userID++) {
            const float *distance_ptr = batch_distance_l.data() + (size_t) batch_userID * n_data_item;
            const int userID = batch_userID + start_userID;
            rank_search.LoopPreprocess(distance_ptr, userID);
        }
        const double tmp_build_memory_index_time = operation_record.get_elapsed_time_second();

        total_compute_sort_time += tmp_compute_sort_time;
        total_build_memory_index_time += tmp_build_memory_index_time;

        batch_compute_sort_time += tmp_compute_sort_time;
        batch_build_memory_index_time += tmp_build_memory_index_time;

        if (batchID % report_every == 0) {
            spdlog::info(
                    "preprocessed {:.1f}%, Mem: {} Mb, {:.4f} s/iter, compute sort time {:.4f}s, build memory index time {:.4f}s",
                    batchID / (0.01 * n_batch), get_current_RSS() / 1000000,
                    batch_record.get_elapsed_time_second(),
                    batch_compute_sort_time, batch_build_memory_index_time);
            batch_compute_sort_time = 0;
            batch_build_memory_index_time = 0;
            batch_record.reset();
        }
    }

    record.reset();
    const bool save_sample_score = true;
    rank_search.SaveIndex(index_basic_dir, dataset_name, index_name.c_str(),
                          save_sample_score, is_query_distribution,
                          n_sample_query, sample_topk);
    total_save_index_time += record.get_elapsed_time_second();

    cstb.FinishCompute(batch_distance_l.data());
}

double ComputeMemoryByNumSample(const int &n_user, const int &n_sample) {
    return 1.0 * n_user * n_sample * sizeof(float) / 1024 / 1024 / 1024;
}

int main(int argc, char **argv) {
    Parameter para;
    LoadOptions(argc, argv, para);
    const char *dataset_name = para.dataset_name.c_str();
    const char *dataset_dir = para.dataset_dir.c_str();
    string index_dir = para.index_dir;
    const char *method_name = para.method_name.c_str();
    const std::string index_name = SampleSearchIndexName(method_name);
    spdlog::info("BuildSampleIndexByComputation dataset_name {}, method_name {}, dataset_dir {}",
                 dataset_name, method_name, dataset_dir);
    spdlog::info("index_name {}, index_dir {}", index_name, index_dir);

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(dataset_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    const int n_sample = para.n_sample;
    const double memory_capacity = ComputeMemoryByNumSample(n_user, n_sample);
    spdlog::info("n_sample {}, memory_capacity {:.2f}GB",
                 n_sample, memory_capacity);

    double total_init_time = 0, total_compute_sort_time = 0, total_build_memory_index_time = 0, total_save_index_time = 0;
    BuildIndex(data_item, user,
               n_sample, para.n_sample_query, para.sample_topk,
               dataset_name, index_name, index_dir.c_str(),
               total_init_time, total_compute_sort_time,
               total_build_memory_index_time, total_save_index_time);

    spdlog::info("finish preprocess and save the index");


    const double index_construction_time = total_init_time + total_compute_sort_time + total_build_memory_index_time;
    spdlog::info(
            "total_index_construction_time: {:.2f}s, total_init_time: {:.2f}s, total_compute_sort_time: {:.2f}s, total_build_memory_index_time: {:.2f}s, total_save_index_time: {:.2f}s",
            index_construction_time, total_init_time, total_compute_sort_time, total_build_memory_index_time,
            total_save_index_time);

    RetrievalResult config;
    char parameter_name[128];
    sprintf(parameter_name, "%s-n_sample_%d", index_name.c_str(), n_sample);

    char build_index_info[256];
    sprintf(build_index_info,
            "total_index_construction_time %.3fs, total_init_time %.3fs, total_compute_sort_time %.3fs, total_build_memory_index_time %.3fs, total_save_index_time %.3fs",
            index_construction_time, total_init_time, total_compute_sort_time, total_build_memory_index_time,
            total_save_index_time);
    config.AddInfo(build_index_info);

    config.WritePerformance(dataset_name, "BuildSampleIndexByComputation", parameter_name);
    return 0;
}