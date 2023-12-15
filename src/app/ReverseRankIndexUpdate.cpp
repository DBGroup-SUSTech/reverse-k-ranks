//
// Created by bianzheng on 2023/6/16.
//

#include "util/VectorIOUpdate.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrixUpdate.hpp"
#include "struct/MethodBaseUpdate.hpp"

#include "Update/QSUpdate.hpp"
#include "Update/QSRPNormalLPUpdate.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>

class Parameter {
public:
    std::string dataset_dir, dataset_name, method_name, index_dir;
    int topk;
    int n_sample, n_sample_query, sample_topk;
    int n_thread;
    std::string update_type, update_operator;
    int updateID;
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
            ("method_name, mn", po::value<std::string>(&para.method_name)->default_value("BatchDiskBruteForce"),
             "method_name")
            ("topk, tk",
             po::value<int>(&para.topk)->default_value(10),
             "the value of topk")
            ("update_type, ut", po::value<std::string>(&para.update_type)->default_value("user"),
             "update type, either user or item")
            ("update_operator, ut", po::value<std::string>(&para.update_operator)->default_value("insert"),
             "update operator, either insert or delete")
            ("updateID, uID", po::value<int>(&para.updateID)->default_value(0),
             "update ID")
            // memory index parameter
            ("n_sample, ns", po::value<int>(&para.n_sample)->default_value(20),
             "number of sample of a rank bound")
            ("n_sample_query, nsq", po::value<int>(&para.n_sample_query)->default_value(150),
             "the numer of sample query in training query distribution")
            ("sample_topk, st", po::value<int>(&para.sample_topk)->default_value(60),
             "topk in training query distribution")
            ("n_thread, nt", po::value<int>(&para.n_thread)->default_value(-1),
             "number of thread for processing");

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
    const char *index_dir = para.index_dir.c_str();
    string method_name = para.method_name;
    const int topk = para.topk;
    const std::string update_type = para.update_type;
    const std::string update_operator = para.update_operator;
    const int updateID = para.updateID;
    spdlog::info("Retrieval method_name {}, dataset_name {}, topk {}, update type {}, update operator {}, updateID {}",
                 method_name, dataset_name, topk,
                 update_type, update_operator, updateID);
    spdlog::info("dataset_dir {}", dataset_dir);
    spdlog::info("index_dir {}", index_dir);

    int n_data_item, n_query_item, n_user, n_update_user, n_update_item, vec_dim;
    vector<VectorMatrixUpdate> data = readDataUpdate(dataset_dir, dataset_name, update_type,
                                                     n_user, n_data_item, n_query_item,
                                                     n_update_user, n_update_item,
                                                     vec_dim);
    VectorMatrixUpdate &user = data[0];
    VectorMatrixUpdate &data_item = data[1];
    VectorMatrixUpdate &query_item = data[2];
    VectorMatrixUpdate &user_update = data[3];
    VectorMatrixUpdate &data_item_update = data[4];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    TimeRecord record;
    record.reset();
    unique_ptr<BaseUpdateIndex> index;
    char parameter_name[256] = "";
    if (method_name == "QSRPNormalLPUpdate") {
        const int n_sample = para.n_sample;
        const int n_sample_query = para.n_sample_query;
        const int sample_topk = para.sample_topk;
        const int n_thread = para.n_thread == -1 ? omp_get_num_procs() : para.n_thread;
        spdlog::info("input parameter: n_sample {} n_sample_query {} sample_topk {} n_thread {}",
                     n_sample, n_sample_query, sample_topk, n_thread);
        index = QSRPNormalLPUpdate::BuildIndex(data_item, user, dataset_name,
                                               n_sample, n_sample_query, sample_topk, n_thread, index_dir);
        sprintf(parameter_name, "top%d-n_sample_%d-n_sample_query_%d-sample_topk_%d-n_thread_%d",
                topk, n_sample, n_sample_query, sample_topk, n_thread);

    } else if (method_name == "QSUpdate") {
        const int n_sample = para.n_sample;
        const int n_sample_query = para.n_sample_query;
        const int sample_topk = para.sample_topk;
        const int n_thread = para.n_thread == -1 ? omp_get_num_procs() : para.n_thread;
        spdlog::info("input parameter: n_sample {} n_sample_query {} sample_topk {} n_thread {}",
                     n_sample, n_sample_query, sample_topk, n_thread);
        index = QSUpdate::BuildIndex(data_item, user, dataset_name,
                                     n_sample, n_sample_query, sample_topk, n_thread, index_dir);
        sprintf(parameter_name, "top%d-n_sample_%d-n_sample_query_%d-sample_topk_%d-n_thread_%d",
                topk, n_sample, n_sample_query, sample_topk, n_thread);

    } else {
        spdlog::error("not such method");
        exit(-1);
    }

    double build_index_time = record.get_elapsed_time_second();
    spdlog::info("finish preprocess and save the index");

    if (!(0 <= updateID && updateID <= 4)) {
        spdlog::error("updateID should be in [0, 4]");
        exit(-1);
    }
    const int n_update = 4;
    VectorMatrixUpdate update_data_vm;
    int n_sample_vecs;
    if (update_type == "user") {
        n_sample_vecs = n_update_user / n_update * updateID;
        update_data_vm = splitVectorMatrix(user_update, n_sample_vecs);

    } else if (update_type == "data_item") {
        n_sample_vecs = n_update_item / n_update * updateID;
        update_data_vm = splitVectorMatrix(data_item_update, n_sample_vecs);
    } else {
        spdlog::error("no such update type");
        exit(-1);
    }
    spdlog::info("VectorMatrixUpdate n_update_user {}, n_update_item {}, update type {}, n_update_sample_vecs {}",
                 n_update_user, n_update_item, update_type, n_sample_vecs);

    RetrievalResult config;
    TimeRecord update_record;
    if (updateID != 0) {
        if (update_operator == "insert") {
            if (update_type == "user") {
                index->InsertUser(update_data_vm);
            } else if (update_type == "data_item") {
                index->InsertItem(update_data_vm);
            } else {
                spdlog::error("no such update type");
            }
        } else if (update_operator == "delete") {
            std::vector<int> delete_vecsID_l(n_sample_vecs);
            std::iota(delete_vecsID_l.begin(), delete_vecsID_l.end(), 0);
            if (update_type == "user") {
                index->DeleteUser(delete_vecsID_l);
            } else if (update_type == "data_item") {
                index->DeleteItem(delete_vecsID_l);
            } else {
                spdlog::error("no such update type");
            }

        } else {
            spdlog::error("no such update operator");
            exit(-1);
        }

    }
    const double update_time = update_record.get_elapsed_time_second();
    spdlog::info("finish update");
    vector<SingleQueryPerformance> query_performance_l(query_item.n_vector_);
    vector<vector<UserRankElement>> result_rk = index->Retrieval(query_item, topk, query_item.n_vector_,
                                                                 query_performance_l);
    index->FinishCompute();
    char performance_info[256];
    std::sprintf(performance_info, "UpdateID: %d", updateID);
    string performance_str = index->PerformanceStatistics(performance_info, topk);
    config.AddRetrievalInfo(performance_str);
    std::sprintf(performance_info, "UpdateID: %d, update time %.3fs", updateID, update_time);
    config.AddRetrievalInfo(performance_info);
    config.AddUpdateInfo(updateID, index->n_data_item_, index->n_user_);

    spdlog::info("finish retrieval, updateID {}, topk {}, update type {}, update operator {} ",
                 updateID, topk, update_type, update_operator);
    spdlog::info("{}", performance_str);

    char update_parameter_name[512];
    std::sprintf(update_parameter_name, "%s-updateID_%d-%s-%s", parameter_name, updateID, update_type.c_str(),
                 update_operator.c_str());
    WriteRankResult(result_rk, dataset_name, method_name.c_str(), update_parameter_name);
    WriteQueryPerformance(query_performance_l, dataset_name, method_name.c_str(), update_parameter_name);

    config.AddExecuteQuery(query_item.n_vector_);
    config.AddMemoryInfo(index->IndexSizeByte());
    spdlog::info("build index time: total {}s", build_index_time);
    config.AddBuildIndexTime(build_index_time);
    char modified_method_name[256];
    sprintf(modified_method_name, "retrieval-%s-updateID_%d-%s-%s", method_name.c_str(), updateID, update_type.c_str(),
            update_operator.c_str());
    config.WritePerformance(dataset_name, modified_method_name, parameter_name);
    return 0;
}