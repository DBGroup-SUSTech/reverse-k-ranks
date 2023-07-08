//
// Created by BianZheng on 2022/11/5.
//

#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/VectorMatrix.hpp"
#include "util/NameTranslation.hpp"

#include "alg/RegressionPruning/BaseLinearRegression.hpp"
#include "alg/RegressionPruning/NormalLinearRegressionLP.hpp"
#include "alg/RegressionPruning/UniformLinearRegressionLP.hpp"

#include "alg/RankBoundRefinement/SampleSearch.hpp"


#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <string>

class Parameter {
public:
    std::string dataset_dir, dataset_name, index_dir, method_name;
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
            ("method_name, tm",
             po::value<std::string>(&para.method_name)->default_value("QueryRankSampleDirectIntLR"),
             "method name")

            ("n_sample, ns", po::value<int>(&para.n_sample)->default_value(-1),
             "number of sample of a rank bound")
            ("n_sample_query, nsq", po::value<int>(&para.n_sample_query)->default_value(150),
             "number of sample query")
            ("sample_topk, st", po::value<int>(&para.sample_topk)->default_value(60),
             "sample topk");

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

void BuildLocalRegressionIndex(const VectorMatrix &data_item, const VectorMatrix &user,
                               const int64_t &n_sample, const int &n_sample_query, const int &sample_topk,
                               const char *dataset_name, const char *basic_index_dir, const std::string &method_name,
                               double &build_index_time) {
    const int n_user = user.n_vector_;
    const int n_data_item = data_item.n_vector_;

    //rank search
    std::unique_ptr<BaseLinearRegression> lr_ins;
    SampleSearch rs_ins;

    const bool load_sample_score = true;
    const bool is_query_distribution = true;
    const string index_name = SampleSearchIndexName(method_name);

    rs_ins = SampleSearch(basic_index_dir, dataset_name, index_name.c_str(),
                          n_sample, load_sample_score, is_query_distribution,
                          n_sample_query, sample_topk);
    spdlog::info("finish load sample index");

    const string regression_method_name = RegressionMethodName(method_name);
    const string regression_index_name = RegressionIndexName(method_name);

    if (regression_method_name == "NormalLinearRegressionLP") {
        lr_ins = std::make_unique<NormalLinearRegressionLP>(n_data_item, n_user, regression_index_name);

    } else if (regression_method_name == "UniformLinearRegressionLP") {
        lr_ins = std::make_unique<UniformLinearRegressionLP>(n_data_item, n_user, regression_index_name);

    } else {
        spdlog::error("no such training method, program exit");
        exit(-1);
    }
    TimeRecord total_record;
    total_record.reset();
    lr_ins->StartPreprocess(rs_ins.known_rank_idx_l_.get(), (int) n_sample);
    spdlog::info("finish regression index preprocess");


    TimeRecord record;
    record.reset();
    const int batch_n_user = lr_ins->batch_n_user_;
    const int n_batch = n_user / batch_n_user + (n_user % batch_n_user == 0 ? 0 : 1);
    const int n_report = 5;
    const int report_every = n_batch / n_report == 0 ? 1 : n_batch / n_report;

    double batch_assign_cache_time = 0, batch_linear_program_time = 0, batch_calc_error_time = 0;

    for (int batchID = 0; batchID < n_batch; batchID++) {
        const int start_userID = batch_n_user * batchID;
        const int end_userID = std::min(batch_n_user * (batchID + 1), n_user);
        const int n_proc_user = end_userID - start_userID;
        std::vector<const float *> sampleIP_l_l(n_proc_user);

        for (int proc_userID = 0; proc_userID < n_proc_user; proc_userID++) {
            const int userID = proc_userID + start_userID;
            sampleIP_l_l[proc_userID] = rs_ins.SampleData(userID);
        }

        double assign_cache_time = 0, linear_program_time = 0, calc_error_time = 0;

        lr_ins->BatchLoopPreprocess(sampleIP_l_l,
                                    start_userID, n_proc_user,
                                    assign_cache_time, linear_program_time, calc_error_time);
        batch_assign_cache_time += assign_cache_time;
        batch_linear_program_time += linear_program_time;
        batch_calc_error_time += calc_error_time;

        if (batchID % report_every == 0) {
            std::cout << "preprocessed " << batchID / (0.01 * n_batch) << " %, "
                      << record.get_elapsed_time_second() << " s/iter, "
                      << "assign cache time " << batch_assign_cache_time << ", "
                      << "linear program time " << batch_linear_program_time << ", "
                      << "calc error time " << batch_calc_error_time << ", "
                      << "Mem: " << get_current_RSS() / 1000000 << " Mb \n";
            record.reset();
            batch_assign_cache_time = 0;
            batch_linear_program_time = 0;
            batch_calc_error_time = 0;
        }
    }

    lr_ins->FinishPreprocess();

    build_index_time = total_record.get_elapsed_time_second();
    lr_ins->SaveIndex(basic_index_dir, dataset_name, n_sample_query, sample_topk);

}

int main(int argc, char **argv) {
    Parameter para;
    LoadOptions(argc, argv, para);
    const char *dataset_name = para.dataset_name.c_str();
    const char *dataset_dir = para.dataset_dir.c_str();
    string index_dir = para.index_dir;
    const string method_name = para.method_name;
    const string program_name = "BuildRegressionIndex";
    spdlog::info("{} dataset_name {}, dataset_dir {}",
                 program_name, dataset_name, dataset_dir);
    spdlog::info("index_dir {}, method_name {}", index_dir, method_name);
    spdlog::info("n_sample {}, n_sample_query {}, sample_topk {}", para.n_sample, para.n_sample_query,
                 para.sample_topk);
    const string regression_method_name = RegressionMethodName(method_name);
    const string regression_index_name = RegressionIndexName(method_name);
    spdlog::info("regression_method_name {}, regression_index_name {}",
                 regression_method_name, regression_index_name);

    const std::int64_t n_sample = para.n_sample;

    int n_data_item, n_query_item, n_user, vec_dim;
    vector<VectorMatrix> data = readData(dataset_dir, dataset_name, n_data_item, n_query_item, n_user,
                                         vec_dim);
    VectorMatrix &user = data[0];
    VectorMatrix &data_item = data[1];
    VectorMatrix &query_item = data[2];
    spdlog::info("n_data_item {}, n_query_item {}, n_user {}, vec_dim {}", n_data_item, n_query_item, n_user, vec_dim);

    double build_index_time = 0;
    BuildLocalRegressionIndex(data_item, user,
                              n_sample, para.n_sample_query, para.sample_topk,
                              dataset_name, index_dir.c_str(), method_name, build_index_time);

    spdlog::info("finish preprocess and save the index");

    RetrievalResult config;

    spdlog::info("{} build index time: total {}s", program_name, build_index_time);

    char parameter_name[128];
    sprintf(parameter_name, "%s-n_sample_%d", method_name.c_str(), (int) n_sample);
    std::string n_sample_info = "n_sample: " + std::to_string(n_sample);
    config.AddInfo(n_sample_info);
    config.AddBuildIndexTime(build_index_time);
    config.WritePerformance(dataset_name, program_name.c_str(), parameter_name);
    return 0;
}