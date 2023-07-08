//
// Created by BianZheng on 2022/7/25.
//

#include "BatchRun/BatchMeasureRetrievalMergeRankByInterval.hpp"
#include "BatchRun/BatchBuildIndexMergeRankByInterval.hpp"
#include "BatchRun/ScoreSampleMeasurePruneRatio.hpp"
#include "BatchRun/BatchRetrievalMergeRankByInterval.hpp"
#include "util/TimeMemory.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <string>

class Parameter {
public:
    std::string basic_dir, dataset_name;
};

void LoadOptions(int argc, char **argv, Parameter &para) {
    namespace po = boost::program_options;

    po::options_description opts("Allowed options");
    opts.add_options()
            ("help,h", "help info")
            ("basic_dir,bd",
             po::value<std::string>(&para.basic_dir)->default_value("/home/bianzheng/Dataset/ReverseMIPS"),
             "basic directory")
            ("dataset_name, ds", po::value<std::string>(&para.dataset_name)->default_value("fake-normal"),
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
    const char *basic_dir = para.basic_dir.c_str();
    spdlog::info("MergeRankByInterval dataset_name {}, basic_dir {}", dataset_name, basic_dir);

//    {
//        TimeRecord record;
//        record.reset();
//
//        const int index_size_gb = 256;
//        BuildIndex(basic_dir, dataset_name, index_size_gb);
//
//        double build_index_time = record.get_elapsed_time_second();
//        spdlog::info("finish preprocess and save the index, build index time {}s", build_index_time);
//    }

//    {
//        ScoreSampleMeasurePruneRatio::MeasurePruneRatio(dataset_name, basic_dir, 512);
//    }

    {
        char index_basic_dir[128];
        sprintf(index_basic_dir, "../index/%s_constructed_index",
                dataset_name);

        //measure MergeRankByInterval
        const int memory_n_sample = 512;
        const uint64_t index_size_gb = 256;
        const int n_eval_query = 1000;

        char disk_index_path[256];
        sprintf(disk_index_path, "%s/%s_MergeRankByInterval%ld.index",
                index_basic_dir, dataset_name, index_size_gb);
        char disk_memory_index_path[256];
        sprintf(disk_memory_index_path, "%s/%s_MergeRankByInterval%ld_memory.index",
                index_basic_dir, dataset_name, index_size_gb);
        char memory_index_path[256];
        sprintf(memory_index_path, "%s/%s_ScoreSearch%d.index",
                index_basic_dir, dataset_name, memory_n_sample);

        BatchMeasureMergeRankByInterval::MeasureMergeRankByInterval(
                disk_index_path, disk_memory_index_path, memory_index_path,
                memory_n_sample, index_size_gb,
                basic_dir, dataset_name, "MeasureScoreSampleMergeRankByInterval",
                n_eval_query);

    }

//    {
//        char index_basic_dir[128];
////        sprintf(index_basic_dir, "../index");
//        sprintf(index_basic_dir, "../index/%s_constructed_index",
//                dataset_name);
//
//        //search on MergeRankByInterval
//        const int memory_n_sample = 512;
//        const uint64_t index_size_gb = 256;
//
//        char bitmap256_path[256];
//        sprintf(bitmap256_path, "%s/%s_MergeRankByInterval%ld.index",
//                index_basic_dir, dataset_name, index_size_gb);
//        char bitmap256_memory_path[256];
//        sprintf(bitmap256_memory_path, "%s/%s_MergeRankByInterval%ld_memory.index",
//                index_basic_dir, dataset_name, index_size_gb);
//        char memory_path[256];
//        sprintf(memory_path, "%s/%s_ScoreSearch%d.index",
//                index_basic_dir, dataset_name, memory_n_sample);
//
//        RunRetrieval(bitmap256_path, bitmap256_memory_path, memory_path,
//                     memory_n_sample, index_size_gb,
//                     basic_dir, dataset_name, "SSMergeRankByIntervalBatchRun");
//    }

//    const char *toptID128_path = "../index/Amazon_TopTID128.index";
//    const char *toptID256_path = "../index/Amazon_TopTID256.index";
//    const char *toptIP128_path = "../index/Amazon_TopTIP128.index";
//    const char *toptIP256_path = "../index/Amazon_TopTIP256.index";
//    const char *ss128_path = "../index/Amazon_ScoreSearch128.index";
//    const char *ss1024_path = "../index/Amazon_ScoreSearch1024.index";

    return 0;
}