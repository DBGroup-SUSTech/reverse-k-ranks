//
// Created by BianZheng on 2022/7/15.
//

#include "BatchRun/BatchMeasureRetrievalTopTIP.hpp"
#include "BatchRun/BatchBuildIndexRSTopTIP.hpp"
#include "BatchRun/BatchRetrievalTopT.hpp"
#include "BatchRun/RankSampleMeasurePruneRatio.hpp"
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
    spdlog::info("RankSampleTopT dataset_name {}, basic_dir {}", dataset_name, basic_dir);

    {
        TimeRecord record;
        record.reset();
        BuildIndex(basic_dir, dataset_name);

        const double build_index_time = record.get_elapsed_time_second();
        spdlog::info("finish preprocess and save the index, build index time {}s", build_index_time);
    }

    {
        RankSampleMeasurePruneRatio::MeasurePruneRatio(dataset_name, basic_dir, 128);
        RankSampleMeasurePruneRatio::MeasurePruneRatio(dataset_name, basic_dir, 512);
    }

    {
        char index_basic_dir[128];
//        sprintf(index_basic_dir, "../index/%s_constructed_index",
//                dataset_name);
        sprintf(index_basic_dir, "../index");

        //search on TopTIP
        const int index_size_gb = 256;
        const int memory_n_sample = 512;
        const int n_eval_query = 100;

        char disk_path[256];
        sprintf(disk_path, "%s/%s_TopTIP%d.index", index_basic_dir, dataset_name, index_size_gb);
        char memory_path[256];
        sprintf(memory_path, "%s/%s_RankSearch%d_TopT.index", index_basic_dir, dataset_name, memory_n_sample);
        BatchMeasureRetrievalTopTIP::MeasureTopTIP(disk_path, memory_path,
                                                   memory_n_sample, index_size_gb,
                                                   basic_dir, dataset_name,
                                                   n_eval_query);
    }

    {
        char index_basic_dir[128];
        sprintf(index_basic_dir, "../index");
//        sprintf(index_basic_dir, "../index/%s_constructed_index",
//                dataset_name);

        //search on TopTIP
        const int index_size_gb = 256;
        const int n_sample = 512;
        char disk_path[256];
        sprintf(disk_path, "%s/%s_TopTIP%d.index",
                index_basic_dir, dataset_name, index_size_gb);
        char memory_path[256];
        sprintf(memory_path, "%s/%s_RankSearch%d_TopT.index",
                index_basic_dir, dataset_name, n_sample);
        RunRetrieval(disk_path, memory_path,
                     n_sample, index_size_gb,
                     basic_dir, dataset_name);
    }

    return 0;
}