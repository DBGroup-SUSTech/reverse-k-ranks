//
// Created by BianZheng on 2022/9/3.
//

#ifndef REVERSE_KRANKS_BUILDINDEXTIMEFILEIO_HPP
#define REVERSE_KRANKS_BUILDINDEXTIMEFILEIO_HPP

#include "struct/UserRankElement.hpp"
#include <spdlog/spdlog.h>

#include <cstring>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <ostream>
#include <string>
#include <map>
#include <iomanip>

namespace ReverseMIPS {

    class RetrievalResult2 {
        std::vector<std::string> config_l;
    public:

        std::string GetConfig(const int ID) {
            return config_l[ID];
        }

        void AddBuildIndexTime(const char *build_index_info, const double &build_index_time) {
            char buff[128];
            sprintf(buff, "Build Index %s %.3fs", build_index_info, build_index_time);
            std::string str(buff);
            this->config_l.emplace_back(str);
        }

        void WritePerformance(const char *dataset_name, const char *method_name,
                              const char *other_name) {
            char resPath[256];
            if (strcmp(other_name, "") == 0) {
                std::sprintf(resPath, "../../result/vis_performance/%s-%s-config.txt",
                             dataset_name, method_name);
            } else {
                std::sprintf(resPath, "../../result/vis_performance/%s-%s-%s-config.txt",
                             dataset_name, method_name, other_name);
            }
            std::ofstream file(resPath);
            if (!file) {
                spdlog::error("error in write result");
            }
            int config_size = (int) config_l.size();
            for (int i = config_size - 1; i >= 0; i--) {
                file << config_l[i] << std::endl;
            }
            file.close();
        }
    };

}
#endif //REVERSE_KRANKS_BUILDINDEXTIMEFILEIO_HPP
