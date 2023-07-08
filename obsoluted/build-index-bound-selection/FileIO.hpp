//
// Created by BianZheng on 2022/10/6.
//

#ifndef REVERSE_KRANKS_FILEIO_HPP
#define REVERSE_KRANKS_FILEIO_HPP

#include <algorithm>
#include <fstream>
#include <iostream>
#include "struct/UserRankElement.hpp"
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

        void AddBuildIndexInfo(const std::string &str) {
            this->config_l.emplace_back(str);
        }

        void AddRetrievalInfo(const std::string &str) {
            this->config_l.emplace_back(str);
        }

        void AddBuildIndexTime(const double &build_index_time) {
            char buff[128];
            sprintf(buff, "build index time %.3fs", build_index_time);
            std::string str(buff);
            this->config_l.emplace_back(str);
        }

        void AddExecuteQuery(const int &n_execute_query) {
            char buff[128];
            sprintf(buff, "number of query item %d", n_execute_query);
            std::string str(buff);
            this->config_l.emplace_back(str);
        }

        void AddQueryInfo(const int n_eval_query) {
            char buff[128];
            sprintf(buff, "number of evaluate query %d", n_eval_query);
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

    void
    WriteRankResult2(const std::vector<std::vector<UserRankElement>> &result,
                    const int &topk, const char *dataset_name, const char *method_name, const char *other_name) {
        int n_query_item = (int) result.size();

        char resPath[256];
        if (strcmp(other_name, "") == 0) {
            std::sprintf(resPath, "../../result/rank/%s-%s-top%d-userID.csv", dataset_name, method_name, topk);
        } else {
            std::sprintf(resPath, "../../result/rank/%s-%s-top%d-%s-userID.csv", dataset_name, method_name, topk,
                         other_name);
        }
        std::ofstream file(resPath);
        if (!file) {
            spdlog::error("error in write result");
        }

        for (int i = 0; i < n_query_item; i++) {
            const unsigned int result_size = result[i].size();
            if (result_size != 0) {
                for (int j = 0; j < result_size - 1; j++) {
                    file << result[i][j].userID_ << ",";
                }
                file << result[i][result_size - 1].userID_ << std::endl;
            } else {
                file << std::endl;
            }

        }
        file.close();

        if (strcmp(other_name, "") == 0) {
            std::sprintf(resPath, "../../result/rank/%s-%s-top%d-rank.csv", dataset_name, method_name, topk);
        } else {
            std::sprintf(resPath, "../../result/rank/%s-%s-top%d-%s-rank.csv", dataset_name, method_name, topk,
                         other_name);
        }
        file.open(resPath);
        if (!file) {
            spdlog::error("error in write result");
        }

        for (int i = 0; i < n_query_item; i++) {
            const unsigned int result_size = result[i].size();
            if (result_size != 0) {
                for (int j = 0; j < result_size - 1; j++) {
                    file << result[i][j].rank_ << ",";
                }
                file << result[i][result_size - 1].rank_ << std::endl;
            } else {
                file << std::endl;
            }

        }
        file.close();

        if (strcmp(other_name, "") == 0) {
            std::sprintf(resPath, "../../result/rank/%s-%s-top%d-IP.csv", dataset_name, method_name, topk);
        } else {
            std::sprintf(resPath, "../../result/rank/%s-%s-top%d-%s-IP.csv", dataset_name, method_name, topk,
                         other_name);
        }
        file.open(resPath);
        if (!file) {
            spdlog::error("error in write result");
        }

        for (int i = 0; i < n_query_item; i++) {
            const unsigned int result_size = result[i].size();
            if (result_size != 0) {
                for (int j = 0; j < result_size - 1; j++) {
                    file << result[i][j].queryIP_ << ",";
                }
                file << result[i][result_size - 1].queryIP_ << std::endl;
            } else {
                file << std::endl;
            }

        }
        file.close();
    }

}
#endif //REVERSE_KRANKS_FILEIO_HPP
