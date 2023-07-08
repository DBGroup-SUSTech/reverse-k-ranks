#pragma once

#include "struct/UserRankElement.hpp"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>
#include <ostream>
#include <string>
#include <map>
#include <iomanip>

namespace ReverseMIPS {

    class RetrievalResult {
        std::vector<std::string> config_l;
    public:

        std::string GetConfig(const int ID) {
            return config_l[ID];
        }

        void AddMemoryInfo(const uint64_t estimated_size_byte) {
            char str[256];
            sprintf(str, "Estimated memory %.3fGB, Peak memory %.3fGB, Current memory %.3fGB",
                    estimated_size_byte * 1.0 / 1024 / 1024 / 1024,
                    get_peak_RSS() * 1.0 / 1024 / 1024 / 1024,
                    get_current_RSS() * 1.0 / 1024 / 1024 / 1024);
            this->config_l.emplace_back(str);
        }

        void AddRetrievalInfo(const std::string &str) {
            this->config_l.emplace_back(str);
        }

        void AddInfo(const std::string &info) {
            this->config_l.emplace_back(info);
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

        void AddUpdateInfo(const int &updateID, const int &n_data_item, const int &n_user) {
            char buff[128];
            sprintf(buff, "updateID %d, n_data_item %d, n_user %d",
                    updateID, n_data_item, n_user);
            std::string str(buff);
            this->config_l.emplace_back(str);
        }

        void WritePerformance(const char *dataset_name, const char *method_name,
                              const char *other_name) {
            char resPath[256];
            std::sprintf(resPath, "../result/vis_performance/%s-%s-%s-performance.txt",
                         dataset_name, method_name, other_name);
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
    WriteRankResult(const std::vector<std::vector<UserRankElement>> &result,
                    const char *dataset_name, const char *method_name, const char *other_name) {
        int n_query_item = (int) result.size();

        char resPath[256];
        std::sprintf(resPath, "../result/rank/%s-%s-%s-userID.csv", dataset_name, method_name, other_name);
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

        std::sprintf(resPath, "../result/rank/%s-%s-%s-rank.csv", dataset_name, method_name, other_name);
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

        std::sprintf(resPath, "../result/rank/%s-%s-%s-IP.csv", dataset_name, method_name, other_name);
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

    class SingleQueryPerformance {
    public:
        int queryID_, n_prune_user_, n_result_user_, n_refine_user_;
        size_t io_cost_, ip_cost_;
        double total_time_, memory_index_time_, io_time_;

        inline SingleQueryPerformance() = default;

        inline SingleQueryPerformance(const int &queryID,
                                      const int &n_prune_user, const int &n_result_user, const int &n_refine_user,
                                      const size_t &ip_cost, const size_t &io_cost,
                                      const double &total_time, const double &memory_index_time,
                                      const double &io_time) {
            this->queryID_ = queryID;
            this->n_prune_user_ = n_prune_user;
            this->n_result_user_ = n_result_user;
            this->n_refine_user_ = n_refine_user;
            this->ip_cost_ = ip_cost;
            this->io_cost_ = io_cost;
            this->total_time_ = total_time;
            this->memory_index_time_ = memory_index_time;
            this->io_time_ = io_time;
        }
    };

    void WriteQueryPerformance(const std::vector<SingleQueryPerformance> &query_performance_l,
                               const char *dataset_name, const char *method_name, const char *other_name) {

        int n_query_item = (int) query_performance_l.size();

        char resPath[256];
        std::sprintf(resPath, "../result/single_query_performance/%s-%s-%s-single-query-performance.csv",
                     dataset_name,
                     method_name, other_name);
        std::ofstream file(resPath);
        if (!file) {
            spdlog::error("error in write result");
        }

        char buff[512];
        sprintf(buff,
                "queryID,n_prune_user,n_result_user,n_refine_user,ip_cost,io_cost,total_time,memory_index_time,io_time");
        std::string str(buff);
        file << str << std::endl;
        for (int i = 0; i < n_query_item; i++) {
            const SingleQueryPerformance &sqp = query_performance_l[i];
            sprintf(buff, "%10d,%10d,%10d,%10d,%10ld,%10ld,%10.2f,%10.2f,%10.2f",
                    sqp.queryID_,
                    sqp.n_prune_user_, sqp.n_result_user_, sqp.n_refine_user_,
                    sqp.ip_cost_, sqp.io_cost_,
                    sqp.total_time_, sqp.memory_index_time_, sqp.io_time_);
            str = std::string(buff);

            file << str << std::endl;
        }
        file.close();
    }

}