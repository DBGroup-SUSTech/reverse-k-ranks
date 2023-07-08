//
// Created by BianZheng on 2022/7/29.
//

#ifndef REVERSE_KRANKS_FILEIO_HPP
#define REVERSE_KRANKS_FILEIO_HPP

#include <vector>
#include <string>
#include <spdlog/spdlog.h>
#include <fstream>

namespace ReverseMIPS::PrintRankBound {
    class PlotPerformanceAttribution {
    public:
        double retrieval_time_, ms_per_query_;
        int topk_;

        PlotPerformanceAttribution() = default;

        PlotPerformanceAttribution(const int &topk, const double &retrieval_time, const double &ms_per_query) {
            this->topk_ = topk;
            this->retrieval_time_ = retrieval_time;
            this->ms_per_query_ = ms_per_query;
        }
    };

    class RetrievalResultAttribution {
        std::vector<std::string> config_l;
        std::vector<PlotPerformanceAttribution> performance_l;
    public:

        std::string GetConfig(const int ID) {
            return config_l[ID];
        }

        void AddBuildIndexInfo(const std::string &str) {
            this->config_l.emplace_back(str);
        }

        void AddRetrievalInfo(const std::string &str, const int &topk, const double &retrieval_time,
                              const double &ms_per_query) {
            this->config_l.emplace_back(str);
            performance_l.emplace_back(topk, retrieval_time, ms_per_query);
        }

        void AddBuildIndexTime(const double &build_index_time) {
            char buff[128];
            sprintf(buff, "build index time %.3fs", build_index_time);
            std::string str(buff);
            this->config_l.emplace_back(str);
        }

        void WriteRankBound(const std::vector<std::pair<int, int>> &rank_bound_l,
                             const int &n_user, const int &topk,
                             const char *dataset_name, const char *method_name, const char *other_name) {
            assert(rank_bound_l.size() == n_user);
            char resPath[1024];
            std::sprintf(resPath, "../../result/attribution/PrintRankBound/%s-%s-top%d-%s-config.txt",
                         dataset_name, method_name, topk, other_name);
            std::ofstream file(resPath);
            if (!file) {
                spdlog::error("error in write result");
            }
            for (int userID = 0; userID < n_user; userID++) {
                file << rank_bound_l[userID].first << ", " << rank_bound_l[userID].second << std::endl;
            }
            file.close();
        }
    };

}
#endif //REVERSE_KRANKS_FILEIO_HPP
