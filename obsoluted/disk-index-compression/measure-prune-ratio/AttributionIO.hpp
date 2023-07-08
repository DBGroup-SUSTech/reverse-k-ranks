//
// Created by BianZheng on 2022/6/23.
//

#ifndef REVERSE_KRANKS_ATTRIBUTIONIO_HPP
#define REVERSE_KRANKS_ATTRIBUTIONIO_HPP
namespace ReverseMIPS {
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


            if (strcmp(other_name, "") == 0) {
                std::sprintf(resPath, "../../result/plot_performance/%s-%s-config.csv",
                             dataset_name, method_name);
            } else {
                std::sprintf(resPath, "../../result/plot_performance/%s-%s-%s-config.csv",
                             dataset_name, method_name, other_name);
            }
            file.open(resPath);
            if (!file) {
                spdlog::error("error in write result");
            }
            int n_topk = (int) performance_l.size();
            file << "topk,retrieval_time,million_second_per_query" << std::endl;
            for (int i = n_topk - 1; i >= 0; i--) {
                file << performance_l[i].topk_ << "," << performance_l[i].retrieval_time_ << ","
                     << performance_l[i].ms_per_query_ << std::endl;
            }

            file.close();
        }
    };
}
#endif //REVERSE_KRANKS_ATTRIBUTIONIO_HPP
