//
// Created by BianZheng on 2022/6/20.
//

#ifndef REVERSE_KRANKS_CANDIDATESIO_HPP
#define REVERSE_KRANKS_CANDIDATESIO_HPP

#include <filesystem>

namespace ReverseMIPS {
    void
    WriteRankResult(const std::vector<std::vector<UserRankElement>> &result, const char *dataset_name,
                    const char *method_name, const char *other_name) {
        int n_query_item = (int) result.size();
        int topk = (int) result[0].size();

        char resPath[256];
        std::sprintf(resPath, "../../result/rank/%s-%s-top%d-%s-userID.csv", dataset_name, method_name, topk,
                     other_name);
        std::ofstream file(resPath);
        if (!file) {
            spdlog::error("error in write result");
        }

        for (int i = 0; i < n_query_item; i++) {
            for (int j = 0; j < topk - 1; j++) {
                file << result[i][j].userID_ << ",";
            }
            file << result[i][topk - 1].userID_ << std::endl;
        }
        file.close();

        std::sprintf(resPath, "../../result/rank/%s-%s-top%d-%s-rank.csv", dataset_name, method_name, topk, other_name);
        file.open(resPath);
        if (!file) {
            spdlog::error("error in write result");
        }

        for (int i = 0; i < n_query_item; i++) {
            for (int j = 0; j < topk - 1; j++) {
                file << result[i][j].rank_ << ",";
            }
            file << result[i][topk - 1].rank_ << std::endl;
        }
        file.close();

        std::sprintf(resPath, "../../result/rank/%s-%s-top%d-%s-IP.csv", dataset_name, method_name, topk, other_name);
        file.open(resPath);
        if (!file) {
            spdlog::error("error in write result");
        }

        for (int i = 0; i < n_query_item; i++) {
            for (int j = 0; j < topk - 1; j++) {
                file << result[i][j].queryIP_ << ",";
            }
            file << result[i][topk - 1].queryIP_ << std::endl;
        }
        file.close();
    }

    class ItemCandidates {
    public:
        int userID_;
        std::vector<int> itemID_l_;
        int rank_lb_, rank_ub_;

        inline ItemCandidates() = default;

        inline ItemCandidates(const std::vector<int> &itemID_l, const int &userID,
                              const int &rank_lb, const int &rank_ub) {
            this->itemID_l_ = itemID_l;
            this->userID_ = userID;
            this->rank_lb_ = rank_lb;
            this->rank_ub_ = rank_ub;
        }

    };

    void CreateCandidateFile(const char *dataset_name) {
        char resPath[256];
        std::sprintf(resPath, "../../result/attribution/Candidate-%s/", dataset_name);

        if (std::filesystem::create_directories(resPath)) {
            spdlog::info("mkdir {}", resPath);
        } else {
            spdlog::info("mkdir exist {}, delete", resPath);
            std::filesystem::remove_all(resPath);
            std::filesystem::create_directories(resPath);
        }

    }

    void
    WriteCandidateResult(const std::vector<std::vector<ItemCandidates>> &result, const int &topk,
                         const char *dataset_name) {
        int n_query_item = (int) result.size();

        char resPath[256];
        for (int qID = 0; qID < n_query_item; qID++) {
            std::sprintf(resPath, "../../result/attribution/Candidate-%s/%s-top%d-qID-%d.txt",
                         dataset_name, dataset_name, topk, qID);
            std::ofstream file(resPath);
            if (!file) {
                spdlog::error("error in write result");
            }

            for (const ItemCandidates &ucandID: result[qID]) {
                file << ucandID.userID_ << ":" << ucandID.rank_lb_ << ":" << ucandID.rank_ub_ << std::endl;
                int item_cand_size = int(ucandID.itemID_l_.size());
                if (item_cand_size == 0) {
                    file << "size0" << std::endl;
                } else {
                    for (int item_candID = 0; item_candID < item_cand_size - 1; item_candID++) {
                        file << ucandID.itemID_l_[item_candID] << ",";
                    }
                    file << ucandID.itemID_l_[ucandID.itemID_l_.size() - 1] << std::endl;
                }

            }
            file.close();
        }

    }
}
#endif //REVERSE_KRANKS_CANDIDATESIO_HPP
