//
// Created by BianZheng on 2022/11/12.
//

#ifndef REVERSE_K_RANKS_ATTRIBUTIONFILEIO_HPP
#define REVERSE_K_RANKS_ATTRIBUTIONFILEIO_HPP

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

    void
    AttributionWriteRankResult(const std::vector<int> &result,
                               const int &topk, const char *dataset_name, const char *other_name) {
        int n_query_item = (int) result.size();

        char resPath[256];
        if (strcmp(other_name, "") == 0) {
            std::sprintf(resPath, "../../result/attribution/RMIPSCandidate/%s-top%d-n_cand.txt", dataset_name, topk);
        } else {
            std::sprintf(resPath, "../../result/attribution/RMIPSCandidate/%s-top%d-%s-n_cand.txt", dataset_name, topk,
                         other_name);
        }
        std::ofstream file(resPath);
        if (!file) {
            spdlog::error("error in write result");
        }

        for (int i = 0; i < n_query_item; i++) {
            file << result[i] << std::endl;
        }
        file.close();
    }

}
#endif //REVERSE_K_RANKS_ATTRIBUTIONFILEIO_HPP
