//
// Created by BianZheng on 2022/7/27.
//

#ifndef REVERSE_KRANKS_FILEIO_HPP
#define REVERSE_KRANKS_FILEIO_HPP

#include "struct/VectorMatrix.hpp"
#include "struct/UserRankElement.hpp"

#include <spdlog/spdlog.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include <random>

namespace ReverseMIPS {

    void WriteSortedUserRank(const std::vector<int> &sample_itemID_l,
                             const std::vector<int> &sorted_user_rank_l,
                             const size_t &n_sample_item, const size_t &n_user,
                             const char *dataset_name) {
        assert(sample_itemID_l.size() == n_sample_item);
        assert(sorted_user_rank_l.size() == n_sample_item * n_user);

        {
            char resPath[256];
            std::sprintf(resPath,
                         "../../index/query_distribution/%s-sample-itemID-n_sample_query_%ld.index",
                         dataset_name, n_sample_item);

            std::ofstream out_stream = std::ofstream(resPath, std::ios::binary | std::ios::out);
            if (!out_stream) {
                spdlog::error("error in write result");
                exit(-1);
            }

            out_stream.write((char *) sample_itemID_l.data(),
                             (std::streamsize) (sizeof(int) * n_sample_item * n_user));

            out_stream.close();
        }

        {
            char resPath[256];
            std::sprintf(resPath,
                         "../../index/query_distribution/%s-sorted-user-rank-n_sample_query_%ld.index",
                         dataset_name, n_sample_item);

            std::ofstream out_stream = std::ofstream(resPath, std::ios::binary | std::ios::out);
            if (!out_stream) {
                spdlog::error("error in write result");
                exit(-1);
            }

            out_stream.write((char *) sorted_user_rank_l.data(),
                             (std::streamsize) (sizeof(int) * n_sample_item * n_user));

            out_stream.close();
        }

    }

    void ReadSortedUserRank(const size_t &n_sample_item, const size_t &n_user,
                            const char *dataset_name,
                            std::vector<int> &sorted_user_rank_l) {
        assert(sorted_user_rank_l.size() == n_sample_item * n_user);

        char resPath[256];
        std::sprintf(resPath,
                     "../../index/query_distribution/%s-sorted-user-rank-n_sample_query_%ld.index",
                     dataset_name, n_sample_item);

        std::ifstream in_stream = std::ifstream(resPath, std::ios::binary);
        if (!in_stream.is_open()) {
            spdlog::error("Open file error");
            exit(-1);
        }

        in_stream.read((char *) sorted_user_rank_l.data(),
                       (std::streamsize) (sizeof(int) * n_sample_item * n_user));

        in_stream.close();
    }

}
#endif //REVERSE_KRANKS_FILEIO_HPP
