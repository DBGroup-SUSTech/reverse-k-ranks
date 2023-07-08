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
#include <filesystem>
#include <iostream>

namespace ReverseMIPS {

    bool TestConsistency(const std::vector<int> &sample_itemID_l,
                         const std::vector<int> &sort_kth_rank_l,
                         const std::vector<int> &sort_sampleID_l,
                         const std::vector<int> &accu_n_user_rank_l,
                         const int &n_data_item,
                         const int &n_sample_item, const int &sample_topk) {

        assert(sample_itemID_l.size() == n_sample_item);
        assert(sort_kth_rank_l.size() == n_sample_item);
        assert(sort_sampleID_l.size() == n_sample_item);
        assert(accu_n_user_rank_l.size() == (size_t) n_sample_item * (n_data_item + 1));

        for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
            const int sort_sampleID = sort_sampleID_l[sampleID];
            assert(0 <= sort_sampleID && sort_sampleID <= n_sample_item - 1);

            const int *user_rank_ptr = accu_n_user_rank_l.data() + (size_t) sort_sampleID * (n_data_item + 1);
            const int *rank_ptr = std::lower_bound(user_rank_ptr, user_rank_ptr + (n_data_item + 1),
                                                   sample_topk,
                                                   [](const int &arr_n_rank, const int &topk) {
                                                       return arr_n_rank < topk;
                                                   });
            const int64_t kth_rank = rank_ptr - user_rank_ptr;
            assert(kth_rank == sort_kth_rank_l[sampleID]);
            assert(0 <= kth_rank && kth_rank <= n_data_item + 1);
        }
        return true;
    }

    void DeleteIfExist(const char *file_name) {
        if (std::filesystem::remove_all(file_name)) {
            std::cout << "file " << file_name << " deleted.\n";
        }
    }

    void WriteDistributionBelowTopk(const std::vector<int> &sample_itemID_l,
                                    const std::vector<int> &sort_kth_rank_l,
                                    const std::vector<int> &sort_sampleID_l,
                                    const std::vector<int> &accu_n_user_rank_l,
                                    const int64_t &n_data_item,
                                    const int64_t &n_sample_item, const int64_t &sample_topk,
                                    const char *dataset_name, const char *index_dir) {
        assert(sample_itemID_l.size() == n_sample_item);
        assert(sort_kth_rank_l.size() == n_sample_item);
        assert(sort_sampleID_l.size() == n_sample_item);
        assert(accu_n_user_rank_l.size() == n_sample_item * (n_data_item + 1));
        assert(TestConsistency(sample_itemID_l, sort_kth_rank_l,
                               sort_sampleID_l, accu_n_user_rank_l,
                               n_data_item, n_sample_item, sample_topk));

        char file_path[256];
        sprintf(file_path, "%s/query_distribution/%s-n_sample_item_%ld-sample_topk_%ld",
                index_dir, dataset_name, n_sample_item, sample_topk);
        DeleteIfExist(file_path);
        std::filesystem::create_directory(file_path);

        {
            char resPath[512];
            std::sprintf(resPath, "%s/sample_itemID_l.txt", file_path);

            std::ofstream out_stream = std::ofstream(resPath, std::ios::out);
            if (!out_stream) {
                spdlog::error("error in write result");
                exit(-1);
            }

            std::vector<int> sort_sample_itemID_l(n_sample_item);
            for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
                const int itemID = sample_itemID_l[sort_sampleID_l[sampleID]];
                sort_sample_itemID_l[sampleID] = itemID;
            }

            for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
                out_stream << sort_sample_itemID_l[sampleID] << std::endl;
            }

            out_stream.close();
        }

        {
            char resPath[512];
            std::sprintf(resPath, "%s/sort_kth_rank_l.index", file_path);

            std::ofstream out_stream = std::ofstream(resPath, std::ios::binary | std::ios::out);
            if (!out_stream) {
                spdlog::error("error in write result");
                exit(-1);
            }

            out_stream.write((char *) sort_kth_rank_l.data(),
                             (std::streamsize) ((size_t) sizeof(int) * n_sample_item));

            out_stream.close();
        }

        {
            char resPath[512];
            std::sprintf(resPath, "%s/accu_n_user_rank_l.index", file_path);

            std::ofstream out_stream = std::ofstream(resPath, std::ios::binary | std::ios::out);
            if (!out_stream) {
                spdlog::error("error in write result");
                exit(-1);
            }

            for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
                const int sorted_sampleID = sort_sampleID_l[sampleID];

                const int *user_rank_ptr = accu_n_user_rank_l.data() + (int64_t) sorted_sampleID * (n_data_item + 1);

                out_stream.write((char *) user_rank_ptr,
                                 (std::streamsize) ((size_t) sizeof(int) * (n_data_item + 1)));
            }

            out_stream.close();
        }

    }

}
#endif //REVERSE_KRANKS_FILEIO_HPP
