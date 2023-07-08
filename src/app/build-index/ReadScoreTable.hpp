//
// Created by BianZheng on 2022/4/12.
//

#ifndef REVERSE_K_RANKS_READSCORETABLE_HPP
#define REVERSE_K_RANKS_READSCORETABLE_HPP

#include "alg/TopkMaxHeap.hpp"
#include "struct/DistancePair.hpp"
#include "struct/UserRankElement.hpp"
#include "util/TimeMemory.hpp"

#include <memory>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class ReadScoreTable {
        int n_data_item_, n_user_;
        const char *index_path_;

    public:
        std::ifstream index_stream_;

        inline ReadScoreTable() {}

        inline ReadScoreTable(const int &n_user, const int &n_data_item, const char *index_path) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->index_path_ = index_path;
        }

        void ReadPreprocess() {
            index_stream_ = std::ifstream(this->index_path_, std::ios::binary | std::ios::in);
            if (!index_stream_) {
                spdlog::error("error in writing index");
            }
        }

        inline void ReadDisk(const int &userID, std::vector<float> &distance_l) {
            assert(distance_l.size() == n_data_item_);
            int64_t offset = (int64_t) userID * n_data_item_;
            offset *= sizeof(float);
            int64_t read_count_offset = n_data_item_ * sizeof(float);
            index_stream_.seekg(offset, std::ios::beg);
            index_stream_.read((char *) distance_l.data(), read_count_offset);
        }

        void FinishRead() {
            index_stream_.close();
        }
    };
}
#endif //REVERSE_K_RANKS_READSCORETABLE_HPP
