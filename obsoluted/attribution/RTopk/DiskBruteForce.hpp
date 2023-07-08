//
// Created by BianZheng on 2022/3/31.
//

#ifndef REVERSE_K_RANKS_DISKBRUTEFORCE_HPP
#define REVERSE_K_RANKS_DISKBRUTEFORCE_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/MethodBase.hpp"
#include <fstream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <map>
#include <spdlog/spdlog.h>

namespace ReverseMIPS::DiskBruteForce {
    class RetrievalResult : public RetrievalResultBase {
    public:
        //unit: second
        //double total_time, read_disk_time, inner_product_time, binary_search_time, second_per_query;
        //int topk;

        inline RetrievalResult() {
        }

        void AddPreprocess(double build_index_time) {
            char buff[1024];
            sprintf(buff, "build index time %.3f", build_index_time);
            std::string str(buff);
            this->config_l.emplace_back(str);
        }

        std::string AddResultConfig(int topk, double total_time, double read_disk_time, double inner_product_time,
                                    double compare_ip_time, double second_per_query) {
            char buff[1024];

            sprintf(buff,
                    "top%d retrieval time:\n\ttotal %.3fs, read disk %.3fs\n\tinner product %.3fs, compare inner product time %.3fs, million second per query %.3fms",
                    topk, total_time, read_disk_time, inner_product_time,
                    compare_ip_time, second_per_query);
            std::string str(buff);
            this->config_l.emplace_back(str);
            return str;
        }

    };

    class Index : public BaseIndex {
        void ResetTimer() {
            read_disk_time_ = 0;
            inner_product_time_ = 0;
            compare_ip_time_ = 0;
        }

    public:
        VectorMatrix user_;
        int vec_dim_, n_data_item_;
        double read_disk_time_, inner_product_time_, compare_ip_time_;
        const char *index_path_;
        TimeRecord read_disk_record_, inner_product_record_, compare_ip_record_;

        Index() {}

        Index(const char *index_path, const int n_data_item, VectorMatrix &user) {
            this->index_path_ = index_path;
            this->vec_dim_ = user.vec_dim_;
            this->user_ = std::move(user);
            this->n_data_item_ = n_data_item;
        }

        std::vector<std::vector<UserRankElement>> Retrieval(VectorMatrix &query_item, const int &topk) override {
            TimeRecord query_record;
            ResetTimer();
            std::ifstream index_stream_ = std::ifstream(this->index_path_, std::ios::binary | std::ios::in);
            if (!index_stream_) {
                spdlog::error("error in writing index");
            }

            if (topk > user_.n_vector_) {
                spdlog::error("top-k is too large, program exit");
                exit(-1);
            }

            int n_query_item = query_item.n_vector_;
            int n_user = user_.n_vector_;

            std::vector<std::vector<UserRankElement>> query_heap_l(n_query_item);

            std::vector<double> queryIP_l(n_user);
            std::vector<double> cmpIP_l(n_user);
            std::vector<int> rank_l(n_user);

            query_record.reset();
            for (int qID = 0; qID < n_query_item; qID++) {
                //calculate distance
                double *query_item_vec = query_item.getVector(qID);
                inner_product_record_.reset();
                for (int userID = 0; userID < n_user; userID++) {
                    double *user_vec = user_.getVector(userID);
                    double queryIP = InnerProduct(query_item_vec, user_vec, vec_dim_);
                    queryIP_l[userID] = queryIP;
                }
                this->inner_product_time_ += inner_product_record_.get_elapsed_time_second();

                read_disk_record_.reset();
                for (int userID = 0; userID < n_user; userID++) {
                    size_t disk_offset = (userID * n_data_item_ + topk) * sizeof(double);
                    index_stream_.seekg(disk_offset, std::ios::beg);
                    double cmp_ip;
                    index_stream_.read((char *) &cmp_ip, sizeof(double));
                    cmpIP_l[userID] = cmp_ip;
                }
                read_disk_time_ += read_disk_record_.get_elapsed_time_second();

                compare_ip_record_.reset();
                for (int userID = 0; userID < n_user; userID++) {
                    if (queryIP_l[userID] > cmpIP_l[userID]) {
                        double queryIP = queryIP_l[userID];
                        query_heap_l[qID].emplace_back(userID, queryIP);
                    }
                }
                compare_ip_time_ += compare_ip_record_.get_elapsed_time_second();

            }

            index_stream_.close();

            return query_heap_l;
        }

    };

    const int write_every_ = 30000;
    const int report_batch_every_ = 5;

    /*
     * bruteforce index
     * shape: 1, type: int, n_data_item
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    Index &BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const char *index_path) {
        std::ofstream out(index_path, std::ios::binary | std::ios::out);
        if (!out) {
            spdlog::error("error in write result");
        }
        const int n_data_item = data_item.n_vector_;
        std::vector<double> distance_cache(write_every_ * n_data_item);
        const int vec_dim = data_item.vec_dim_;
        const int n_batch = user.n_vector_ / write_every_;
        const int n_remain = user.n_vector_ % write_every_;
        user.vectorNormalize();

        TimeRecord batch_report_record;
        batch_report_record.reset();
        for (int i = 0; i < n_batch; i++) {
#pragma omp parallel for default(none) shared(i, data_item, user, distance_cache) shared(write_every_, n_data_item, vec_dim)
            for (int cacheID = 0; cacheID < write_every_; cacheID++) {
                int userID = write_every_ * i + cacheID;
                for (int itemID = 0; itemID < n_data_item; itemID++) {
                    double ip = InnerProduct(data_item.getVector(itemID), user.getVector(userID), vec_dim);
                    distance_cache[cacheID * n_data_item + itemID] = ip;
                }
                std::sort(distance_cache.begin() + cacheID * n_data_item,
                          distance_cache.begin() + (cacheID + 1) * n_data_item, std::greater<double>());
            }
            out.write((char *) distance_cache.data(), distance_cache.size() * sizeof(double));

            if (i % report_batch_every_ == 0) {
                spdlog::info("preprocessed {}%, {} s/iter Mem: {} Mb", i / (0.01 * n_batch),
                             batch_report_record.get_elapsed_time_second(), get_current_RSS() / 1000000);
                batch_report_record.reset();
            }

        }

        if (n_remain != 0) {
            for (int cacheID = 0; cacheID < n_remain; cacheID++) {
                int userID = cacheID + write_every_ * n_batch;
                for (int itemID = 0; itemID < data_item.n_vector_; itemID++) {
                    double ip = InnerProduct(data_item.getRawData() + itemID * vec_dim,
                                             user.getRawData() + userID * vec_dim, vec_dim);
                    distance_cache[cacheID * data_item.n_vector_ + itemID] = ip;
                }

                std::sort(distance_cache.begin() + cacheID * n_data_item,
                          distance_cache.begin() + (cacheID + 1) * n_data_item, std::greater<double>());
            }

            out.write((char *) distance_cache.data(),
                      n_remain * data_item.n_vector_ * sizeof(double));
        }

        static Index index(index_path, n_data_item, user);
        return index;
    }

}

#endif //REVERSE_K_RANKS_DISKBRUTEFORCE_HPP
