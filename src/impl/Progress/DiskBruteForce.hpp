//
// Created by BianZheng on 2022/2/25.
//

#ifndef REVERSE_KRANKS_DISKBRUTEFORCE_HPP
#define REVERSE_KRANKS_DISKBRUTEFORCE_HPP

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

    class Index : public BaseIndex {
        void ResetTimer() {
            total_retrieval_time_ = 0;
            read_disk_time_ = 0;
            inner_product_time_ = 0;
            binary_search_time_ = 0;
        }

    public:
        VectorMatrix user_;
        int vec_dim_, n_data_item_;
        size_t n_cache; //should larger than top-k
        double total_retrieval_time_, read_disk_time_, inner_product_time_, binary_search_time_;
        TimeRecord total_retrieval_record_, read_disk_record_, inner_product_record_, binary_search_record_;
        const char *index_path_;

        Index() {}

        Index(const char *index_path, const int n_data_item, VectorMatrix &user) {
            this->index_path_ = index_path;
            this->vec_dim_ = user.vec_dim_;
            this->user_ = std::move(user);
            this->n_data_item_ = n_data_item;
            this->n_cache = std::min(user_.n_vector_, 10000);
        }

        std::vector<std::vector<UserRankElement>>
        Retrieval(const VectorMatrix &query_item, const int &topk, const int &n_execute_query,
                  std::vector<SingleQueryPerformance> &query_performance_l) override {
            TimeRecord query_record;
            ResetTimer();
            std::ifstream index_stream_ = std::ifstream(this->index_path_, std::ios::binary | std::ios::in);
            if (!index_stream_) {
                spdlog::error("error in writing index");
            }

            if (n_execute_query > query_item.n_vector_) {
                spdlog::error("n_execute_query larger than n_query_item, program exit");
                exit(-1);
            }

            if (topk > user_.n_vector_) {
                spdlog::error("top-k is too large, program exit");
                exit(-1);
            }

//            size_t avail_memory = get_avail_memory();
            size_t a = user_.n_vector_;
//            size_t b = (size_t) 4000000000 / n_data_item_;
//            size_t b = (size_t) 4000000 / n_data_item_;
            size_t b = 20000;
            n_cache = std::min(a, b);
//            n_cache = a > b ? b : a;
            std::vector<float> distance_cache(n_cache * n_data_item_);
            int n_query_item = n_execute_query;
            int n_user = user_.n_vector_;
            int n_batch = (int) (n_user / n_cache);
            int n_remain = (int) (n_user % n_cache);
            const int report_query_every_ = 30;
            spdlog::info("n_user {}, n_batch {}, n_remain {}, n_cache {}", n_user, n_batch, n_remain, n_cache);

            std::vector<std::vector<UserRankElement>> query_heap_l(n_query_item, std::vector<UserRankElement>(topk));

            std::vector<float> queryIP_l(n_user);
            std::vector<int> rank_l(n_user);

            TimeRecord heap_record;
            double tmp_inner_product_time = 0;
            double tmp_binary_search_time = 0;
            double tmp_read_disk_time = 0;
            double tmp_other_time = 0;

            query_record.reset();
            for (int qID = 0; qID < n_query_item; qID++) {
                total_retrieval_record_.reset();
                //calculate distance
                float *query_item_vec = query_item.getVector(qID);
                inner_product_record_.reset();
                for (int userID = 0; userID < n_user; userID++) {
                    float *user_vec = user_.getVector(userID);
                    float queryIP = InnerProduct(query_item_vec, user_vec, vec_dim_);
                    queryIP_l[userID] = queryIP;
                }
                this->inner_product_time_ += inner_product_record_.get_elapsed_time_second();
                tmp_inner_product_time += inner_product_record_.get_elapsed_time_second();

                index_stream_.seekg(0, std::ios::beg);
                for (int batchID = 0; batchID < n_batch; batchID++) {
                    read_disk_record_.reset();
                    int64_t read_count = (int64_t) n_cache * n_data_item_ * sizeof(float);
                    index_stream_.read((char *) distance_cache.data(), read_count);
                    read_disk_time_ += read_disk_record_.get_elapsed_time_second();
                    tmp_read_disk_time += read_disk_record_.get_elapsed_time_second();

                    binary_search_record_.reset();
                    for (int cacheID = 0; cacheID < n_cache; cacheID++) {
                        int userID = batchID * n_cache + cacheID;
                        int tmp_rank = BinarySearch(queryIP_l[userID], cacheID, distance_cache);
                        rank_l[userID] = tmp_rank;
                    }
                    binary_search_time_ += binary_search_record_.get_elapsed_time_second();
                    tmp_binary_search_time += binary_search_record_.get_elapsed_time_second();

                }

                read_disk_record_.reset();
                int64_t read_count = (int64_t) n_remain * n_data_item_ * sizeof(float);
                index_stream_.read((char *) distance_cache.data(), read_count);
                read_disk_time_ += read_disk_record_.get_elapsed_time_second();
                tmp_read_disk_time += read_disk_record_.get_elapsed_time_second();

                binary_search_record_.reset();
                for (int cacheID = 0; cacheID < n_remain; cacheID++) {
                    int userID = n_batch * n_cache + cacheID;
                    int tmp_rank = BinarySearch(queryIP_l[userID], cacheID, distance_cache);
                    rank_l[userID] = tmp_rank;
                }
                binary_search_time_ += binary_search_record_.get_elapsed_time_second();
                tmp_binary_search_time += binary_search_record_.get_elapsed_time_second();

                heap_record.reset();
                std::vector<UserRankElement> &query_heap = query_heap_l[qID];
                for (int userID = 0; userID < topk; userID++) {
                    query_heap[userID] = UserRankElement(userID, rank_l[userID], queryIP_l[userID]);
                }

                std::make_heap(query_heap_l[qID].begin(), query_heap_l[qID].end(), std::less<UserRankElement>());

                for (int userID = topk; userID < n_user; userID++) {
                    UserRankElement min_heap_ele = query_heap.front();
                    int tmp_rank = rank_l[userID];
                    float queryIP = queryIP_l[userID];
                    UserRankElement element(userID, tmp_rank, queryIP);
                    if (min_heap_ele > element) {
                        std::pop_heap(query_heap.begin(), query_heap.end(), std::less<UserRankElement>());
                        query_heap.pop_back();
                        query_heap.push_back(element);
                        std::push_heap(query_heap.begin(), query_heap.end(), std::less<UserRankElement>());
                    }
                }
                tmp_other_time += heap_record.get_elapsed_time_second();
                total_retrieval_time_ += total_retrieval_record_.get_elapsed_time_second();

                if (qID % report_query_every_ == 0) {
                    spdlog::info("top-{} retrieval query number {}%, {} s/iter Mem: {} Mb", topk,
                                 qID / (0.01 * n_query_item),
                                 query_record.get_elapsed_time_second(), get_current_RSS() / 1000000);
                    query_record.reset();
                    spdlog::info("tmp: inner product {}s, binary search {}s, read disk {}s, other {}s",
                                 tmp_inner_product_time, tmp_binary_search_time, tmp_read_disk_time, tmp_other_time);
                    tmp_inner_product_time = 0;
                    tmp_binary_search_time = 0;
                    tmp_read_disk_time = 0;
                    tmp_other_time = 0;
                }

            }

            index_stream_.close();

            for (int qID = 0; qID < n_query_item; qID++) {
                std::sort(query_heap_l[qID].begin(), query_heap_l[qID].end(), std::less<UserRankElement>());
            }

            return query_heap_l;
        }

        int BinarySearch(float queryIP, int cacheID, std::vector<float> &distance_cache) const {
            auto iter_begin = distance_cache.begin() + cacheID * n_data_item_;
            auto iter_end = distance_cache.begin() + (cacheID + 1) * n_data_item_;

            auto lb_ptr = std::lower_bound(iter_begin, iter_end, queryIP,
                                           [](const float &arrIP, float queryIP) {
                                               return arrIP > queryIP;
                                           });
            return (int) (lb_ptr - iter_begin) + 1;
        }

        std::string
        PerformanceStatistics(const int &topk) override {
            // int topk;
            //double total_time,
            //          inner_product_time, read_disk_time, binary_search_time;
            //unit: second

            char buff[1024];

            sprintf(buff,
                    "top%d retrieval time:\n\ttotal %.3fs\n\tinner product %.3fs, read disk %.3fs, binary search %.3fs",
                    topk, total_retrieval_time_,
                    inner_product_time_, read_disk_time_, binary_search_time_);
            std::string str(buff);
            return str;
        }

        uint64_t IndexSizeByte() override {
            return 0;
        }

    };

    const int write_every_ = 30000;
    const int report_batch_every_ = 5;

    /*
     * bruteforce index
     * shape: 1, type: int, n_data_item
     * shape: n_user * n_data_item, type: float, the distance pair for each user
     */

    std::unique_ptr<Index> BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const char *index_path) {
        std::ofstream out(index_path, std::ios::binary | std::ios::out);
        if (!out) {
            spdlog::error("error in write result");
        }
        const int n_data_item = data_item.n_vector_;
        std::vector<float> distance_cache(write_every_ * n_data_item);
        const int vec_dim = data_item.vec_dim_;
        const int n_batch = user.n_vector_ / write_every_;
        const int n_remain = user.n_vector_ % write_every_;

        TimeRecord batch_report_record;
        batch_report_record.reset();
        for (int i = 0; i < n_batch; i++) {
#pragma omp parallel for default(none) shared(i, data_item, user, distance_cache) shared(write_every_, n_data_item, vec_dim)
            for (int cacheID = 0; cacheID < write_every_; cacheID++) {
                int userID = write_every_ * i + cacheID;
                for (int itemID = 0; itemID < n_data_item; itemID++) {
                    float ip = InnerProduct(data_item.getVector(itemID), user.getVector(userID), vec_dim);
                    distance_cache[cacheID * n_data_item + itemID] = ip;
                }
                std::sort(distance_cache.begin() + cacheID * n_data_item,
                          distance_cache.begin() + (cacheID + 1) * n_data_item, std::greater<float>());
            }
            out.write((char *) distance_cache.data(), distance_cache.size() * sizeof(float));

            if (i % report_batch_every_ == 0) {
                spdlog::info("preprocessed {}%, {} s/iter Mem: {} Mb", i / (0.01 * n_batch),
                             batch_report_record.get_elapsed_time_second(), get_current_RSS() / 1000000);
                batch_report_record.reset();
            }

        }

        for (int cacheID = 0; cacheID < n_remain; cacheID++) {
            int userID = cacheID + write_every_ * n_batch;
            for (int itemID = 0; itemID < data_item.n_vector_; itemID++) {
                float ip = InnerProduct(data_item.getRawData() + itemID * vec_dim,
                                         user.getRawData() + userID * vec_dim, vec_dim);
                distance_cache[cacheID * data_item.n_vector_ + itemID] = ip;
            }

            std::sort(distance_cache.begin() + cacheID * n_data_item,
                      distance_cache.begin() + (cacheID + 1) * n_data_item, std::greater<float>());
        }

        out.write((char *) distance_cache.data(),
                  n_remain * data_item.n_vector_ * sizeof(float));

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(index_path, n_data_item, user);
        return index_ptr;
    }

}

#endif //REVERSE_KRANKS_DISKBRUTEFORCE_HPP
