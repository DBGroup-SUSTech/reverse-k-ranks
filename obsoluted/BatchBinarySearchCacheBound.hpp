//
// Created by BianZheng on 2022/2/20.
//

#ifndef REVERSE_KRANKS_BATCH_BINARYSEARCHCACHEBOUND_HPP
#define REVERSE_KRANKS_BATCH_BINARYSEARCHCACHEBOUND_HPP

#include "struct/VectorMatrix.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/MethodBase.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "util/TimeMemory.hpp"
#include "util/VectorIO.hpp"
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <set>
#include <cassert>
#include <spdlog/spdlog.h>

namespace ReverseMIPS::BinarySearchCacheBound {
    class RetrievalResult {
    public:
        //unit: second
        double total_time, read_disk_time, inner_product_time, coarse_binary_search_time, fine_binary_search_time, second_per_query;
        int topk;

        inline RetrievalResult(double total_time, double read_disk_time, double inner_product_time,
                               double coarse_binary_search_time, double fine_binary_search_time,
                               double second_per_query, int topk) {
            this->total_time = total_time;
            this->read_disk_time = read_disk_time;
            this->inner_product_time = inner_product_time;
            this->coarse_binary_search_time = coarse_binary_search_time;
            this->fine_binary_search_time = fine_binary_search_time;
            this->second_per_query = second_per_query;

            this->topk = topk;
        }

        void AddMap(std::map<std::string, std::string> &performance_m) const {
            char buff[256];
            sprintf(buff, "top%d retrieval\t\t total time", topk);
            std::string str1(buff);
            performance_m.emplace(str1, double2string(total_time));

            sprintf(buff, "top%d retrieval\t\t read disk time", topk);
            std::string str2(buff);
            performance_m.emplace(str2, double2string(read_disk_time));

            sprintf(buff, "top%d retrieval\t\t inner product time", topk);
            std::string str3(buff);
            performance_m.emplace(str3, double2string(inner_product_time));

            sprintf(buff, "top%d retrieval\t\t coarse binary search time", topk);
            std::string str4(buff);
            performance_m.emplace(str4, double2string(coarse_binary_search_time));

            sprintf(buff, "top%d retrieval\t\t fine binary search time", topk);
            std::string str5(buff);
            performance_m.emplace(str5, double2string(fine_binary_search_time));

            sprintf(buff, "top%d retrieval\t\t second per query time", topk);
            std::string str6(buff);
            performance_m.emplace(str6, double2string(second_per_query));
        }

        [[nodiscard]] std::string ToString() const {
            char arr[512];
            sprintf(arr,
                    "top%d retrieval time:\n\ttotal %.3fs, read disk %.3fs\n\tinner product %.3fs, coarse binary search %.3fs, fine binary search %.3fs, million second per query %.3fms",
                    topk, total_time, read_disk_time, inner_product_time, coarse_binary_search_time,
                    fine_binary_search_time,
                    second_per_query * 1000);
            std::string str(arr);
            return str;
        }

    };

    class Index : public BaseIndex {
        void ResetTimer() {
            read_disk_time_ = 0;
            inner_product_time_ = 0;
            coarse_binary_search_time_ = 0;
            fine_binary_search_time_ = 0;
        }

    public:
        //bound for binary search, store in memory
        std::vector<double> bound_distance_table_; // n_user * n_cache_rank_
        std::vector<int> known_rank_idx_l_; // n_cache_rank_
        int n_cache_rank_;

        //read index on disk
        const char *index_path_;

        VectorMatrix user_;
        int vec_dim_, n_data_item_, n_user_;
        double read_disk_time_, inner_product_time_, coarse_binary_search_time_, fine_binary_search_time_;
        TimeRecord read_disk_record_, inner_product_record_, coarse_binary_search_record_, fine_binary_search_record_;

        Index(const std::vector<double> &bound_distance_table,
              const std::vector<int> &known_rank_idx_l,
              const char *index_path) {
            this->bound_distance_table_ = bound_distance_table;
            this->known_rank_idx_l_ = known_rank_idx_l;
            this->index_path_ = index_path;
            this->n_cache_rank_ = (int) known_rank_idx_l.size();
        }

        void setUserItemMatrix(const VectorMatrix &user, const VectorMatrix &data_item) {
            this->user_ = user;
            this->n_user_ = user.n_vector_;
            this->n_data_item_ = data_item.n_vector_;
            this->vec_dim_ = user.vec_dim_;
        }

        std::vector<std::vector<UserRankElement>> Retrieval(VectorMatrix &query_item, const int topk) override {
            ResetTimer();
            std::ifstream index_stream_ = std::ifstream(this->index_path_, std::ios::binary | std::ios::in);
            if (!index_stream_) {
                std::printf("error in writing index\n");
            }

            if (topk > user_.n_vector_) {
                printf("top-k is too large, program exit\n");
                exit(-1);
            }

            //coarse binary search
            //store bucketID in the set
            const int n_query_item = query_item.n_vector_;

            //first dimension: userID, key: bucketID, value: queryItemID, queryIP, shape: n_user * unordered_map
            std::vector<std::unordered_map<int, std::vector<std::pair<int, double>>>> candidates_invert_index_l(
                    n_user_, std::unordered_map<int, std::vector<std::pair<int, double>>>());
            //store the bucketID that queryIP fall in, for each query. used for coarse binary search
            std::vector<UserRankElement> user_bucket_l(n_user_);

            for (int queryID = 0; queryID < n_query_item; ++queryID) {
                for (int userID = 0; userID < n_user_; ++userID) {
                    inner_product_record_.reset();
                    double *user_vec = user_.getVector(userID);
                    double *query_item_vec = query_item.getVector(queryID);
                    double queryIP = InnerProduct(query_item_vec, user_vec, vec_dim_);
                    this->inner_product_time_ += inner_product_record_.get_elapsed_time_second();
                    coarse_binary_search_record_.reset();
                    int bucketID = MemoryBinarySearch(queryIP, userID);
                    this->coarse_binary_search_time_ += coarse_binary_search_record_.get_elapsed_time_second();
                    assert(0 <= bucketID && bucketID <= n_cache_rank_);
                    user_bucket_l[userID] = UserRankElement(userID, bucketID, queryIP);
                }
                //small bucketID means higher rank
                std::sort(user_bucket_l.begin(), user_bucket_l.end(), std::less<UserRankElement>());
                int topk_bucketID = user_bucket_l[topk - 1].rank_;
                int end_ptr = topk;
                while (end_ptr < n_user_ && topk_bucketID == user_bucket_l[end_ptr].rank_) {
                    ++end_ptr;
                }
                for (int i = 0; i < end_ptr; ++i) {
                    int tmp_userID = user_bucket_l[i].userID_;
                    int tmp_bucketID = user_bucket_l[i].rank_;
                    double tmp_queryIP = user_bucket_l[i].queryIP_;

                    auto find_iter = candidates_invert_index_l[tmp_userID].find(tmp_bucketID);
                    if (find_iter == candidates_invert_index_l[tmp_userID].end()) {
                        candidates_invert_index_l[tmp_userID].insert(
                                std::make_pair(tmp_bucketID, std::vector<std::pair<int, double>>{
                                        std::make_pair(queryID, tmp_queryIP)}));
                    } else {
                        find_iter->second.emplace_back(queryID, tmp_queryIP);
                    }

                }
            }
            user_bucket_l.clear();

            std::vector<std::vector<UserRankElement>> query_heap_l(n_query_item, std::vector<UserRankElement>());
            for (int queryID = 0; queryID < n_query_item; ++queryID) {
                query_heap_l.reserve(topk);
            }

            //read the candidate rank all in one time
            std::vector<double> distance_cache(n_data_item_);
            for (int userID = 0; userID < n_user_; userID++) {
                std::unordered_map<int, std::vector<std::pair<int, double>>> &invert_index = candidates_invert_index_l[userID];
                for (auto &iter: invert_index) {
                    int bucketID = iter.first;
                    int start_idx = bucketID == 0 ? 0 : known_rank_idx_l_[bucketID - 1] + 1;
                    int end_idx = bucketID == n_cache_rank_ ? n_data_item_ : known_rank_idx_l_[bucketID];
                    assert(start_idx < end_idx);
                    read_disk_record_.reset();
                    index_stream_.seekg(sizeof(double) * (userID * n_data_item_ + start_idx), std::ios::beg);
                    index_stream_.read((char *) distance_cache.data(), (end_idx - start_idx) * sizeof(double));
                    read_disk_time_ += read_disk_record_.get_elapsed_time_second();
                    auto start_iter = distance_cache.begin();
                    auto end_iter = distance_cache.begin() + end_idx - start_idx;
                    for (auto &queryIter: iter.second) {
                        int queryID = queryIter.first;
                        double queryIP = queryIter.second;
                        fine_binary_search_record_.reset();
                        auto lb_ptr = std::lower_bound(start_iter, end_iter, queryIP,
                                                       [](const double &info, double value) {
                                                           return info > value;
                                                       });
                        fine_binary_search_time_ += fine_binary_search_record_.get_elapsed_time_second();
                        int offset_rank = (int) (lb_ptr - start_iter);
                        int base_rank = bucketID == 0 ? 0 : known_rank_idx_l_[bucketID - 1] + 1;
                        int rank = base_rank + offset_rank + 1;
                        if (query_heap_l[queryID].size() < topk) {
                            query_heap_l[queryID].emplace_back(userID, rank, queryIP);
                        } else {
                            std::vector<UserRankElement> &minHeap = query_heap_l[queryID];
                            std::make_heap(minHeap.begin(), minHeap.end(), std::less<UserRankElement>());
                            UserRankElement minHeapEle = minHeap.front();
                            UserRankElement rankElement(userID, rank, queryIP);
                            if (minHeapEle.rank_ > rankElement.rank_) {
                                std::pop_heap(minHeap.begin(), minHeap.end(), std::less<UserRankElement>());
                                minHeap.pop_back();
                                minHeap.push_back(rankElement);
                                std::push_heap(minHeap.begin(), minHeap.end(), std::less<UserRankElement>());
                            }

                        }
                    }
                }
            }

            index_stream_.close();

            for (int qID = 0; qID < n_query_item; qID++) {
                std::sort(query_heap_l[qID].begin(), query_heap_l[qID].end(), std::less<UserRankElement>());
                assert(query_heap_l[qID].size() == topk);
            }
            return query_heap_l;
        }

        //return the index of the bucket it belongs to
        [[nodiscard]] int MemoryBinarySearch(double queryIP, int userID) const {
            auto iter_begin = bound_distance_table_.begin() + userID * n_cache_rank_;
            auto iter_end = bound_distance_table_.begin() + (userID + 1) * n_cache_rank_;

            auto lb_ptr = std::lower_bound(iter_begin, iter_end, queryIP,
                                           [](const double &arrIP, double queryIP) {
                                               return arrIP > queryIP;
                                           });
            return (int) (lb_ptr - iter_begin);
        }

    };

    const int write_every_ = 100;
    const int report_batch_every_ = 5;

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    Index BuildIndex(const VectorMatrix &data_item, const VectorMatrix &user, const char *index_path) {
        std::ofstream out(index_path, std::ios::binary | std::ios::out);
        if (!out) {
            std::printf("error in write result\n");
        }
        const int n_data_item = data_item.n_vector_;
        std::vector<double> write_distance_cache(write_every_ * n_data_item);
        const int vec_dim = data_item.vec_dim_;
        const int n_batch = user.n_vector_ / write_every_;
        const int n_remain = user.n_vector_ % write_every_;

        //隔着多少个建模
        const int cache_bound_every = 10;
        const int n_cache_rank = n_data_item / cache_bound_every;
        std::vector<int> known_rank_idx_l;
        for (int known_rank_idx = cache_bound_every - 1;
             known_rank_idx < n_data_item; known_rank_idx += cache_bound_every) {
            known_rank_idx_l.emplace_back(known_rank_idx);
        }
        assert(known_rank_idx_l.size() == n_cache_rank);

        //used for coarse binary search
        std::vector<double> bound_distance_table(user.n_vector_ * n_cache_rank);

        TimeRecord batch_report_record;
        batch_report_record.reset();
        for (int i = 0; i < n_batch; i++) {
#pragma omp parallel for default(none) shared(i, data_item, user, write_distance_cache, bound_distance_table, known_rank_idx_l) shared(n_cache_rank, write_every_, n_data_item, vec_dim)
            for (int cacheID = 0; cacheID < write_every_; cacheID++) {
                int userID = write_every_ * i + cacheID;
                for (int itemID = 0; itemID < n_data_item; itemID++) {
                    double ip = InnerProduct(data_item.getVector(itemID), user.getVector(userID), vec_dim);
                    write_distance_cache[cacheID * n_data_item + itemID] = ip;
                }
                std::sort(write_distance_cache.begin() + cacheID * n_data_item,
                          write_distance_cache.begin() + (cacheID + 1) * n_data_item, std::greater<double>());

                auto array_begin = write_distance_cache.begin() + cacheID * n_data_item;
                for (int bucketID = 0; bucketID < n_cache_rank; bucketID++) {
                    int itemID = known_rank_idx_l[bucketID];
                    bound_distance_table[userID * n_cache_rank + bucketID] = array_begin[itemID];
                }
            }
            out.write((char *) write_distance_cache.data(), write_distance_cache.size() * sizeof(double));

            if (i % report_batch_every_ == 0) {
                std::cout << "preprocessed " << i / (0.01 * n_batch) << " %, "
                          << batch_report_record.get_elapsed_time_second() << " s/iter" << " Mem: "
                          << get_current_RSS() / 1000000 << " Mb \n";
                batch_report_record.reset();
            }

        }

        for (int cacheID = 0; cacheID < n_remain; cacheID++) {
            int userID = write_every_ * n_batch + cacheID;
            for (int itemID = 0; itemID < data_item.n_vector_; itemID++) {
                double ip = InnerProduct(data_item.rawData_ + itemID * vec_dim,
                                         user.rawData_ + userID * vec_dim, vec_dim);
                write_distance_cache[cacheID * data_item.n_vector_ + itemID] = ip;
            }

            std::sort(write_distance_cache.begin() + cacheID * n_data_item,
                      write_distance_cache.begin() + (cacheID + 1) * n_data_item, std::greater<double>());

            auto array_begin = write_distance_cache.begin() + cacheID * n_data_item;
            for (int bucketID = 0; bucketID < n_cache_rank; bucketID++) {
                int itemID = known_rank_idx_l[bucketID];
                bound_distance_table[userID * n_cache_rank + bucketID] = array_begin[itemID];
            }
        }

        out.write((char *) write_distance_cache.data(),
                  n_remain * data_item.n_vector_ * sizeof(double));
        Index index(bound_distance_table, known_rank_idx_l, index_path);
        index.setUserItemMatrix(user, data_item);
        return index;
    }

}
#endif //REVERSE_KRANKS_BATCH_BINARYSEARCHCACHEBOUND_HPP
