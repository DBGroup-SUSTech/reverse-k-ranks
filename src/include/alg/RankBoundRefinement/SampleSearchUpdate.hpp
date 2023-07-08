//
// Created by bianzheng on 2023/6/16.
//

#ifndef REVERSE_KRANKS_SAMPLESEARCHUPDATE_HPP
#define REVERSE_KRANKS_SAMPLESEARCHUPDATE_HPP

#include "struct/DistancePair.hpp"
#include <iostream>
#include <fstream>
#include <memory>
#include <cfloat>
#include <omp.h>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class SampleSearchUpdate {

        size_t n_sample_, n_data_item_, n_user_;
        size_t n_insert_item_, n_delete_item_;
        int n_thread_;
        std::unique_ptr<float[]> bound_distance_table_; // n_user * n_sample_
        std::unique_ptr<float[]> insert_item_score_; // n_user_ * n_insert_item
        std::unique_ptr<float[]> delete_item_score_; // n_user_ * n_delete_item
    public:
        std::unique_ptr<int[]> known_rank_idx_l_; // n_sample_

        inline SampleSearchUpdate() {}

        inline SampleSearchUpdate(const char *index_path, const char *dataset_name,
                                  const char *sample_search_index_name,
                                  const size_t &n_sample,
                                  const bool &load_sample_score, const bool &is_query_distribution,
                                  const size_t &n_sample_query = 0, const size_t &sample_topk = 0,
                                  const int &n_thread = omp_get_num_procs()) {
            n_thread_ = n_thread;
            LoadIndex(index_path, dataset_name, sample_search_index_name,
                      n_sample,
                      load_sample_score, is_query_distribution,
                      n_sample_query, sample_topk);

            std::cout << "first 50 rank: ";
            const int end_rankID = std::min((int) n_sample_, 50);
            for (int rankID = 0; rankID < end_rankID; rankID++) {
                std::cout << known_rank_idx_l_[rankID] << " ";
            }
            std::cout << std::endl;
            std::cout << "last 50 rank: ";
            const int start_rankID = std::max(0, (int) n_sample_ - 50);
            for (int rankID = start_rankID; rankID < n_sample_; rankID++) {
                std::cout << known_rank_idx_l_[rankID] << " ";
            }
            std::cout << std::endl;
            n_insert_item_ = 0;
            n_delete_item_ = 0;
            insert_item_score_ = nullptr;
            delete_item_score_ = nullptr;
        }

        void LoopPreprocess(const float *distance_ptr, const int &userID) {
            for (int crankID = 0; crankID < n_sample_; crankID++) {
                unsigned int rankID = known_rank_idx_l_[crankID];
                bound_distance_table_[(size_t) n_sample_ * userID + crankID] = distance_ptr[rankID];
            }
        }

        [[nodiscard]] const float *SampleData(const int &userID) const {
            return bound_distance_table_.get() + (size_t) n_sample_ * userID;
        }

        void InsertUser(const VectorMatrixUpdate &insert_user, const float *distance_ptr) {
            std::unique_ptr<float[]> new_bound_distance_table = std::make_unique<float[]>(
                    (size_t) (n_user_ + insert_user.n_vector_) * n_sample_);
            std::memcpy(new_bound_distance_table.get(), bound_distance_table_.get(),
                        (size_t) sizeof(float) * n_sample_ * n_user_);
            for (int userID = 0; userID < insert_user.n_vector_; userID++) {
                const int new_userID = (int) n_user_ + userID;
                for (int crankID = 0; crankID < n_sample_; crankID++) {
                    unsigned int rank = known_rank_idx_l_[crankID];
                    new_bound_distance_table[(size_t) new_userID * n_sample_ + crankID] =
                            distance_ptr[(size_t) userID * n_data_item_ + rank];
                }
            }
            bound_distance_table_ = std::move(new_bound_distance_table);
            this->n_user_ = this->n_user_ + insert_user.n_vector_;
        }

        void DeleteUser(const std::vector<int> &del_userID_l) {
            std::unique_ptr<float[]> new_bound_distance_table = std::make_unique<float[]>(
                    (size_t) n_sample_ * (n_user_ - del_userID_l.size()));

            std::unordered_set<int> del_userID_set(del_userID_l.begin(), del_userID_l.end());
            std::vector<int> remain_userID_l;
            for (int userID = 0; userID < n_user_; userID++) {
                if (del_userID_set.find(userID) == del_userID_set.end()) {
                    remain_userID_l.emplace_back(userID);
                }
            }
            assert(remain_userID_l.size() + del_userID_l.size() == n_user_);
            for (int candID = 0; candID < remain_userID_l.size(); candID++) {
                const int userID = remain_userID_l[candID];
                std::memcpy(new_bound_distance_table.get() + (size_t) candID * n_sample_,
                            bound_distance_table_.get() + (size_t) userID * n_sample_,
                            (size_t) sizeof(float) * n_sample_);
            }
            bound_distance_table_ = std::move(new_bound_distance_table);
            this->n_user_ = this->n_user_ - del_userID_l.size();
        }

        void InsertItem(const VectorMatrixUpdate &insert_data_item, const float *item_score_ptr) {

            std::unique_ptr<float[]> new_insert_item_score = std::make_unique<float[]>(
                    n_user_ * (n_insert_item_ + insert_data_item.n_vector_));
            for (int userID = 0; userID < n_user_; userID++) {
                if (insert_item_score_ != nullptr) {
                    assert(n_insert_item_ != 0);
                    std::memcpy(new_insert_item_score.get() +
                                (size_t) userID * (n_insert_item_ + insert_data_item.n_vector_),
                                insert_item_score_.get() + (size_t) userID * n_insert_item_,
                                (size_t) sizeof(float) * n_insert_item_);
                } else {
                    assert(n_insert_item_ == 0);
                }
                std::memcpy(new_insert_item_score.get() +
                            (size_t) userID * (n_insert_item_ + insert_data_item.n_vector_) +
                            (size_t) n_insert_item_,
                            item_score_ptr + (size_t) userID * insert_data_item.n_vector_,
                            (size_t) sizeof(float) * insert_data_item.n_vector_);
                std::sort(new_insert_item_score.get() +
                          (size_t) userID * (n_insert_item_ + insert_data_item.n_vector_),
                          new_insert_item_score.get() +
                          (size_t) (userID + 1) * (n_insert_item_ + insert_data_item.n_vector_),
                          std::greater());

            }
            insert_item_score_ = std::move(new_insert_item_score);

            n_insert_item_ += insert_data_item.n_vector_;

        }

        void DeleteItem(const std::vector<int> &del_itemID_l, const float *item_score_ptr) {
            std::unique_ptr<float[]> new_delete_item_score = std::make_unique<float[]>(
                    n_user_ * (n_delete_item_ + del_itemID_l.size()));
            for (int userID = 0; userID < n_user_; userID++) {
                if (delete_item_score_ != nullptr) {
                    std::memcpy(new_delete_item_score.get() +
                                (size_t) userID * (n_delete_item_ + del_itemID_l.size()),
                                delete_item_score_.get() + (size_t) userID * n_delete_item_,
                                (size_t) sizeof(float) * n_delete_item_);
                } else {
                    assert(n_delete_item_ == 0);
                }
                std::memcpy(new_delete_item_score.get() +
                            (size_t) userID * (n_delete_item_ + del_itemID_l.size()) +
                            (size_t) n_delete_item_,
                            item_score_ptr + (size_t) userID * del_itemID_l.size(),
                            (size_t) sizeof(float) * del_itemID_l.size());
                std::sort(new_delete_item_score.get() + userID * (n_delete_item_ + del_itemID_l.size()),
                          new_delete_item_score.get() + (userID + 1) * (n_delete_item_ + del_itemID_l.size()),
                          std::greater());

            }
            delete_item_score_ = std::move(new_delete_item_score);

            n_delete_item_ += del_itemID_l.size();
        }

        inline void
        CoarseBinarySearch(const float &queryIP, const int &userID,
                           int &rank_lb, int &rank_ub) const {
            float *search_iter = bound_distance_table_.get() + (size_t) n_sample_ * userID;

            float *iter_begin = search_iter;
            float *iter_end = search_iter + n_sample_;

            float *lb_ptr = std::lower_bound(iter_begin, iter_end, queryIP,
                                             [](const float &arrIP, float queryIP) {
                                                 return arrIP >= queryIP;
                                             });
//            float *lb_ptr = std::lower_bound(iter_begin, iter_end, queryIP,
//                                              [](const float &arrIP, float queryIP) {
//                                                  return arrIP > queryIP;
//                                              });
            unsigned int bucket_idx = (lb_ptr - iter_begin);
            unsigned int tmp_rank_lb = bucket_idx == n_sample_ ? n_data_item_ : known_rank_idx_l_[bucket_idx];
            unsigned int tmp_rank_ub = bucket_idx == 0 ? 0 : known_rank_idx_l_[bucket_idx - 1];
            assert(tmp_rank_ub <= tmp_rank_lb);


            if (bucket_idx == n_sample_) {
                rank_lb = (int) n_data_item_;
                rank_ub = (int) tmp_rank_ub;
            } else if (bucket_idx == 0) {
                rank_lb = (int) tmp_rank_lb;
                rank_ub = 0;
            } else if (tmp_rank_lb - tmp_rank_ub <= 1) {
                rank_lb = (int) tmp_rank_lb;
                rank_ub = (int) tmp_rank_lb;
            } else {
                rank_lb = (int) tmp_rank_lb;
                rank_ub = (int) tmp_rank_ub;
            }

            assert(0 <= rank_lb - rank_ub &&
                   rank_lb - rank_ub <= std::max(known_rank_idx_l_[n_sample_ - 1],
                                                 (int) n_data_item_ - known_rank_idx_l_[n_sample_ - 1]));
        }

        inline int
        SearchInsertDelete(const float &queryIP, const int &userID) const {
            int rank_offset = 0;
            if (insert_item_score_ != nullptr || n_insert_item_ != 0) {
                assert(insert_item_score_ != nullptr && n_insert_item_ != 0);
                float *search_iter = insert_item_score_.get() + (size_t) userID * n_insert_item_;

                float *iter_begin = search_iter;
                float *iter_end = search_iter + n_insert_item_;

                float *lb_ptr = std::lower_bound(iter_begin, iter_end, queryIP,
                                                 [](const float &arrIP, float queryIP) {
                                                     return arrIP >= queryIP;
                                                 });
                rank_offset += (lb_ptr - iter_begin);
            }

            if (delete_item_score_ != nullptr || n_delete_item_ != 0) {
                assert(delete_item_score_ != nullptr && n_delete_item_ != 0);
                float *search_iter = delete_item_score_.get() + (size_t) userID * n_delete_item_;

                float *iter_begin = search_iter;
                float *iter_end = search_iter + n_delete_item_;

                float *lb_ptr = std::lower_bound(iter_begin, iter_end, queryIP,
                                                 [](const float &arrIP, float queryIP) {
                                                     return arrIP >= queryIP;
                                                 });
                rank_offset -= (lb_ptr - iter_begin);
            }

            return rank_offset;
        }

        void RankBound(const std::vector<float> &queryIP_l,
                       const std::vector<char> &prune_l, const std::vector<char> &result_l,
                       std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l) const {
            assert(queryIP_l.size() == n_user_);
            assert(prune_l.size() == n_user_);
            assert(result_l.size() == n_user_);
            assert(rank_lb_l.size() == n_user_);
            assert(rank_ub_l.size() == n_user_);

#pragma omp parallel for default(none) shared(prune_l, result_l, queryIP_l, rank_lb_l, rank_ub_l) num_threads(n_thread_) schedule(dynamic, 1000)
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID] || result_l[userID]) {
                    continue;
                }
                int lower_rank, upper_rank;
                float queryIP = queryIP_l[userID];

                CoarseBinarySearch(queryIP, userID,
                                   lower_rank, upper_rank);
                if (n_insert_item_ == 0 && n_delete_item_ == 0) {
                    rank_lb_l[userID] = lower_rank;
                    rank_ub_l[userID] = upper_rank;
                } else {
                    const int rank_offset = SearchInsertDelete(queryIP, userID);

                    const int new_rank_lb = lower_rank + rank_offset;
                    const int new_rank_ub = upper_rank + rank_offset;

                    rank_lb_l[userID] = Filter(new_rank_lb);
                    rank_ub_l[userID] = Filter(new_rank_ub);
                }

            }

        }

        [[nodiscard]] int Filter(const int input) const {
//            if (input < 0) {
//                return 0;
//            } else {
//                return input;
//            }
            if (input < 0) {
                return 0;
            } else if (input > n_data_item_ + n_insert_item_ - n_delete_item_) {
                return n_data_item_ + n_insert_item_ - n_delete_item_;
            } else {
                return input;
            }
        }

        void SearchInsertDelete(const std::vector<std::pair<float, float>> &queryIP_bound_l,
                                std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l, const int queryID = 0) const {
            assert(queryIP_bound_l.size() == n_user_);
            assert(rank_lb_l.size() == n_user_);
            assert(rank_ub_l.size() == n_user_);

            if (n_insert_item_ != 0 || n_delete_item_ != 0) {
#pragma omp parallel for default(none) shared(queryIP_bound_l, rank_lb_l, rank_ub_l, queryID) num_threads(n_thread_)
                for (int userID = 0; userID < n_user_; userID++) {
                    int lower_rank = rank_lb_l[userID];
                    int upper_rank = rank_ub_l[userID];
                    std::pair<float, float> queryIP_bound_pair = queryIP_bound_l[userID];

                    const int rank_lb_offset = SearchInsertDelete(queryIP_bound_pair.first, userID);
                    const int rank_ub_offset = SearchInsertDelete(queryIP_bound_pair.second, userID);

                    const int new_rank_lb = lower_rank + rank_lb_offset;
                    const int new_rank_ub = upper_rank + rank_ub_offset;

                    rank_lb_l[userID] = Filter(new_rank_lb);
                    rank_ub_l[userID] = Filter(new_rank_ub);

                    assert(0 <= rank_lb_l[userID] &&
                           rank_lb_l[userID] <= n_data_item_ + n_insert_item_ - n_delete_item_);
                    assert(0 <= rank_ub_l[userID] &&
                           rank_ub_l[userID] <= n_data_item_ + n_insert_item_ - n_delete_item_);
                }
            }

        }

        void SaveIndex(const char *index_basic_dir, const char *dataset_name,
                       const char *method_name,
                       const bool &save_sample_score, const bool &is_query_distribution,
                       const size_t &n_sample_query, const size_t &sample_topk) {
            char index_abs_dir[256];
            if (save_sample_score) {
                sprintf(index_abs_dir, "%s/memory_index", index_basic_dir);
            } else {
                sprintf(index_abs_dir, "%s/qrs_to_sample_index", index_basic_dir);
            }

            char index_path[512];
            if (is_query_distribution) {
                sprintf(index_path,
                        "%s/%s-%s-n_sample_%ld-n_sample_query_%ld-sample_topk_%ld.index",
                        index_abs_dir, method_name, dataset_name, n_sample_, n_sample_query, sample_topk);
            } else {
                sprintf(index_path,
                        "%s/%s-%s-n_sample_%ld.index",
                        index_abs_dir, method_name, dataset_name, n_sample_);

            }

            std::ofstream out_stream_ = std::ofstream(index_path, std::ios::binary | std::ios::out);
            if (!out_stream_) {
                spdlog::error("error in write result, not found index");
                exit(-1);
            }
            out_stream_.write((char *) &n_sample_, sizeof(size_t));
            out_stream_.write((char *) &n_data_item_, sizeof(size_t));
            out_stream_.write((char *) &n_user_, sizeof(size_t));

            out_stream_.write((char *) known_rank_idx_l_.get(), (int64_t) (n_sample_ * sizeof(int)));
            if (save_sample_score) {
                out_stream_.write((char *) bound_distance_table_.get(),
                                  (int64_t) (n_user_ * n_sample_ * sizeof(float)));
            }

            out_stream_.close();
        }

        void LoadIndex(const char *index_basic_dir, const char *dataset_name,
                       const char *index_name,
                       const size_t &n_sample,
                       const bool &load_sample_score, const bool &is_query_distribution,
                       const size_t &n_sample_query = 0, const size_t &sample_topk = 0) {
            char index_abs_dir[256];
            if (load_sample_score) {
                sprintf(index_abs_dir, "%s/memory_index", index_basic_dir);
            } else {
                sprintf(index_abs_dir, "%s/qrs_to_sample_index", index_basic_dir);
            }

            char index_path[512];
            if (is_query_distribution) {
                sprintf(index_path,
                        "%s/%s-%s-n_sample_%ld-n_sample_query_%ld-sample_topk_%ld.index",
                        index_abs_dir, index_name, dataset_name, n_sample, n_sample_query, sample_topk);
            } else {
                sprintf(index_path,
                        "%s/%s-%s-n_sample_%ld.index",
                        index_abs_dir, index_name, dataset_name, n_sample);
            }
            spdlog::info("index path {}", index_path);

            std::ifstream index_stream = std::ifstream(index_path, std::ios::binary | std::ios::in);
            if (!index_stream) {
                spdlog::error("error in reading index");
                exit(-1);
            }

            index_stream.read((char *) &n_sample_, sizeof(size_t));
            index_stream.read((char *) &n_data_item_, sizeof(size_t));
            index_stream.read((char *) &n_user_, sizeof(size_t));
            assert(n_sample_ == n_sample);

            known_rank_idx_l_ = std::make_unique<int[]>(n_sample_);
            index_stream.read((char *) known_rank_idx_l_.get(), (int64_t) (sizeof(int) * n_sample_));

            bound_distance_table_ = std::make_unique<float[]>(n_user_ * n_sample_);
            if (load_sample_score) {
                index_stream.read((char *) bound_distance_table_.get(),
                                  (int64_t) (sizeof(float) * n_user_ * n_sample_));
            }

            index_stream.close();
        }

        uint64_t IndexSizeByte() {
            const uint64_t known_rank_idx_size = sizeof(int) * n_sample_;
            const uint64_t bound_distance_table_size = sizeof(float) * n_user_ * n_sample_;
            return known_rank_idx_size + bound_distance_table_size;
        }

    };
}
#endif //REVERSE_KRANKS_SAMPLESEARCHUPDATE_HPP
