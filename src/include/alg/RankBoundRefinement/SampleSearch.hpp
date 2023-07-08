//
// Created by BianZheng on 2022/11/3.
//

#ifndef REVERSE_K_RANKS_SAMPLESEARCH_HPP
#define REVERSE_K_RANKS_SAMPLESEARCH_HPP

#include "struct/DistancePair.hpp"
#include <iostream>
#include <fstream>
#include <memory>
#include <cfloat>
#include <omp.h>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class SampleSearch {

        size_t n_sample_, n_data_item_, n_user_;
        int n_thread_;
        std::unique_ptr<float[]> bound_distance_table_; // n_user * n_sample_
    public:
        std::unique_ptr<int[]> known_rank_idx_l_; // n_sample_

        inline SampleSearch() {}

        inline SampleSearch(const int &n_data_item, const int &n_user,
                            const std::vector<int> &known_rank_l, const int &n_sample) {
            this->n_sample_ = n_sample;
            this->n_data_item_ = n_data_item;
            this->n_user_ = n_user;
            known_rank_idx_l_ = std::make_unique<int[]>(n_sample_);
            bound_distance_table_ = std::make_unique<float[]>(n_user_ * n_sample_);
            if (n_sample <= 0 || n_sample >= n_data_item) {
                spdlog::error("n_sample too small or too large, program exit");
                exit(-1);
            }
            assert(n_sample > 0);
            assert(known_rank_l.size() == n_sample);

            for (int sampleID = 0; sampleID < n_sample; sampleID++) {
                known_rank_idx_l_[sampleID] = known_rank_l[sampleID];
            }

        }

        inline SampleSearch(const char *index_path, const char *dataset_name, const char *sample_search_index_name,
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
        }

        void LoopPreprocess(const DistancePair *distance_ptr, const int &userID) {
            for (int crankID = 0; crankID < n_sample_; crankID++) {
                unsigned int rankID = known_rank_idx_l_[crankID];
                bound_distance_table_[n_sample_ * userID + crankID] = distance_ptr[rankID].dist_;
            }
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

                rank_lb_l[userID] = lower_rank;
                rank_ub_l[userID] = upper_rank;
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
#endif //REVERSE_K_RANKS_SAMPLESEARCH_HPP
