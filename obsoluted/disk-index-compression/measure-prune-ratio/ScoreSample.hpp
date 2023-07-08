//
// Created by BianZheng on 2022/5/19.
//

#ifndef REVERSE_K_RANKS_SCORESAMPLE_HPP
#define REVERSE_K_RANKS_SCORESAMPLE_HPP

#include "alg/DiskIndex/ReadAll.hpp"
#include "ScoreSearch.hpp"
#include "alg/RankBoundRefinement/RankSearch.hpp"
#include "alg/RankBoundRefinement/PruneCandidateByBound.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/UserRankElement.hpp"
#include "util/TimeMemory.hpp"
#include "util/VectorIO.hpp"
#include "util/FileIO.hpp"
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <set>
#include <cassert>
#include <spdlog/spdlog.h>

namespace ReverseMIPS::ScoreSample {

    class Index {
        void ResetTimer() {
            inner_product_time_ = 0;
            interval_search_time_ = 0;
            read_disk_time_ = 0;
            exact_rank_refinement_time_ = 0;
            interval_prune_ratio_ = 0;
        }

    public:
        //for interval search, store in memory
        ScoreSearch interval_ins_;

        VectorMatrix user_;
        int vec_dim_, n_data_item_, n_user_;
        double inner_product_time_, interval_search_time_, read_disk_time_, exact_rank_refinement_time_;
        TimeRecord inner_product_record_, interval_search_record_;
        double interval_prune_ratio_;

        //temporary retrieval variable
        std::vector<bool> prune_l_;
        std::vector<double> queryIP_l_;
        std::vector<int> rank_lb_l_;
        std::vector<int> rank_ub_l_;

        Index(
                //interval search
                ScoreSearch &interval_ins,
                //general retrieval
                VectorMatrix &user, const int &n_data_item) {
            //interval search
            this->interval_ins_ = std::move(interval_ins);
            //general retrieval
            this->n_user_ = user.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->user_ = std::move(user);
            this->n_data_item_ = n_data_item;
            assert(0 < this->user_.vec_dim_);

            //retrieval variable
            this->prune_l_.resize(n_user_);
            this->queryIP_l_.resize(n_user_);
            this->rank_lb_l_.resize(n_user_);
            this->rank_ub_l_.resize(n_user_);

        }

        void Retrieval(const VectorMatrix &query_item, const int &topk) {
            ResetTimer();

            if (topk > user_.n_vector_) {
                spdlog::error("top-k is too large, program exit");
                exit(-1);
            }

            const int n_query_item = query_item.n_vector_;
            std::vector<std::vector<UserRankElement>> query_heap_l(n_query_item);
            for (int qID = 0; qID < n_query_item; qID++) {
                query_heap_l[qID].resize(topk);
            }

            // store queryIP
            std::vector<int> rank_topk_max_heap(topk);
            for (int queryID = 0; queryID < n_query_item; queryID++) {
                prune_l_.assign(n_user_, false);
                rank_lb_l_.assign(n_user_, n_data_item_);
                rank_ub_l_.assign(n_user_, 0);

                const double *query_vecs = query_item.getVector(queryID);

                //calculate the exact IP
                inner_product_record_.reset();
                for (int userID = 0; userID < n_user_; userID++) {
                    if (prune_l_[userID]) {
                        continue;
                    }
                    queryIP_l_[userID] = InnerProduct(user_.getVector(userID), query_vecs, vec_dim_);
                }
                this->inner_product_time_ += inner_product_record_.get_elapsed_time_second();

                interval_search_record_.reset();
                //count rank bound
                interval_ins_.RankBound(queryIP_l_, prune_l_, topk, rank_lb_l_, rank_ub_l_);
                //prune the bound
                PruneCandidateByBound(rank_lb_l_, rank_ub_l_,
                                      n_user_, topk,
                                      prune_l_, rank_topk_max_heap);

                this->interval_search_time_ += interval_search_record_.get_elapsed_time_second();
                int n_candidate = 0;
                for (int userID = 0; userID < n_user_; userID++) {
                    if (!prune_l_[userID]) {
                        n_candidate++;
                    }
                }
                assert(n_candidate >= topk);
                interval_prune_ratio_ += 1.0 * (n_user_ - n_candidate) / n_user_;

            }
            interval_prune_ratio_ /= n_query_item;
        }

        std::string
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &ms_per_query) {
            // int topk;
            //double total_time,
            //          inner_product_time, interval_search_time_,
            //          read_disk_time_, exact_rank_refinement_time_,
            //          interval_prune_ratio_
            //double ms_per_query;
            //unit: second

            char buff[1024];
            sprintf(buff,
                    "top%d retrieval time:\n\ttotal %.3fs\n\tinner product %.3fs, interval search %.3fs, \n\tread disk time %.3f, exact rank refinement time %.3fs\n\tinterval prune ratio %.4f\n\tmillion second per query %.3fms",
                    topk, retrieval_time,
                    inner_product_time_, interval_search_time_,
                    read_disk_time_, exact_rank_refinement_time_,
                    interval_prune_ratio_,
                    ms_per_query);
            std::string str(buff);
            return str;
        }

    };

    const int write_every_ = 10;
    const int report_batch_every_ = 100;

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    std::unique_ptr<Index> BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const char *index_path,
                                      const int &n_interval) {
        const int n_data_item = data_item.n_vector_;
        const int vec_dim = data_item.vec_dim_;
        const int n_user = user.n_vector_;

        user.vectorNormalize();

        //interval search
        ScoreSearch interval_ins(n_interval, n_user, n_data_item);

        std::vector<double> write_distance_cache(write_every_ * n_data_item);
        const int n_batch = n_user / write_every_;
        const int n_remain = n_user % write_every_;
        spdlog::info("write_every_ {}, n_batch {}, n_remain {}", write_every_, n_batch, n_remain);

        TimeRecord batch_report_record;
        batch_report_record.reset();
        for (int i = 0; i < n_batch; i++) {
#pragma omp parallel for default(none) shared(i, data_item, user, write_distance_cache, interval_ins) shared(write_every_, n_data_item, vec_dim, n_interval)
            for (int cacheID = 0; cacheID < write_every_; cacheID++) {
                int userID = write_every_ * i + cacheID;
                for (int itemID = 0; itemID < n_data_item; itemID++) {
                    double ip = InnerProduct(data_item.getVector(itemID), user.getVector(userID), vec_dim);
                    write_distance_cache[cacheID * n_data_item + itemID] = ip;
                }
                std::sort(write_distance_cache.begin() + cacheID * n_data_item,
                          write_distance_cache.begin() + (cacheID + 1) * n_data_item, std::greater());

                //interval search
                const double *distance_ptr = write_distance_cache.data() + cacheID * n_data_item;
                interval_ins.LoopPreprocess(distance_ptr, userID);

            }

            if (i % report_batch_every_ == 0) {
                std::cout << "preprocessed " << i / (0.01 * n_batch) << " %, "
                          << batch_report_record.get_elapsed_time_second() << " s/iter" << " Mem: "
                          << get_current_RSS() / 1000000 << " Mb \n";
                batch_report_record.reset();
            }
        }

        {
            for (int cacheID = 0; cacheID < n_remain; cacheID++) {
                int userID = write_every_ * n_batch + cacheID;
                for (int itemID = 0; itemID < n_data_item; itemID++) {
                    double ip = InnerProduct(data_item.getVector(itemID), user.getVector(userID), vec_dim);
                    write_distance_cache[cacheID * n_data_item + itemID] = ip;
                }

                std::sort(write_distance_cache.begin() + cacheID * n_data_item,
                          write_distance_cache.begin() + (cacheID + 1) * n_data_item, std::greater<double>());

                //interval search
                const double *distance_ptr = write_distance_cache.data() + cacheID * n_data_item;
                interval_ins.LoopPreprocess(distance_ptr, userID);
            }
        }

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(
                //interval search
                interval_ins,
                //general retrieval
                user, n_data_item);
        return index_ptr;
    }

}
#endif //REVERSE_K_RANKS_SCORESAMPLE_HPP
