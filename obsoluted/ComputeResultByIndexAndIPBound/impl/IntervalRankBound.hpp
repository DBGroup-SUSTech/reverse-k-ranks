//
// Created by BianZheng on 2022/3/17.
//

#ifndef REVERSE_KRANKS_INTERVALRANKBOUND_HPP
#define REVERSE_KRANKS_INTERVALRANKBOUND_HPP

#include "alg/DiskIndex/ReadAll.hpp"
#include "alg/Prune/IPbound/PartDimPartIntPrune.hpp"
#include "alg/Prune/IntervalSearch.hpp"
#include "alg/Prune/RankSearch.hpp"
#include "alg/Prune/PruneCandidateByBound.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "alg/SVD.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/MethodBase.hpp"
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

namespace ReverseMIPS::IntervalRankBound {

    class Index : public BaseIndex {
        void ResetTimer() {
            read_disk_time_ = 0;
            inner_product_time_ = 0;
            coarse_binary_search_time_ = 0;
            fine_binary_search_time_ = 0;
            interval_search_time_ = 0;
            interval_prune_ratio_ = 0;
            rank_search_prune_ratio_ = 0;
        }

    public:
        //for interval search, store in memory
        IntervalSearch interval_ins_;
        //interval search bound
        SVD svd_ins_;
        PartDimPartIntPrune interval_prune_;

        //for rank search, store in memory
        RankSearch rank_ins_;
        //read all instance
        ReadAll disk_ins_;

        VectorMatrix user_;
        int vec_dim_, n_data_item_, n_user_;
        double interval_search_time_, inner_product_time_, coarse_binary_search_time_, read_disk_time_, fine_binary_search_time_;
        TimeRecord interval_search_record_, inner_product_record_, coarse_binary_search_record_;
        double interval_prune_ratio_, rank_search_prune_ratio_;

        //temporary retrieval variable
        std::unique_ptr<double[]> query_ptr_;
        std::vector<bool> prune_l_;
        std::vector<std::pair<double, double>> IPbound_l_;
        std::vector<double> queryIP_l_;
        std::vector<int> rank_lb_l_;
        std::vector<int> rank_ub_l_;

        Index(
                //interval search
                IntervalSearch &interval_ins,
                //interval search bound
                SVD &svd_ins, PartDimPartIntPrune &interval_prune,
                // rank search
                RankSearch &rank_ins,
                //disk index
                ReadAll &disk_ins,
                //general retrieval
                VectorMatrix &user, const int &n_data_item) {
            //interval search
            this->interval_ins_ = std::move(interval_ins);
            //interval search bound
            this->svd_ins_ = std::move(svd_ins);
            this->interval_prune_ = std::move(interval_prune);
            //rank search
            this->rank_ins_ = std::move(rank_ins);
            //read disk
            this->disk_ins_ = std::move(disk_ins);
            //general retrieval
            this->n_user_ = user.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->user_ = std::move(user);
            this->n_data_item_ = n_data_item;
            assert(0 < this->user_.vec_dim_);

            //retrieval variable
            this->query_ptr_ = std::make_unique<double[]>(vec_dim_);
            this->prune_l_.resize(n_user_);
            this->IPbound_l_.resize(n_user_);
            this->queryIP_l_.resize(n_user_);
            this->rank_lb_l_.resize(n_user_);
            this->rank_ub_l_.resize(n_user_);

        }

        std::vector<std::vector<UserRankElement>> Retrieval(VectorMatrix &query_item, const int &topk) override {
            ResetTimer();
            disk_ins_.RetrievalPreprocess();

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

                double *query_vecs = query_ptr_.get();
                svd_ins_.TransferQuery(query_item.getVector(queryID), vec_dim_, query_vecs);

                interval_search_record_.reset();
                //get the ip bound
                interval_prune_.IPBound(query_vecs, user_, prune_l_, IPbound_l_, queryIP_l_.data());
                //count rank bound
                interval_ins_.RankBound(IPbound_l_, prune_l_, topk, rank_lb_l_, rank_ub_l_);
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

                //calculate the exact IP
                inner_product_record_.reset();
                const int check_dim = interval_prune_.check_dim_;
                const int remain_dim = interval_prune_.remain_dim_;
                const double *query_remain_vecs = query_vecs + check_dim;
                for (int userID = 0; userID < n_user_; userID++) {
                    if (prune_l_[userID]) {
                        continue;
                    }
                    const double *user_vecs = user_.getVector(userID) + check_dim;
                    double prev_left = queryIP_l_[userID];
                    double prev_right = InnerProduct(user_vecs, query_remain_vecs, remain_dim);
                    queryIP_l_[userID] = prev_left + prev_right;
//                    queryIP_l_[userID] = InnerProduct(user_.getVector(userID), query_vecs, vec_dim_);
                }
                this->inner_product_time_ += inner_product_record_.get_elapsed_time_second();

                //coarse binary search
                coarse_binary_search_record_.reset();
                rank_ins_.RankBound(queryIP_l_, topk, rank_lb_l_, rank_ub_l_, IPbound_l_, prune_l_, rank_topk_max_heap);
                PruneCandidateByBound(rank_lb_l_, rank_ub_l_,
                                      n_user_, topk,
                                      prune_l_, rank_topk_max_heap);
                coarse_binary_search_time_ += coarse_binary_search_record_.get_elapsed_time_second();
                n_candidate = 0;
                for (int userID = 0; userID < n_user_; userID++) {
                    if (!prune_l_[userID]) {
                        n_candidate++;
                    }
                }
                assert(n_candidate >= topk);
                rank_search_prune_ratio_ += 1.0 * (n_user_ - n_candidate) / n_user_;

                //read disk and fine binary search
                disk_ins_.GetRank(queryIP_l_, rank_lb_l_, rank_ub_l_, prune_l_);

                for (int candID = 0; candID < topk; candID++) {
                    query_heap_l[queryID][candID] = disk_ins_.user_topk_cache_l_[candID];
                }
                assert(query_heap_l[queryID].size() == topk);
            }
            disk_ins_.FinishRetrieval();

            fine_binary_search_time_ = disk_ins_.exact_rank_refinement_time_;
            read_disk_time_ = disk_ins_.read_disk_time_;

            interval_prune_ratio_ /= n_query_item;
            rank_search_prune_ratio_ /= n_query_item;
            return query_heap_l;
        }

        std::string
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &second_per_query) override {
            // int topk;
            //double total_time,
            //          interval_search_time_, inner_product_time, read_disk_time
            //          coarse_binary_search_time_, read_disk_time_, exact_rank_refinement_time_,
            //          interval_prune_ratio_, rank_search_prune_ratio_
            //double second_per_query;
            //unit: second

            char buff[1024];
            sprintf(buff,
                    "top%d retrieval time:\n\ttotal %.3fs\n\tinterval search %.3fs, inner product %.3fs, coarse binary search %.3fs\n\tread disk time %.3f, fine binary search %.3fs\n\tinterval prune ratio %.4f, rank search prune ratio %.4f\n\tmillion second per query %.3fms",
                    topk, retrieval_time,
                    interval_search_time_, inner_product_time_, coarse_binary_search_time_,
                    read_disk_time_, fine_binary_search_time_,
                    interval_prune_ratio_, rank_search_prune_ratio_,
                    second_per_query);
            std::string str(buff);
            return str;
        }

    };

    const int write_every_ = 1000;
    const int report_batch_every_ = 100;

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    std::unique_ptr<Index> BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const char *index_path,
                                      const int &cache_bound_every, const int &n_interval) {
        const int n_data_item = data_item.n_vector_;
        const int vec_dim = data_item.vec_dim_;
        const int n_user = user.n_vector_;

        user.vectorNormalize();

        const double SIGMA = 0.7;
        const double scale = 100;
        SVD svd_ins;
        int check_dim = svd_ins.Preprocess(user, data_item, SIGMA);

        PartDimPartIntPrune interval_prune;
        interval_prune.Preprocess(user, check_dim, scale);

        //interval search
        IntervalSearch interval_ins(n_interval, n_user, n_data_item);

        //rank search
        RankSearch rank_ins(cache_bound_every, n_data_item, n_user);

        //disk index
        ReadAll disk_ins(n_user, n_data_item, index_path, rank_ins.n_max_disk_read_);

        std::vector<double> write_distance_cache(write_every_ * n_data_item);
        const int n_batch = n_user / write_every_;
        const int n_remain = n_user % write_every_;
        spdlog::info("write_every_ {}, n_batch {}, n_remain {}", write_every_, n_batch, n_remain);

        TimeRecord batch_report_record;
        batch_report_record.reset();
        for (int i = 0; i < n_batch; i++) {
#pragma omp parallel for default(none) shared(i, data_item, user, write_distance_cache, rank_ins, check_dim, interval_ins) shared(write_every_, n_data_item, vec_dim, n_interval)
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

                //rank search
                rank_ins.LoopPreprocess(distance_ptr, userID);
            }
            disk_ins.BuildIndexLoop(write_distance_cache, write_every_);

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

                //rank search
                rank_ins.LoopPreprocess(distance_ptr, userID);
            }
            disk_ins.BuildIndexLoop(write_distance_cache, n_remain);
        }

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(
                //interval search
                interval_ins,
                //interval search bound
                svd_ins, interval_prune,
                //rank search
                rank_ins,
                //disk index
                disk_ins,
                //general retrieval
                user, n_data_item);
        return index_ptr;
    }

}
#endif //REVERSE_KRANKS_INTERVALRANKBOUND_HPP
