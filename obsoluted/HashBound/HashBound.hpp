//
// Created by BianZheng on 2022/5/5.
//

#ifndef REVERSE_KRANKS_HASHRANKBOUND_HPP
#define REVERSE_KRANKS_HASHRANKBOUND_HPP

#include "alg/DiskIndex/ReadAll.hpp"
#include "alg/RankBoundRefinement/PruneCandidateByBound.hpp"
#include "HashSearch.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/MethodBase.hpp"
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

namespace ReverseMIPS::HashBound {

    class Index : public BaseIndex {
        void ResetTimer() {
            inner_product_time_ = 0;
            rank_bound_prune_time_ = 0;
            read_disk_time_ = 0;
            exact_rank_refinement_time_ = 0;
            rank_prune_ratio_ = 0;
        }

        //rank search
        HashSearch rank_ins_;
        //read disk
        ReadAll disk_ins_;

        VectorMatrix user_;
        int vec_dim_, n_data_item_, n_user_;
        double inner_product_time_, rank_bound_prune_time_, read_disk_time_, exact_rank_refinement_time_;
        TimeRecord inner_product_record_, rank_bound_prune_record_;
        double rank_prune_ratio_;
    public:

        //temporary retrieval variable
        std::vector<double> queryIP_l_;
        std::vector<int> rank_lb_l_;
        std::vector<int> rank_ub_l_;
        std::vector<bool> prune_l_;

        Index(//rank search
                HashSearch &rank_ins,
                //disk index
                ReadAll &disk_ins,
                //general retrieval
                VectorMatrix &user, const int &n_data_item
        ) {
            //rank search
            this->rank_ins_ = std::move(rank_ins);
            //read disk
            this->disk_ins_ = std::move(disk_ins);

            this->user_ = std::move(user);
            this->vec_dim_ = this->user_.vec_dim_;
            this->n_user_ = this->user_.n_vector_;
            this->n_data_item_ = n_data_item;

            //retrieval variable
            queryIP_l_.resize(n_user_);
            rank_lb_l_.resize(n_user_);
            rank_ub_l_.resize(n_user_);
            prune_l_.resize(n_user_);
        }

        std::vector<std::vector<UserRankElement>> Retrieval(const VectorMatrix &query_item, const int &topk) override {
            ResetTimer();
            disk_ins_.RetrievalPreprocess();

            if (topk > user_.n_vector_) {
                spdlog::error("top-k is too large, program exit");
                exit(-1);
            }

            //coarse binary search
            const int n_query_item = query_item.n_vector_;

            std::vector<std::vector<UserRankElement>> query_heap_l(n_query_item);
            for (int qID = 0; qID < n_query_item; qID++) {
                query_heap_l[qID].reserve(topk);
            }

            // for binary search, check the number
            std::vector<int> rank_topk_max_heap(topk);
            for (int queryID = 0; queryID < n_query_item; queryID++) {
                prune_l_.assign(n_user_, false);
                rank_lb_l_.assign(n_user_, n_data_item_);
                rank_ub_l_.assign(n_user_, 0);

                //calculate IP
                double *query_item_vec = query_item.getVector(queryID);
                inner_product_record_.reset();
                for (int userID = 0; userID < n_user_; userID++) {
                    double *user_vec = user_.getVector(userID);
                    double queryIP = InnerProduct(query_item_vec, user_vec, vec_dim_);
                    queryIP_l_[userID] = queryIP;
                }
                this->inner_product_time_ += inner_product_record_.get_elapsed_time_second();

                //rank search
                rank_bound_prune_record_.reset();
                rank_ins_.RankBound(queryIP_l_, prune_l_, rank_lb_l_, rank_ub_l_);

                PruneCandidateByBound(rank_lb_l_, rank_ub_l_,
                                      n_user_, topk,
                                      prune_l_, rank_topk_max_heap);
                rank_bound_prune_time_ += rank_bound_prune_record_.get_elapsed_time_second();

                int n_candidate = 0;
                for (int userID = 0; userID < n_user_; userID++) {
                    if (!prune_l_[userID]) {
                        n_candidate++;
                    }
                }
                assert(n_candidate >= topk);
                rank_prune_ratio_ += 1.0 * (n_user_ - n_candidate) / n_user_;

                //read disk and fine binary search
                disk_ins_.GetRank(queryIP_l_, rank_lb_l_, rank_ub_l_, prune_l_);

                for (int candID = 0; candID < topk; candID++) {
                    query_heap_l[queryID].emplace_back(disk_ins_.user_topk_cache_l_[candID]);
                }
                assert(query_heap_l[queryID].size() == topk);
            }
            disk_ins_.FinishRetrieval();

            read_disk_time_ = disk_ins_.read_disk_time_;
            exact_rank_refinement_time_ = disk_ins_.exact_rank_refinement_time_;

            rank_prune_ratio_ /= n_query_item;

            return query_heap_l;
        }

        std::string
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &ms_per_query) override {
            // int topk;
            //double total_time,
            //          inner_product_time, rank bound prune, read_disk_time
            //          exact rank refinement;
            //double rank_prune_ratio;
            //double ms_per_query;
            //unit: second

            char buff[1024];

            sprintf(buff,
                    "top%d retrieval time: total %.3fs\n\tinner product %.3fs, rank bound prune %.3fs, read disk %.3fs\n\texact rank refinement %.3fs\n\trank prune ratio %.4f\n\tmillion second per query %.3fms",
                    topk, retrieval_time,
                    inner_product_time_, rank_bound_prune_time_, read_disk_time_,
                    exact_rank_refinement_time_,
                    rank_prune_ratio_,
                    ms_per_query);
            std::string str(buff);
            return str;
        }

    };

    const int write_every_ = 100;
    const int report_batch_every_ = 100;

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    std::unique_ptr<Index>
    BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const char *index_path,
               const int &cache_bound_every, const int &n_interval) {
        const int n_user = user.n_vector_;
        const int n_data_item = data_item.n_vector_;
        std::vector<double> write_distance_cache(write_every_ * n_data_item);
        const int vec_dim = data_item.vec_dim_;
        const int n_batch = user.n_vector_ / write_every_;
        const int n_remain = user.n_vector_ % write_every_;

        user.vectorNormalize();

        //rank search
        HashSearch rank_ins(n_data_item, n_user, cache_bound_every, n_interval);
        //disk index
        ReadAll disk_ins(n_user, n_data_item, index_path, rank_ins.n_max_disk_read_);

        TimeRecord batch_report_record;
        batch_report_record.reset();
        for (int i = 0; i < n_batch; i++) {
#pragma omp parallel for default(none) shared(i, data_item, user, write_distance_cache, rank_ins) shared(write_every_, n_data_item, vec_dim)
            for (int cacheID = 0; cacheID < write_every_; cacheID++) {
                int userID = write_every_ * i + cacheID;
                for (int itemID = 0; itemID < n_data_item; itemID++) {
                    double ip = InnerProduct(data_item.getVector(itemID), user.getVector(userID), vec_dim);
                    write_distance_cache[cacheID * n_data_item + itemID] = ip;
                }
                std::sort(write_distance_cache.begin() + cacheID * n_data_item,
                          write_distance_cache.begin() + (cacheID + 1) * n_data_item, std::greater());

                //rank search
                const double *distance_ptr = write_distance_cache.data() + cacheID * n_data_item;
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
                for (int itemID = 0; itemID < data_item.n_vector_; itemID++) {
                    double ip = InnerProduct(data_item.getRawData() + itemID * vec_dim,
                                             user.getRawData() + userID * vec_dim, vec_dim);
                    write_distance_cache[cacheID * data_item.n_vector_ + itemID] = ip;
                }

                std::sort(write_distance_cache.begin() + cacheID * n_data_item,
                          write_distance_cache.begin() + (cacheID + 1) * n_data_item, std::greater());

                //rank search
                const double *distance_ptr = write_distance_cache.data() + cacheID * n_data_item;
                rank_ins.LoopPreprocess(distance_ptr, userID);
            }

            disk_ins.BuildIndexLoop(write_distance_cache, n_remain);
        }

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(rank_ins, disk_ins, user, n_data_item);
        return index_ptr;
    }

}

#endif //REVERSE_KRANKS_HASHRANKBOUND_HPP
