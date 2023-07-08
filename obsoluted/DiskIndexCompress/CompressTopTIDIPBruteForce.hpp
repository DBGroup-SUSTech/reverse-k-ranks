//
// Created by bianzheng on 2022/5/3.
//

#ifndef REVERSE_KRANKS_COMPRESSTOPTIDIPBRUTEFORCE_HPP
#define REVERSE_KRANKS_COMPRESSTOPTIDIPBRUTEFORCE_HPP

#include "TopTIDIP.hpp"
#include "alg/RankBoundRefinement/PruneCandidateByBound.hpp"
#include "../ScoreSample/ScoreSearch.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "score_computation/ComputeScoreTable.hpp"
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
#include "boost/sort/sort.hpp"

namespace ReverseMIPS::CompressTopTIDIPBruteForce {

    class Index : public BaseIndex {
        void ResetTimer() {
            inner_product_time_ = 0;
            hash_search_time_ = 0;
            read_disk_time_ = 0;
            exact_rank_time_ = 0;

            hash_prune_ratio_ = 0;
        }

    public:
        //for hash search, store in memory
        ScoreSearch rank_bound_ins_;
        //read all instance
        TopTIDIP disk_ins_;

        VectorMatrix user_, data_item_;
        int vec_dim_, n_data_item_, n_user_;
        double inner_product_time_, hash_search_time_, read_disk_time_, exact_rank_time_;
        TimeRecord inner_product_record_, hash_search_record_;
        double hash_prune_ratio_;

        //temporary retrieval variable
        std::vector<bool> prune_l_;
        std::vector<double> queryIP_l_;
        std::vector<int> rank_lb_l_;
        std::vector<int> rank_ub_l_;
        std::unique_ptr<double[]> query_cache_;

        Index(
                // hash search
                ScoreSearch &rank_bound_ins,
                //disk index
                TopTIDIP &disk_ins,
                //general retrieval
                VectorMatrix &user, VectorMatrix &data_item) {
            //rank search
            this->rank_bound_ins_ = std::move(rank_bound_ins);
            //read disk
            this->disk_ins_ = std::move(disk_ins);
            //general retrieval
            this->n_user_ = user.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->user_ = std::move(user);
            this->n_data_item_ = data_item.n_vector_;
            this->data_item_ = std::move(data_item);
            assert(0 < this->user_.vec_dim_);

            //retrieval variable
            this->prune_l_.resize(n_user_);
            this->queryIP_l_.resize(n_user_);
            this->rank_lb_l_.resize(n_user_);
            this->rank_ub_l_.resize(n_user_);
            this->query_cache_ = std::make_unique<double[]>(vec_dim_);

        }

        std::vector<std::vector<UserRankElement>> Retrieval(const VectorMatrix &query_item, const int &topk) override {
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

                const double *tmp_query_vecs = query_item.getVector(queryID);
                double *query_vecs = query_cache_.get();
                disk_ins_.PreprocessQuery(tmp_query_vecs, vec_dim_, query_vecs);

                //calculate the exact IP
                inner_product_record_.reset();
                for (int userID = 0; userID < n_user_; userID++) {
                    if (prune_l_[userID]) {
                        continue;
                    }
                    queryIP_l_[userID] = InnerProduct(user_.getVector(userID), query_vecs, vec_dim_);
                }
                this->inner_product_time_ += inner_product_record_.get_elapsed_time_second();

                //coarse binary search
                hash_search_record_.reset();
                rank_bound_ins_.RankBound(queryIP_l_, prune_l_, topk, rank_lb_l_, rank_ub_l_);
                PruneCandidateByBound(rank_lb_l_, rank_ub_l_,
                                      n_user_, topk,
                                      prune_l_, rank_topk_max_heap);
                hash_search_time_ += hash_search_record_.get_elapsed_time_second();
                int n_candidate = 0;
                for (int userID = 0; userID < n_user_; userID++) {
                    if (!prune_l_[userID]) {
                        n_candidate++;
                    }
                }
                assert(n_candidate >= topk);
                hash_prune_ratio_ += 1.0 * (n_user_ - n_candidate) / n_user_;

                //read disk and fine binary search
                disk_ins_.GetRank(queryIP_l_, rank_lb_l_, rank_ub_l_, prune_l_, user_, data_item_);

                for (int candID = 0; candID < topk; candID++) {
                    query_heap_l[queryID][candID] = disk_ins_.user_topk_cache_l_[candID];
                }
                assert(query_heap_l[queryID].size() == topk);
            }
            disk_ins_.FinishRetrieval();

            exact_rank_time_ = disk_ins_.exact_rank_time_;
            read_disk_time_ = disk_ins_.read_disk_time_;

            hash_prune_ratio_ /= n_query_item;
            return query_heap_l;
        }

        std::string
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &ms_per_query) override {
            // int topk;
            //double total_time,
            //          inner_product_time, hash_search_time_,
            //          read_disk_time_, exact_rank_time_,
            //          hash_prune_ratio_
            //double ms_per_query;
            //unit: second

            char buff[1024];
            sprintf(buff,
                    "top%d retrieval time:\n\ttotal %.3fs\n\tinner product %.3fs, hash search %.3fs\n\tread disk time %.3f, exact rank time %.3fs\n\thash prune ratio %.4f\n\tmillion second per query %.3fms",
                    topk, retrieval_time,
                    inner_product_time_, hash_search_time_,
                    read_disk_time_, exact_rank_time_,
                    hash_prune_ratio_,
                    ms_per_query);
            std::string str(buff);
            return str;
        }

        std::string BuildIndexStatistics() override {
            char buffer[512];
            double index_size_gb = 1.0 * n_user_ * disk_ins_.topt_ * sizeof(DistancePair) / (1024 * 1024 * 1024);
            sprintf(buffer, "Build Index Info: index size %.3f GB", index_size_gb);
            return buffer;
        };

    };

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    std::unique_ptr<Index> BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const char *index_path,
                                      const int &n_interval, const uint64_t &index_size_gb) {
        const int n_data_item = data_item.n_vector_;
        const int vec_dim = data_item.vec_dim_;
        const int n_user = user.n_vector_;

        user.vectorNormalize();

        //disk index
        const uint64_t index_size_byte = (uint64_t) index_size_gb * 1024 * 1024 * 1024;
        const uint64_t predict_index_size_byte = (uint64_t) (sizeof(double) + sizeof(int)) * n_data_item * n_user;
        const uint64_t topt_big_size = index_size_byte / (sizeof(double) + sizeof(int)) / n_user;
        int topt = int(topt_big_size);
        if (index_size_byte >= predict_index_size_byte) {
            spdlog::info("index size larger than the whole score table, use whole table setting");
            topt = n_data_item;
        }
        TopTIDIP disk_ins(n_user, n_data_item, vec_dim, index_path, topt);
        disk_ins.PreprocessData(user, data_item);

        //rank search
        ScoreSearch rank_bound_ins(n_interval, n_user, n_data_item);

        //Compute Score Table
        ComputeScoreTable cst(user, data_item);
        std::vector<DistancePair> distance_pair_l(n_data_item);

        TimeRecord record;
        record.reset();
        for (int userID = 0; userID < n_user; userID++) {
            cst.ComputeSortItems(userID, distance_pair_l.data());

            rank_bound_ins.LoopPreprocess(distance_pair_l.data(), userID);
            disk_ins.BuildIndexLoop(distance_pair_l.data(), 1);

            if (userID % cst.report_every_ == 0) {
                std::cout << "preprocessed " << userID / (0.01 * n_user) << " %, "
                          << record.get_elapsed_time_second() << " s/iter" << " Mem: "
                          << get_current_RSS() / 1000000 << " Mb \n";
                record.reset();
            }
        }
        cst.FinishCompute();

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(
                //score search
                rank_bound_ins,
                //disk index
                disk_ins,
                //general retrieval
                user, data_item);
        return index_ptr;
    }

}

#endif //REVERSE_KRANKS_COMPRESSTOPTIDIPBRUTEFORCE_HPP
