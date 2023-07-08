//
// Created by BianZheng on 2022/6/3.
//

#ifndef REVERSE_KRANKS_QUADRATICRANKBOUND_HPP
#define REVERSE_KRANKS_QUADRATICRANKBOUND_HPP

#include "score_computation/ComputeScoreTable.hpp"
#include "alg/DiskIndex/ReadAll.hpp"
#include "alg/RankBoundRefinement/PruneCandidateByBound.hpp"
#include "QuadraticRankSearch.hpp"
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

namespace ReverseMIPS::QuadraticRankBound {

    class Index : public BaseIndex {
        void ResetTimer() {
            inner_product_time_ = 0;
            coarse_binary_search_time_ = 0;
            read_disk_time_ = 0;
            fine_binary_search_time_ = 0;
            rank_prune_ratio_ = 0;
        }

        //rank search
        QuadraticRankSearch rank_ins_;
        //read disk
        ReadAll disk_ins_;

        VectorMatrix user_;
        int vec_dim_, n_data_item_, n_user_;
        double inner_product_time_, coarse_binary_search_time_, read_disk_time_, fine_binary_search_time_;
        TimeRecord inner_product_record_, coarse_binary_search_record_;
        double rank_prune_ratio_;
    public:

        //temporary retrieval variable
        // store queryIP
        std::vector<std::pair<double, double>> IPbound_l_;
        std::vector<double> queryIP_l_;
        std::vector<int> rank_lb_l_;
        std::vector<int> rank_ub_l_;
        std::vector<bool> prune_l_;

        Index(//rank search
                QuadraticRankSearch &rank_ins,
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
            IPbound_l_.resize(n_user_);
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
                IPbound_l_.assign(n_user_, std::pair<double, double>(-std::numeric_limits<double>::max(),
                                                                     std::numeric_limits<double>::max()));

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
                coarse_binary_search_record_.reset();
                rank_ins_.RankBound(queryIP_l_, topk, rank_lb_l_, rank_ub_l_, IPbound_l_, prune_l_, rank_topk_max_heap);

                PruneCandidateByBound(rank_lb_l_, rank_ub_l_,
                                      n_user_, topk,
                                      prune_l_, rank_topk_max_heap);
                coarse_binary_search_time_ += coarse_binary_search_record_.get_elapsed_time_second();

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
            fine_binary_search_time_ = disk_ins_.exact_rank_refinement_time_;

            rank_prune_ratio_ /= n_query_item;

            return query_heap_l;
        }

        std::string VariancePerformanceMetricName() override {
            return "queryID, retrieval time, second per query, rank prune ratio";
        }

        std::string VariancePerformanceStatistics(
                const double &retrieval_time, const double &second_per_query, const int &queryID) override {
            char str[256];
            sprintf(str, "%d,%.3f,%.3f,%.3f", queryID, retrieval_time, second_per_query, rank_prune_ratio_);
            std::string res(str);
            return res;
        };

        std::string
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &ms_per_query) override {
            // int topk;
            //double total_time,
            //          inner_product_time, coarse_binary_search_time, read_disk_time
            //          fine_binary_search_time;
            //double rank_prune_ratio;
            //double ms_per_query;
            //unit: second

            char buff[1024];

            sprintf(buff,
                    "top%d retrieval time: total %.3fs\n\tinner product %.3fs, coarse binary search %.3fs, read disk %.3fs\n\tfine binary search %.3fs\n\trank prune ratio %.4f\n\tmillion second per query %.3fms",
                    topk, retrieval_time,
                    inner_product_time_, coarse_binary_search_time_, read_disk_time_,
                    fine_binary_search_time_,
                    rank_prune_ratio_,
                    ms_per_query);
            std::string str(buff);
            return str;
        }

    };

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    std::unique_ptr<Index>
    BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const char *index_path, const int &n_sample) {
        const int n_user = user.n_vector_;
        const int n_data_item = data_item.n_vector_;
        const int vec_dim = data_item.vec_dim_;

        user.vectorNormalize();

        //rank search
        QuadraticRankSearch rank_ins(n_sample, n_data_item, n_user);
        //disk index
        ReadAll disk_ins(n_user, n_data_item, index_path, rank_ins.n_max_disk_read_);

        //Compute Score Table
        ComputeScoreTable cst(user, data_item);

        TimeRecord record;
        record.reset();
        std::vector<double> distance_l(n_data_item);
        for (int userID = 0; userID < n_user; userID++) {
            cst.ComputeSortItems(userID, distance_l.data());

            rank_ins.LoopPreprocess(distance_l.data(), userID);
            disk_ins.BuildIndexLoop(distance_l, 1);

            if (userID % cst.report_every_ == 0) {
                std::cout << "preprocessed " << userID / (0.01 * n_user) << " %, "
                          << record.get_elapsed_time_second() << " s/iter" << " Mem: "
                          << get_current_RSS() / 1000000 << " Mb \n";
                record.reset();
            }
        }
        cst.FinishCompute();

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(rank_ins, disk_ins, user, n_data_item);
        return index_ptr;
    }

}

#endif //REVERSE_KRANKS_QUADRATICRANKBOUND_HPP
