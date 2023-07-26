//
// Created by bianzheng on 2023/4/18.
//

#ifndef REVERSE_KRANKS_RTREEHEIGHT_HPP
#define REVERSE_KRANKS_RTREE_HPP

#include "struct/VectorMatrix.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/MethodBase.hpp"
#include "util/TimeMemory.hpp"
#include "util/VectorIO.hpp"

#include "rtree/const.h"
#include "rtree/dyy_RTK_method.hpp"
#include "rtree/dyy_data.hpp"

#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <cfloat>
#include <cassert>
#include <spdlog/spdlog.h>

namespace ReverseMIPS::Rtree {

    class Index : public BaseIndex {
        void ResetTimer() {
            total_retrieval_time_ = 0;
            total_single_query_retrieval_time_ = 0;
            total_ip_cost_ = 0;
            n_proc_query_ = 0;
        }

        dyy::Data rtree_ins_;

        VectorMatrix user_, data_item_;
        int vec_dim_, n_data_item_, n_user_;
        size_t stop_time_;
        double total_retrieval_time_, total_single_query_retrieval_time_;
        TimeRecord total_retrieval_record_;
        size_t total_ip_cost_;
        double n_proc_query_;
    public:

        //temporary retrieval variable
        std::vector<int> item_cand_l_;

        Index(
                //ip_bound_ins
                dyy::Data &&rtree_ins,
                //general retrieval
                VectorMatrix &data_item,
                VectorMatrix &user,
                const size_t &stop_time
        ) : rtree_ins_(std::move(rtree_ins)) {

            this->data_item_ = std::move(data_item);
            this->n_data_item_ = this->data_item_.n_vector_;
            this->user_ = std::move(user);
            this->vec_dim_ = this->user_.vec_dim_;
            this->n_user_ = this->user_.n_vector_;
            this->stop_time_ = stop_time;

            //retrieval variable
            item_cand_l_.resize(n_data_item_);
        }

        std::vector<std::vector<UserRankElement>>
        Retrieval(const VectorMatrix &query_item, const int &topk, const int &n_execute_query,
                  std::vector<SingleQueryPerformance> &query_performance_l) override {
            ResetTimer();

            if (n_execute_query > query_item.n_vector_) {
                spdlog::error("n_execute_query larger than n_query_item, program exit");
                exit(-1);
            }

            if (topk > user_.n_vector_) {
                spdlog::error("top-k is too large, program exit");
                exit(-1);
            }

            //coarse binary search
            const int n_query_item = n_execute_query;
            std::vector<std::vector<UserRankElement>> query_heap_l(n_query_item, std::vector<UserRankElement>(topk));
            total_retrieval_record_.reset();
            for (int queryID = 0; queryID < n_query_item; queryID++) {
                size_t ip_cost = 0;
                TimeRecord query_record;
                query_record.reset();
                float *query_vecs = query_item.getVector(queryID);
                dyy::Point q(query_vecs, vec_dim_, 0);
                auto rkr = dyy::RTK::rkrmethod(rtree_ins_.RtreeP, rtree_ins_.Weights, q, topk, ip_cost);

                int res_i = 0;
                while (!rkr.empty()) {
                    std::pair<int, int> p = rkr.top();
                    const int rank = p.first;
                    const int userID = p.second;
                    query_heap_l[queryID][res_i] = UserRankElement(userID, rank, -1);
                    res_i++;
                    rkr.pop();
                }
                std::reverse(query_heap_l[queryID].begin(), query_heap_l[queryID].end());
                assert(res_i == topk);

                const double single_query_retrieval_time = query_record.get_elapsed_time_second();
                total_single_query_retrieval_time_ += single_query_retrieval_time;
                total_ip_cost_ += ip_cost;

//                assert(n_proc_user == n_user_);
                n_proc_query_++;

                query_performance_l[queryID] = SingleQueryPerformance(queryID, 0, topk, 0,
                                                                      ip_cost, 0,
                                                                      single_query_retrieval_time, 0, 0);

                spdlog::info(
                        "finish queryID {}, single_query_retrieval_time {:.2f}s, ip_cost {}, n_proc_query {:.2f}",
                        queryID, single_query_retrieval_time, ip_cost, n_proc_query_);
                if (total_retrieval_record_.get_elapsed_time_second() > (double) stop_time_) {
                    spdlog::info("total retrieval time larger than stop time, retrieval exit");
                    break;
                }
            }
            total_retrieval_time_ = total_retrieval_record_.get_elapsed_time_second();

            return query_heap_l;
        }

        std::string
        PerformanceStatistics(const int &topk) override {
            // int topk;
            //double total_time,
            //          inner_product_time, inner_product_bound_time
            //double early_prune_ratio_
            //unit: second

            char buff[1024];

            sprintf(buff,
                    "top%d retrieval time: total %.3fs, n_proc_query %.3f\n\ttotal single query retrieval time %.3fs, total ip cost %ld",
                    topk, total_retrieval_time_, n_proc_query_,
                    total_single_query_retrieval_time_, total_ip_cost_);
            std::string str(buff);
            return str;
        }

        uint64_t IndexSizeByte() override {
            return 0;
        }


    };

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: float, the distance pair for each user
     */
    std::unique_ptr<Index>
    BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const size_t &stop_time) {
        user.vectorNormalize();
        assert(user.vec_dim_ == data_item.vec_dim_);

        int n_user = user.n_vector_;
        int n_data_item = data_item.n_vector_;
        int vec_dim = user.vec_dim_;

        dyy::Point::DIM = vec_dim;
        dyy::Mbr::DIM = vec_dim;
        dyy::RTreeNode::DIM = vec_dim;

        dyy::Data data;
        std::vector<dyy::Point> data_item_data(n_data_item);
        for (int itemID = 0; itemID < n_data_item; itemID++) {
            dyy::Point point(data_item.getVector(itemID), vec_dim, itemID);
            data_item_data[itemID] = point;
        }

        std::vector<dyy::Point> user_data(n_user);
        for (int userID = 0; userID < n_user; userID++) {
            dyy::Point point(user.getVector(userID), vec_dim, userID);
            user_data[userID] = point;
        }

        data.Products = data_item_data;
        data.Weights = user_data;

        dyy::Data::buildTree(data.Products, data.entriesP, &data.RtreeP);

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(std::move(data), data_item, user,
                                                                   stop_time);
        return index_ptr;
    }

}
#endif //REVERSE_KRANKS_RTREEHEIGHT_HPP
