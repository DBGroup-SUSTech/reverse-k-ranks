//
// Created by bianzheng on 2023/5/29.
//

#ifndef REVERSE_KRANKS_SIMPFERPPCACHE_HPP
#define REVERSE_KRANKS_SIMPFERPPCACHE_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "alg/RankBoundRefinement/PruneCandidateByBound.hpp"

#include "simpfer_plus_plus/SimpferPlusPlusData.hpp"
#include "simpfer_plus_plus/SimpferPlusPlusBuildIndex.hpp"
#include "simpfer_plus_plus/SimpferPlusPlusRetrieval.hpp"

#include "fexipro/alg/SVDIntUpperBoundIncrPrune2.h"

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
#include <filesystem>

namespace ReverseMIPS::SimpferPPCache {

    class Index : public BaseIndex {
        void ResetTimer() {
            total_retrieval_time_ = 0;
            total_ip_cost_ = 0;
            n_proc_query_ = 0;
        }

    public:
        SimpferPlusPlusCacheIndex simpfer_index_;

        VectorMatrix user_, data_item_;
        int vec_dim_, n_data_item_, n_user_;
        double total_retrieval_time_;
        TimeRecord total_retrieval_record_;
        size_t total_ip_cost_;
        int n_proc_query_;

        Index(
                SimpferPlusPlusCacheIndex &&simpfer_index,
                //general retrieval
                VectorMatrix &user, VectorMatrix &data_item)
                : simpfer_index_(std::move(simpfer_index)) {
            //index for reverseMIPS
            //general retrieval
            this->n_user_ = user.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->user_ = std::move(user);
            this->n_data_item_ = data_item.n_vector_;
            this->data_item_ = std::move(data_item);
            assert(0 < this->user_.vec_dim_);

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

            const int n_query_item = n_execute_query;
            std::vector<std::vector<UserRankElement>> query_heap_l(n_query_item);

            std::vector<SimpferPlusPlusData> query_sd_l;
            TransformData(query_item, query_sd_l);

            // store queryIP
            for (int queryID = 0; queryID < n_query_item; queryID++) {
                total_retrieval_record_.reset();

                int rtk_topk = 1;
                std::vector<int> result_userID_l;
                size_t this_query_ip_count = 0;
                int result_size = 0;

                while (rtk_topk <= n_data_item_) {
                    size_t ip_count = 0;
                    simpfer_index_.RTopKRetrieval(query_sd_l[queryID], user_, rtk_topk, queryID,
                                                  result_userID_l,
                                                  ip_count, result_size);
                    query_sd_l[queryID].init();
                    const double accu_retrieval_time = total_retrieval_record_.get_elapsed_time_second();
                    this_query_ip_count += ip_count;
                    if (result_size > 0) {
                        spdlog::info(
                                "queryID {}, result_size {}, rtk_topk {}, accu_ip_cost {}, accu_query_time {:.2f}s,",
                                queryID, result_size, rtk_topk, this_query_ip_count, accu_retrieval_time);
                    }

                    if (result_size >= topk) {
                        break;
                    } else {
                        if (rtk_topk == n_data_item_) {
                            break;
                        } else if (rtk_topk < n_data_item_ && rtk_topk * 2 >= n_data_item_) {
                            rtk_topk = n_data_item_;
                        } else {
                            rtk_topk *= 2;
                        }
                    }

                }

                const double query_time = total_retrieval_record_.get_elapsed_time_second();
                total_retrieval_time_ += query_time;
                total_ip_cost_ += this_query_ip_count;
                n_proc_query_++;

                spdlog::info(
                        "queryID {}, result_size {}, rtk_topk {}, ip_cost {}, query_time {:.2f}s, total_ip_cost {}, total_retrieval_time {:.2f}s",
                        queryID, result_size, rtk_topk, this_query_ip_count, query_time, total_ip_cost_, total_retrieval_time_);
                assert(result_userID_l.size() == result_size);

                query_performance_l[queryID] = SingleQueryPerformance(queryID, 0, topk, 0,
                                                                      this_query_ip_count, 0,
                                                                      query_time, 0, 0);

                for (int resultID = 0; resultID < result_size; resultID++) {
                    const int userID = result_userID_l[resultID];
                    query_heap_l[queryID].emplace_back(userID);
                }

            }

            simpfer_index_.FinishCompute();

            return query_heap_l;
        }

        std::string
        PerformanceStatistics(const int &topk) override {
            // int topk;
            //size_t total_ip_cost;
            //double total_time,
            //unit: second

            char buff[1024];
            sprintf(buff,
                    "top%d retrieval:\n\ttotal IP cost %ld, total time %.3fs, n_proc_query %d",
                    topk, total_ip_cost_, total_retrieval_time_, n_proc_query_);
            std::string str(buff);
            return str;
        }

        uint64_t IndexSizeByte() override {
            return simpfer_index_.IndexSizeByte();
        }

    };

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: float, the distance pair for each user
     */

    std::unique_ptr<Index>
    BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const char *index_path, const char *dataset_name,
               const int &simpfer_k_max) {
        const int n_data_item = data_item.n_vector_;
        const int vec_dim = data_item.vec_dim_;
        const int n_user = user.n_vector_;

        std::vector<SimpferPlusPlusData> user_sd_l;
        TransformData(user, user_sd_l);
        spdlog::info("finish transform user");

        std::vector<SimpferPlusPlusData> data_item_sd_l;
        TransformData(data_item, data_item_sd_l);
        spdlog::info("finish transform data item");

        std::vector<SimpferPlusPlusBlock> block_l;
        pre_processing(user_sd_l, data_item_sd_l, block_l, n_user, simpfer_k_max, vec_dim);
        spdlog::info("finish simpfer preprocessing");

        char index_filename[256];
        sprintf(index_filename, "%s/rmips_index/%s.index", index_path, dataset_name);

        SimpferPlusPlusCacheIndex simpfer_index(user_sd_l, data_item_sd_l, block_l,
                                                simpfer_k_max, index_filename);

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(
                std::move(simpfer_index),
                //general retrieval
                user, data_item);
        return index_ptr;
    }

}
#endif //REVERSE_KRANKS_SIMPFERPPCACHE_HPP
