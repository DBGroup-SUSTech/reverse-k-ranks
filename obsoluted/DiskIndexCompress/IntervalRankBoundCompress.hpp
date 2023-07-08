//
// Created by BianZheng on 2022/4/13.
//

#ifndef REVERSE_KRANKS_INTERVALRANKBOUNDCOMPRESS_HPP
#define REVERSE_KRANKS_INTERVALRANKBOUNDCOMPRESS_HPP

#include "RankBucket.hpp"
#include "alg/Prune/IPbound/FullIntPrune.hpp"
#include "alg/Prune/IntervalSearch.hpp"
#include "alg/Prune/RankSearch.hpp"
#include "alg/Prune/PruneCandidateByBound.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "alg/SVD.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/MethodBase.hpp"
#include "struct/DistancePair.hpp"
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

namespace ReverseMIPS::IntervalRankBoundCompress {

    class RetrievalResult : public RetrievalResultBase {
    public:
        //unit: second
        //double total_time, read_disk_time, inner_product_time,
        //          coarse_binary_search_time, fine_binary_search_time, interval_search_time;
        //double interval_prune_ratio, binary_search_prune_ratio;
        //double second_per_query;
        //int topk;

        inline RetrievalResult() = default;

        void AddPreprocess(double build_index_time) {
            char buff[1024];
            sprintf(buff, "build index time %.3f", build_index_time);
            std::string str(buff);
            this->config_l.emplace_back(str);
        }

        std::string AddResultConfig(const int &topk,
                                    const double &total_time, const double &interval_search_time,
                                    const double &inner_product_time,
                                    const double &coarse_binary_search_time, const double &read_disk_time,
                                    const double &fine_binary_search_time,
                                    const double &interval_prune_ratio,
                                    const double &rank_search_prune_ratio,
                                    const double &second_per_query) {
            char buff[1024];
            sprintf(buff,
                    "top%d retrieval time:\n\ttotal %.3fs, interval search %.3fs, inner product %.3fs\n\tcoarse binary search %.3fs, read disk time %.3f, fine binary search %.3fs\n\tinterval prune ratio %.4f, rank search prune ratio %.4f\n\tmillion second per query %.3fms",
                    topk,
                    total_time, interval_search_time, inner_product_time,
                    coarse_binary_search_time, read_disk_time, fine_binary_search_time,
                    interval_prune_ratio, rank_search_prune_ratio,
                    second_per_query);
            std::string str(buff);
            this->config_l.emplace_back(str);
            return str;
        }

    };

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
        FullIntPrune interval_prune_;

        //for rank search, store in memory
        RankSearch rank_ins_;
        //read all instance
        RankBucket disk_ins_;

        VectorMatrix user_, data_item_;
        int vec_dim_, n_data_item_, n_user_;
        double interval_search_time_, inner_product_time_, coarse_binary_search_time_, read_disk_time_, fine_binary_search_time_;
        TimeRecord interval_search_record_, inner_product_record_, coarse_binary_search_record_;
        double interval_prune_ratio_, rank_search_prune_ratio_;

        //temporary retrieval variable
        std::unique_ptr<double[]> query_ptr_;
        std::vector<bool> prune_l_;
        std::vector<std::pair<double, double>> ip_bound_l_;
        std::vector<double> queryIP_l_;
        std::vector<int> rank_lb_l_;
        std::vector<int> rank_ub_l_;

        Index(
                //interval search
                IntervalSearch &interval_ins,
                //interval search bound
                SVD &svd_ins, FullIntPrune &interval_prune,
                // rank search
                RankSearch &rank_ins,
                //disk index
                RankBucket &disk_ins,
                //general retrieval
                VectorMatrix &user, VectorMatrix &data_item) {
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
            this->n_data_item_ = data_item.n_vector_;
            this->data_item_ = std::move(data_item);
            assert(0 < this->user_.vec_dim_);

            //retrieval variable
            this->query_ptr_ = std::make_unique<double[]>(vec_dim_);
            this->prune_l_.resize(n_user_);
            this->ip_bound_l_.resize(n_user_);
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
                interval_prune_.IPBound(query_vecs, user_, prune_l_, ip_bound_l_);
                //count rank bound
                interval_ins_.RankBound(ip_bound_l_, prune_l_, topk, rank_lb_l_, rank_ub_l_);
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
                for (int userID = 0; userID < n_user_; userID++) {
                    if (prune_l_[userID]) {
                        continue;
                    }
                    queryIP_l_[userID] = InnerProduct(user_.getVector(userID), query_vecs, vec_dim_);
                }
                this->inner_product_time_ += inner_product_record_.get_elapsed_time_second();

                //coarse binary search
                coarse_binary_search_record_.reset();
                rank_ins_.RankBound(queryIP_l_, topk, rank_lb_l_, rank_ub_l_, prune_l_, rank_topk_max_heap);
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
                disk_ins_.GetRank(queryIP_l_, rank_lb_l_, rank_ub_l_, prune_l_, user_, data_item_, queryID);

                for (int candID = 0; candID < topk; candID++) {
                    query_heap_l[queryID][candID] = disk_ins_.user_topk_cache_l_[candID];
                }
                assert(query_heap_l[queryID].size() == topk);
            }
            disk_ins_.FinishRetrieval();

            fine_binary_search_time_ = disk_ins_.fine_binary_search_time_;
            read_disk_time_ = disk_ins_.read_disk_time_;

            interval_prune_ratio_ /= n_query_item;
            rank_search_prune_ratio_ /= n_query_item;
            return query_heap_l;
        }

    };

    const int report_batch_every = 100;

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    Index &BuildIndex(VectorMatrix &user, VectorMatrix &data_item, const char *index_path,
                      const int &n_merge_user, const int &compress_rank_every,
                      const int &cache_bound_every, const int &n_interval) {
        const int n_data_item = data_item.n_vector_;
        const int vec_dim = data_item.vec_dim_;
        const int n_user = user.n_vector_;

        user.vectorNormalize();

        const double SIGMA = 0.7;
        const double scale = 100;
        SVD svd_ins;
        int check_dim = svd_ins.Preprocess(user, data_item, SIGMA);

        FullIntPrune interval_prune;
        interval_prune.Preprocess(user, check_dim, scale);

        //interval search
        IntervalSearch interval_ins(n_interval, n_user, n_data_item);

        //rank search
        RankSearch rank_ins(cache_bound_every, n_data_item, n_user);

        //disk index
        RankBucket disk_ins(user, n_data_item, index_path, n_merge_user, compress_rank_every);
        std::vector<std::vector<int>> &eval_seq_l = disk_ins.BuildIndexMergeUser();
        assert(eval_seq_l.size() == n_merge_user);

        std::vector<std::set<int>> cache_bucket_vector(disk_ins.n_cache_rank_);

        std::vector<DistancePair> distance_pair_l(n_data_item);
        std::vector<double> IP_l(n_data_item);
        TimeRecord batch_report_record;
        batch_report_record.reset();
        for (int labelID = 0; labelID < n_merge_user; labelID++) {
            std::vector<int> &user_l = eval_seq_l[labelID];
            const unsigned int n_eval = user_l.size();

            for (int evalID = 0; evalID < n_eval; evalID++) {
                int userID = user_l[evalID];
#pragma omp parallel for default(none) shared(n_data_item, data_item, user, userID, vec_dim, distance_pair_l)
                for (int itemID = 0; itemID < n_data_item; itemID++) {
                    double ip = InnerProduct(data_item.getVector(itemID), user.getVector(userID), vec_dim);
                    distance_pair_l[itemID] = DistancePair(ip, itemID);
                }
                std::sort(distance_pair_l.begin(), distance_pair_l.end(), std::greater());

                for (int itemID = 0; itemID < n_data_item; itemID++) {
                    IP_l[itemID] = distance_pair_l[itemID].dist_;
                }

                //interval search
                double upper_bound = IP_l[0] + 0.01;
                double lower_bound = IP_l[n_data_item - 1] - 0.01;
                std::pair<double, double> bound_pair = std::make_pair(lower_bound, upper_bound);
                const double *distance_ptr = IP_l.data();
                interval_ins.LoopPreprocess(bound_pair, distance_ptr, userID);

                //rank search
                rank_ins.LoopPreprocess(distance_ptr, userID);

                disk_ins.BuildIndexLoop(distance_pair_l, userID, cache_bucket_vector);
            }
            disk_ins.WriteIndex(labelID, cache_bucket_vector);

            if (labelID % report_batch_every == 0) {
                std::cout << "preprocessed " << labelID / (0.01 * n_merge_user) << " %, "
                          << batch_report_record.get_elapsed_time_second() << " s/iter" << " Mem: "
                          << get_current_RSS() / 1000000 << " Mb \n";
                batch_report_record.reset();
            }
        }
        disk_ins.FinishWrite();

        static Index index(
                //interval search
                interval_ins,
                //interval search bound
                svd_ins, interval_prune,
                //rank search
                rank_ins,
                //disk index
                disk_ins,
                //general retrieval
                user, data_item);
        return index;
    }

}
#endif //REVERSE_KRANKS_INTERVALRANKBOUNDCOMPRESS_HPP
