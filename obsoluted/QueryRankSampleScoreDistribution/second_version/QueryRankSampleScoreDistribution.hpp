//
// Created by BianZheng on 2022/10/17.
//

#ifndef REVERSE_K_RANKS_QUERYRANKSAMPLESCOREDISTRIBUTION_HPP
#define REVERSE_K_RANKS_QUERYRANKSAMPLESCOREDISTRIBUTION_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "alg/TopkMaxHeap.hpp"
#include "alg/DiskIndex/ReadAll.hpp"
#include "alg/DiskIndex/ReadAllDirectIO.hpp"
#include "BitIndex.hpp"
#include "alg/RankBoundRefinement/PruneCandidateByBound.hpp"
#include "alg/RankBoundRefinement/SampleSearch.hpp"

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

namespace ReverseMIPS::QueryRankSampleScoreDistribution {

    class Index : public BaseIndex {
        void ResetTimer() {
            total_retrieval_time_ = 0;
            inner_product_time_ = 0;
            rank_bound_time_ = 0;
            prune_user_time_ = 0;
            read_disk_time_ = 0;
            compute_rank_time_ = 0;
            rank_prune_ratio_ = 0;
            total_ip_cost_ = 0;
            total_io_cost_ = 0;
            sample_refine_user_ = 0;
            total_refine_user_ = 0;
        }

        //rank search
        SampleSearch rank_ins_;
        //rank search
        BitIndex bit_index_;
        //read disk
        ReadAllDirectIO disk_ins_;

        static constexpr int batch_n_user_ = 1024;
        int n_batch_;

        VectorMatrix user_;
        int vec_dim_, n_data_item_, n_user_;
        double total_retrieval_time_, inner_product_time_, rank_bound_time_, prune_user_time_, read_disk_time_, compute_rank_time_;
        TimeRecord total_retrieval_record_, inner_product_record_, rank_bound_record_, prune_user_record_;
        uint64_t total_ip_cost_, total_io_cost_, sample_refine_user_, total_refine_user_;
        double rank_prune_ratio_;

    public:

        //temporary retrieval variable
        std::vector<bool> prune_l_;
        std::vector<bool> result_l_;
        std::vector<int> refine_seq_l_;
        std::vector<float> queryIP_l_;
        std::vector<std::pair<float, float>> IPbound_l_;
        std::vector<int> rank_lb_l_;
        std::vector<int> rank_ub_l_;
        std::vector<int> bucketID_l_;

        Index(//rank search
                SampleSearch &rank_ins,
                BitIndex &bit_index,
                //disk index
                ReadAllDirectIO &disk_ins,
                //general retrieval
                VectorMatrix &user, const int &n_data_item
        ) {
            //rank search
            this->rank_ins_ = std::move(rank_ins);
            this->bit_index_ = std::move(bit_index);
            //read disk
            this->disk_ins_ = std::move(disk_ins);

            this->user_ = std::move(user);
            this->vec_dim_ = this->user_.vec_dim_;
            this->n_user_ = this->user_.n_vector_;
            this->n_data_item_ = n_data_item;
            this->n_batch_ = n_user_ / batch_n_user_ + (n_user_ % batch_n_user_ == 0 ? 0 : 1);

            //retrieval variable
            this->prune_l_.resize(n_user_);
            this->result_l_.resize(n_user_);
            this->refine_seq_l_.resize(n_user_);
            this->queryIP_l_.resize(n_user_);
            this->IPbound_l_.resize(n_user_);
            this->rank_lb_l_.resize(n_user_);
            this->rank_ub_l_.resize(n_user_);
            this->bucketID_l_.resize(n_user_);
        }

        std::vector<std::vector<UserRankElement>>
        Retrieval(const VectorMatrix &query_item, const int &topk, const int &n_execute_query,
                  std::vector<SingleQueryPerformance> &query_performance_l) override {
            ResetTimer();
            disk_ins_.RetrievalPreprocess();

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

            // for binary search, check the number
            for (int queryID = 0; queryID < n_query_item; queryID++) {
                total_retrieval_record_.reset();
                prune_l_.assign(n_user_, false);
                result_l_.assign(n_user_, false);

                const float *query_vecs = query_item.getVector(queryID);

                //calculate IP
                inner_product_record_.reset();
#pragma omp parallel for default(none) shared(queryIP_l_, query_vecs) num_threads(omp_get_num_procs())
                for (int batchID = 0; batchID < n_batch_; batchID++) {
                    const int start_userID = batchID * batch_n_user_;
                    const int end_userID = std::min(n_user_, (batchID + 1) * batch_n_user_);
                    for (int userID = start_userID; userID < end_userID; userID++) {
                        queryIP_l_[userID] = InnerProduct(query_vecs, user_.getVector(userID), vec_dim_);
                    }
                }
                const double tmp_inner_product_time = inner_product_record_.get_elapsed_time_second();
                this->inner_product_time_ += tmp_inner_product_time;
                const int ip_cost = n_user_;
                this->total_ip_cost_ += ip_cost;

                rank_bound_record_.reset();
                rank_ins_.RankBound(queryIP_l_, prune_l_, result_l_, IPbound_l_, bucketID_l_, rank_lb_l_, rank_ub_l_);
                const double tmp_rank_bound_time = rank_bound_record_.get_elapsed_time_second();
                rank_bound_time_ += tmp_rank_bound_time;

                //rank search
                int refine_user_size = n_user_;
                int n_result_user = 0;
                int n_prune_user = 0;
                prune_user_record_.reset();
                PruneCandidateByBound(rank_lb_l_, rank_ub_l_,
                                      n_user_, topk,
                                      refine_seq_l_, refine_user_size,
                                      n_result_user, n_prune_user,
                                      prune_l_, result_l_);
                const double tmp_prune_user_time = prune_user_record_.get_elapsed_time_second();
                prune_user_time_ += tmp_prune_user_time;
                assert(n_result_user + n_prune_user + refine_user_size <= n_user_);
                assert(0 <= n_result_user && n_result_user <= topk);
                sample_refine_user_ += refine_user_size;

                //rank search
                rank_bound_record_.reset();
                bit_index_.ScoreDistributionRankBound(queryIP_l_, prune_l_, result_l_, bucketID_l_,
                                                      IPbound_l_,
                                                      rank_lb_l_, rank_ub_l_);
                const double tmp_second_rank_bound_time = rank_bound_record_.get_elapsed_time_second();
                rank_bound_time_ += tmp_second_rank_bound_time;

                prune_user_record_.reset();
                PruneCandidateByBound(rank_lb_l_, rank_ub_l_,
                                      n_user_, topk,
                                      refine_seq_l_, refine_user_size,
                                      n_result_user, n_prune_user,
                                      prune_l_, result_l_);
                const double tmp_prune_user_time2 = prune_user_record_.get_elapsed_time_second();
                prune_user_time_ += tmp_prune_user_time2;
                assert(n_result_user + n_prune_user + refine_user_size <= n_user_);
                assert(0 <= n_result_user && n_result_user <= topk);

                //read disk and fine binary search
                size_t io_cost = 0;
                double tmp_read_disk_time = 0;
                double tmp_refine_rank_time = 0;
                disk_ins_.GetRank(queryIP_l_, rank_lb_l_, rank_ub_l_,
                                  refine_seq_l_, refine_user_size, topk - n_result_user,
                                  io_cost, tmp_read_disk_time, tmp_refine_rank_time);
                total_io_cost_ += io_cost;
                total_refine_user_ += disk_ins_.n_read_disk_user_;
                rank_prune_ratio_ += 1.0 * (n_user_ - disk_ins_.n_read_disk_user_) / n_user_;
                read_disk_time_ += tmp_read_disk_time;
                compute_rank_time_ += tmp_refine_rank_time;

                int n_cand = 0;
                for (int userID = 0; userID < n_user_; userID++) {
                    if (result_l_[userID]) {
                        query_heap_l[queryID][n_cand] = UserRankElement(userID, rank_lb_l_[userID], queryIP_l_[userID]);
                        n_cand++;
                    }
                }

                for (int candID = n_cand; candID < topk; candID++) {
                    query_heap_l[queryID][candID] = disk_ins_.user_topk_cache_l_[candID - n_cand];
                }
                assert(n_cand + disk_ins_.n_refine_user_ >= topk);
                assert(query_heap_l[queryID].size() == topk);

                const double total_time =
                        total_retrieval_record_.get_elapsed_time_second();
                total_retrieval_time_ += total_time;
                const double &memory_index_time = tmp_rank_bound_time + tmp_inner_product_time;
                query_performance_l[queryID] = SingleQueryPerformance(queryID,
                                                                      n_prune_user, n_result_user,
                                                                      disk_ins_.n_read_disk_user_,
                                                                      ip_cost, io_cost,
                                                                      total_time,
                                                                      memory_index_time, tmp_read_disk_time);
            }
            disk_ins_.FinishRetrieval();

            rank_prune_ratio_ /= n_query_item;

            return query_heap_l;
        }

        std::string
        PerformanceStatistics(const int &topk) override {
            // int topk;
            //double total_time,
            //          inner_product_time, coarse_binary_search_time, read_disk_time
            //          fine_binary_search_time;
            //double rank_prune_ratio;
            //unit: second

            char buff[1024];

            sprintf(buff,
                    "top%d retrieval time: total %.3fs\n\tinner product %.3fs, rank bound %.3fs, prune user %.3fs, read disk %.3fs, compute rank %.3fs\n\ttotal ip cost %ld, total io cost %ld, sample refine user %ld, total refine user %ld, rank prune ratio %.4f",
                    topk, total_retrieval_time_,
                    inner_product_time_, rank_bound_time_, prune_user_time_, read_disk_time_, compute_rank_time_,
                    total_ip_cost_, total_io_cost_, sample_refine_user_, total_refine_user_, rank_prune_ratio_);
            std::string str(buff);
            return str;
        }

        uint64_t IndexSizeByte() override {
            return rank_ins_.IndexSizeByte();
        }

    };

    /*
     * bruteforce index
     * shape: n_user * n_data_item, type: float, the distance pair for each user
     */

    std::unique_ptr<Index>
    BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const char *index_path,
               const char *dataset_name, const char *index_basic_dir,
               const int &n_sample, const int &n_sample_query, const int &sample_topk, const int &n_bit) {
        const int n_user = user.n_vector_;
        const int n_data_item = data_item.n_vector_;

        //rank search
        SampleSearch rank_ins(index_basic_dir, dataset_name, "QueryRankSampleScoreDistribution",
                              n_sample, true, true, n_sample_query, sample_topk);
        BitIndex bit_index(n_data_item, n_user, n_sample, n_bit);

        ReadAllDirectIO disk_ins(n_user, n_data_item, index_path);

        //disk index
        ReadAll read_ins(n_user, n_data_item, index_path);
        read_ins.RetrievalPreprocess();

        const int report_every = 10000;
        TimeRecord record;
        record.reset();
        std::vector<float> distance_l(n_data_item);
        for (int userID = 0; userID < n_user; userID++) {
            read_ins.ReadDiskNoCache(userID, distance_l);

            bit_index.LoopPreprocess(distance_l.data(), rank_ins.known_rank_idx_l_.get(), userID);

            if (userID % report_every == 0) {
                std::cout << "preprocessed " << userID / (0.01 * n_user) << " %, "
                          << record.get_elapsed_time_second() << " s/iter" << " Mem: "
                          << get_current_RSS() / 1000000 << " Mb \n";
                record.reset();
            }
        }
        read_ins.FinishRetrieval();

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(rank_ins, bit_index, disk_ins, user, n_data_item);
        return index_ptr;
    }

}
#endif //REVERSE_K_RANKS_QUERYRANKSAMPLESCOREDISTRIBUTION_HPP
