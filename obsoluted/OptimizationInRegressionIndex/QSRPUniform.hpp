//
// Created by bianzheng on 2023/4/14.
//

#ifndef REVERSE_KRANKS_QSRPUNIFORM_HPP
#define REVERSE_KRANKS_QSRPUNIFORM_HPP

#include "util/NameTranslation.hpp"

#include "alg/SpaceInnerProduct.hpp"
#include "alg/TopkMaxHeap.hpp"
#include "alg/DiskIndex/ComputeAllIPBound.hpp"
#include "alg/QueryIPBound/PartDimPartNorm.hpp"
#include "alg/RegressionPruning/UniformLinearRegressionFast.hpp"
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

namespace ReverseMIPS::QSRPUniform {

    class Index : public BaseIndex {
        void ResetTimer() {
            total_retrieval_time_ = 0;
            inner_product_time_ = 0;
            rank_bound_time_ = 0;
            prune_user_time_ = 0;
            refine_user_time_ = 0;

            total_ip_cost_ = 0;
            total_refine_ip_cost_ = 0;
            total_refine_user_ = 0;

            rank_prune_ratio_ = 0;
            ip_bound_prune_ratio_ = 0;
            avg_compute_item_ = 0;
        }

        // IP Bound
        SVD svd_ins_;
        PartDimPartNorm ip_bound_ins_;
        //rank bound search
        UniformLinearRegressionFast rank_bound_ins_;
        //rank search
        SampleSearch rank_ins_;
        //read disk
        ComputeAllIPBound disk_ins_;

        VectorMatrix user_, data_item_;
        int vec_dim_, n_data_item_, n_user_;
        double total_retrieval_time_, inner_product_time_, rank_bound_time_, prune_user_time_, refine_user_time_;
        TimeRecord total_retrieval_record_, inner_product_record_, rank_bound_record_, prune_user_record_;
        uint64_t total_ip_cost_, total_refine_ip_cost_, total_refine_user_;
        double rank_prune_ratio_, ip_bound_prune_ratio_, avg_compute_item_;

    public:

        //temporary retrieval variable
        std::vector<char> prune_l_;
        std::vector<char> result_l_;
        std::vector<float> queryIP_l_;
        std::vector<int> rank_lb_l_;
        std::vector<int> rank_ub_l_;
        std::vector<std::pair<float, float>> queryIP_bound_l_;
        std::unique_ptr<float[]> query_vecs_ptr_;

        Index(
                //SVD ins
                SVD &svd_ins,
                //ip bound ins
                PartDimPartNorm &ip_bound_ins,
                //rank search for compute loose rank bound
                UniformLinearRegressionFast &rank_bound_ins,
                //rank search
                SampleSearch &rank_ins,
                //disk index
                ComputeAllIPBound &disk_ins,
                //general retrieval
                VectorMatrix &user, VectorMatrix &data_item
        ) {
            // IP Bound
            this->svd_ins_ = std::move(svd_ins);
            this->ip_bound_ins_ = std::move(ip_bound_ins);
            this->rank_bound_ins_ = std::move(rank_bound_ins);
            //rank search
            this->rank_ins_ = std::move(rank_ins);
            //read disk
            this->disk_ins_ = std::move(disk_ins);

            this->user_ = std::move(user);
            this->data_item_ = std::move(data_item);
            this->vec_dim_ = this->user_.vec_dim_;
            this->n_user_ = this->user_.n_vector_;
            this->n_data_item_ = this->data_item_.n_vector_;

            //retrieval variable
            this->prune_l_.resize(n_user_);
            this->result_l_.resize(n_user_);
            this->queryIP_l_.resize(n_user_);
            this->rank_lb_l_.resize(n_user_);
            this->rank_ub_l_.resize(n_user_);
            this->queryIP_bound_l_.resize(n_user_);
            query_vecs_ptr_ = make_unique<float[]>(vec_dim_);
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

                const float *old_query_vecs = query_item.getVector(queryID);
                float *query_vecs = query_vecs_ptr_.get();
                svd_ins_.TransferQuery(old_query_vecs, vec_dim_, query_vecs);

                inner_product_record_.reset();
                ip_bound_ins_.IPBound(query_vecs, user_, queryIP_bound_l_, n_user_);
                const double tmp_ip_bound_time = inner_product_record_.get_elapsed_time_second();
                this->inner_product_time_ += tmp_ip_bound_time;

                rank_bound_record_.reset();
                rank_bound_ins_.RankBound(queryIP_bound_l_, rank_lb_l_, rank_ub_l_);
                const double tmp_first_rank_bound_time = rank_bound_record_.get_elapsed_time_second();
                rank_bound_time_ += tmp_first_rank_bound_time;

                int refine_user_size = n_user_;
                int n_result_user = 0;
                int n_prune_user = 0;
                prune_user_record_.reset();
                PruneCandidateByBound(rank_lb_l_, rank_ub_l_,
                                      n_user_, topk,
                                      refine_user_size, n_result_user, n_prune_user,
                                      prune_l_, result_l_);
                const double tmp_prune_user_time = prune_user_record_.get_elapsed_time_second();
                prune_user_time_ += tmp_prune_user_time;
                ip_bound_prune_ratio_ += 1.0 * (n_user_ - refine_user_size) / n_user_;
                assert(n_result_user + n_prune_user + refine_user_size <= n_user_);
                assert(0 <= n_result_user && n_result_user <= n_user_);
//                spdlog::error("refine user size {}", refine_user_size);

                //calculate IP
                inner_product_record_.reset();
                ip_bound_ins_.ComputeRemainDim(query_vecs, user_, prune_l_, result_l_, queryIP_l_, n_user_);
                const double tmp_inner_product_time = inner_product_record_.get_elapsed_time_second();
                this->inner_product_time_ += tmp_inner_product_time;
                const int ip_cost = refine_user_size;
                this->total_ip_cost_ += ip_cost;

                //rank search
                rank_bound_record_.reset();
                rank_ins_.RankBound(queryIP_l_, prune_l_, result_l_, rank_lb_l_, rank_ub_l_);
                const double tmp_second_rank_bound_time = rank_bound_record_.get_elapsed_time_second();
                rank_bound_time_ += tmp_second_rank_bound_time;

                prune_user_record_.reset();
                PruneCandidateByBound(rank_lb_l_, rank_ub_l_,
                                      n_user_, topk,
                                      refine_user_size, n_result_user, n_prune_user,
                                      prune_l_, result_l_);
                const double tmp_prune_user_time2 = prune_user_record_.get_elapsed_time_second();
                prune_user_time_ += tmp_prune_user_time2;
                assert(n_result_user + n_prune_user + refine_user_size <= n_user_);
                assert(0 <= n_result_user && n_result_user <= n_user_);

                //read disk and fine binary search
                size_t refine_ip_cost = 0;
                double tmp_refine_user_time = 0;
                int n_refine_user = 0;
                int64_t n_compute_item = 0;
                disk_ins_.GetRank(user_, data_item_, queryIP_l_,
                                  rank_lb_l_, rank_ub_l_,
                                  prune_l_, result_l_,
                                  std::max(0, topk - n_result_user), refine_ip_cost, n_refine_user, n_compute_item,
                                  tmp_refine_user_time);
                total_refine_ip_cost_ += refine_ip_cost;
                total_refine_user_ += n_refine_user;
                rank_prune_ratio_ += 1.0 * (double) (n_user_ - n_refine_user) / n_user_;
                avg_compute_item_ += (double) n_compute_item;
                refine_user_time_ += tmp_refine_user_time;

                int n_cand = 0;
                for (int userID = 0; userID < n_user_; userID++) {
                    if (result_l_[userID]) {
                        query_heap_l[queryID][n_cand] = UserRankElement(userID, rank_lb_l_[userID], queryIP_l_[userID]);
                        n_cand++;
                    }
                    if (n_cand == topk) {
                        break;
                    }
                }

                for (int candID = n_cand; candID < topk; candID++) {
                    query_heap_l[queryID][candID] = disk_ins_.user_topk_cache_l_[candID - n_cand];
                }
                assert(n_cand + n_refine_user >= topk);
                assert(query_heap_l[queryID].size() == topk);

                const double total_time =
                        total_retrieval_record_.get_elapsed_time_second();
                total_retrieval_time_ += total_time;
                const double &memory_index_time = tmp_ip_bound_time + tmp_first_rank_bound_time +
                                                  tmp_inner_product_time + tmp_second_rank_bound_time;
                query_performance_l[queryID] = SingleQueryPerformance(queryID,
                                                                      n_prune_user, n_result_user,
                                                                      (int) n_refine_user,
                                                                      ip_cost + refine_ip_cost, 0,
                                                                      total_time,
                                                                      memory_index_time, 0);
            }

            rank_prune_ratio_ /= n_query_item;
            ip_bound_prune_ratio_ /= n_query_item;
            avg_compute_item_ /= total_refine_user_;

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
                    "top%d retrieval time: total %.3fs\n\tinner product %.3fs, rank bound %.3fs, prune user %.3fs, refine user %.3fs\n\ttotal ip cost %ld, total refine ip cost %ld, total refine user %ld, IP bound prune ratio %.4f, rank prune ratio %.4f, average compute item %.3f",
                    topk, total_retrieval_time_,
                    inner_product_time_, rank_bound_time_, prune_user_time_, refine_user_time_,
                    total_ip_cost_, total_refine_ip_cost_, total_refine_user_, ip_bound_prune_ratio_, rank_prune_ratio_,
                    avg_compute_item_);
            std::string str(buff);
            return str;
        }

        uint64_t IndexSizeByte() override {
            return rank_ins_.IndexSizeByte() + rank_bound_ins_.IndexSizeByte() + ip_bound_ins_.IndexSizeByte();
        }

    };

    std::unique_ptr<Index>
    BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const char *dataset_name,
               const int &n_sample, const int &n_sample_query, const int &sample_topk, const int &epoch_rp,
               const char *index_basic_dir) {
        const int n_user = user.n_vector_;
        const int vec_dim = user.vec_dim_;

        SVD svd_ins;
        const float SIGMA = 0.7;
        const int check_dim = svd_ins.Preprocess(user, data_item, SIGMA);

        PartDimPartNorm ip_bound_ins(n_user, vec_dim, check_dim);
        ip_bound_ins.Preprocess(user);

        const std::string index_name = IndexName("QSRPUniform");
        //rank search
        SampleSearch rank_ins(index_basic_dir, dataset_name, index_name.c_str(),
                              n_sample, true, true,
                              n_sample_query, sample_topk);
        spdlog::info("finish load sample index");

        UniformLinearRegressionFast rank_bound_ins(index_basic_dir, dataset_name, "QSRPUniform",
                                                   n_sample, n_sample_query, sample_topk,
                                                   epoch_rp);
        spdlog::info("finish load linear regression index");

        //disk index
        ComputeAllIPBound disk_ins(user, data_item, check_dim);

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(svd_ins, ip_bound_ins, rank_bound_ins, rank_ins,
                                                                   disk_ins,
                                                                   user, data_item);
        return index_ptr;
    }

}
#endif //REVERSE_KRANKS_QSRPUNIFORM_HPP
