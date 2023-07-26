//
// Created by bianzheng on 2023/5/3.
//

#ifndef REVERSE_KRANKS_SIMPFERPLUSPLUSRETRIEVAL_HPP
#define REVERSE_KRANKS_SIMPFERPLUSPLUSRETRIEVAL_HPP

#include "fexipro/alg/SVDIntUpperBoundIncrPrune2.h"
#include "simpfer_plus_plus/SimpferPlusPlusBuildIndex.hpp"
#include "simpfer_plus_plus/SimpferPlusPlusData.hpp"
#include "alg/SpaceInnerProduct.hpp"

namespace ReverseMIPS {
    class SimpferPlusPlusIndex {

    public:
        int64_t n_user_, n_data_item_, vec_dim_;
        int64_t k_max_;
        std::vector<SimpferPlusPlusData> user_sd_l_;
        std::vector<SimpferPlusPlusData> data_item_sd_l_;
        std::vector<SimpferPlusPlusBlock> block_l_;
        SIRPrune sir_prune_;

        inline SimpferPlusPlusIndex() = default;

        inline SimpferPlusPlusIndex(std::vector<SimpferPlusPlusData> &user_sd_l,
                                    std::vector<SimpferPlusPlusData> &data_item_sd_l,
                                    std::vector<SimpferPlusPlusBlock> &block_l,
                                    SIRPrune &&sirPrune,
                                    const int &k_max,
                                    const int &n_user, const int &n_data_item, const int &vec_dim)
                : user_sd_l_(std::move(user_sd_l)), data_item_sd_l_(std::move(data_item_sd_l)),
                  block_l_(std::move(block_l)), sir_prune_(std::move(sirPrune)) {
            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->vec_dim_ = vec_dim;
            this->k_max_ = k_max;
        }

        void ComputeByIndex(const SimpferPlusPlusData &query_item, const int &rtk_topk,
                            std::vector<int> &result_userID_l, const int &vec_dim, size_t &ip_count) {

            // norm computation & matrix
            float norm = 0;
            Eigen::MatrixXf Q(1, vec_dim);
            for (unsigned int i = 0; i < vec_dim; ++i) {
                norm += query_item.vec[i] * query_item.vec[i];
                Q(0, i) = query_item.vec[i];
            }
            norm = sqrt(norm);

#pragma omp parallel for default(none) reduction(+:ip_count) shared( norm, rtk_topk, Q, result_userID_l) schedule(dynamic)  num_threads(omp_get_num_procs())
            for (unsigned int i = 0; i < block_l_.size(); ++i) {
                // compute upper-bound in this block
                float upperbound = block_l_[i].member[0]->norm * norm;

                // block-level filtering
                if (upperbound > block_l_[i].lowerbound_array[rtk_topk - 1]) {
                    Eigen::MatrixXf I = block_l_[i].M * Q.transpose();
                    ip_count += block_l_[i].member.size();

//                    for (unsigned int j = 0; j < block_l_[i].member.size() - 1; ++j) {
                    for (unsigned int j = 0; j < block_l_[i].member.size(); ++j) {
                        if (I(j) >= block_l_[i].member[j]->lowerbound_array[rtk_topk - 1]) {
#pragma omp critical
                            {
                                result_userID_l.push_back(block_l_[i].member[j]->identifier);
                            };
                        }
                    }
                }
            }

        }

        void ComputeByBruteforce(const SimpferPlusPlusData &query_item, const Matrix &user_matrix,
                                 const int &rtk_topk,
                                 std::vector<int> &result_userID_l,
                                 size_t &ip_count,
                                 int &result_size) {

            std::vector<VectorElement> topk_res_l(n_user_);

            sir_prune_.topK(user_matrix, rtk_topk, topk_res_l, ip_count);

            std::vector<float> queryIP_l(n_user_);
#pragma omp parallel for default(none) shared(queryIP_l, query_item, user_matrix)
            for (int userID = 0; userID < n_user_; userID++) {
                const float queryIP = InnerProduct(query_item.vec.data(), user_matrix.getRowPtr(userID),
                                                   (int) vec_dim_);
                queryIP_l[userID] = queryIP;
            }

            for (int userID = 0; userID < n_user_; userID++) {
                float topk_IP = topk_res_l[userID].data;
                const float queryIP = queryIP_l[userID];

                if (queryIP > topk_IP) {
                    result_size++;
                    result_userID_l.push_back((int) userID);
                }

            }
            ip_count += n_user_;
        }


        // main operation
        void RTopKRetrieval(const SimpferPlusPlusData &query_item, Matrix &user_matrix, const int &rtk_topk,
                            std::vector<int> &result_userID_l,
                            size_t &ip_count, int &result_size) {

            result_userID_l.clear();
            ip_count = 0;
            result_size = 0;
            assert((int) result_userID_l.size() == 0);

            if (rtk_topk > k_max_) {
                ComputeByBruteforce(query_item, user_matrix, rtk_topk,
                                    result_userID_l,
                                    ip_count,
                                    result_size);
            } else {
                ComputeByIndex(query_item, rtk_topk,
                               result_userID_l, vec_dim_, ip_count);
                result_size = result_userID_l.size();

            }

        }

        // main operation
        void SimpferPPAboveKMax(const SimpferPlusPlusData &query_item, Matrix &user_matrix, const int &rtk_topk,
                                std::vector<int> &result_userID_l,
                                size_t &ip_count, int &result_size) {

            result_userID_l.clear();
            ip_count = 0;
            result_size = 0;
            assert((int) result_userID_l.size() == 0);

            if (rtk_topk > k_max_) {
                ComputeByBruteforce(query_item, user_matrix, rtk_topk,
                                    result_userID_l,
                                    ip_count,
                                    result_size);
            } else {
                result_size = 0;

            }


        }

        // main operation
        void SimpferPPBelowKMax(const SimpferPlusPlusData &query_item, Matrix &user_matrix, const int &rtk_topk,
                                std::vector<int> &result_userID_l,
                                size_t &ip_count, int &result_size) {

            result_userID_l.clear();
            ip_count = 0;
            result_size = 0;
            assert((int) result_userID_l.size() == 0);

            if (rtk_topk > k_max_) {
                return;
            } else {
                ComputeByIndex(query_item, rtk_topk,
                               result_userID_l, vec_dim_, ip_count);
                result_size = result_userID_l.size();

            }

        }

        uint64_t IndexSizeByte() const {
            return 0;
        }
    };

}
#endif //REVERSE_KRANKS_SIMPFERPLUSPLUSRETRIEVAL_HPP
