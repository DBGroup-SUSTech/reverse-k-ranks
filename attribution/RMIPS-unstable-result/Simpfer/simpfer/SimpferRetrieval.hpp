//
// Created by BianZheng on 2022/9/11.
//

#ifndef REVERSE_K_RANKS_SIMPFERRETRIEVAL_HPP
#define REVERSE_K_RANKS_SIMPFERRETRIEVAL_HPP

#include "SimpferBuildIndex.hpp"
#include "SimpferData.hpp"

#include "fexipro/alg/SVDIntUpperBoundIncrPrune2.h"
#include "fexipro/structs/VectorElement.h"

#include "alg/SpaceInnerProduct.hpp"
#include "util/TimeMemory.hpp"


namespace ReverseMIPS {
    class SimpferIndex {
        // verification by sequential scan
        // return value is whether contains in the reverse top-k result
        bool SequentialScan(SimpferData &user, const float &ip_, const int &rtk_topk, size_t &ip_count) {

            for (unsigned int itemID = 0; itemID < n_data_item_; ++itemID) {
                const float item_norm_ub = user.norm_ * data_item_sd_l_[itemID].norm_;
                // guarantee to be top-k
                if (ip_ >= item_norm_ub || user.threshold_ >= item_norm_ub) {
                    return true;
                }

                // compute ip
                const float ip = InnerProduct(user.vec_.data(), data_item_sd_l_[itemID].vec_.data(), (int) vec_dim_);
                ++ip_count;

                // update top-k
                user.UpdateTopk(ip, data_item_sd_l_[itemID].ID_, rtk_topk);

                // out from top-k
                if (user.threshold_ > ip_) {
                    return false;
                }
            }
            if (user.threshold_ > ip_) {
                return false;
            } else {
                return true;
            }
        }

    public:
        int64_t n_user_, n_data_item_, vec_dim_;
        int64_t k_max_;
        std::vector<SimpferData> user_sd_l_;
        std::vector<SimpferData> data_item_sd_l_;
        std::vector<SimpferBlock> block_l_;
        SIRPrune sir_prune_;

        inline SimpferIndex() = default;

        inline SimpferIndex(std::vector<SimpferData> &user_sd_l,
                            std::vector<SimpferData> &data_item_sd_l,
                            std::vector<SimpferBlock> &block_l,
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

        void ComputeByIndex(const SimpferData &query_item, const int &rtk_topk,
                            std::vector<int> &result_userID_l,
                            int &n_block_prune, int &sample_prune, int &norm_prune, size_t &ip_count,
                            int &result_size) {
            // init
            for (unsigned int userID = 0; userID < n_user_; ++userID) {
                user_sd_l_[userID].init();
            }

            // query_norm computation
            float query_norm = InnerProduct(query_item.vec_.data(), query_item.vec_.data(), (int) vec_dim_);
            query_norm = std::sqrt(query_norm);

#pragma omp parallel for default(none) reduction(+:ip_count, norm_prune, sample_prune, n_block_prune, result_size) shared(result_userID_l, query_norm, rtk_topk, query_item)
            for (unsigned int blockID = 0; blockID < block_l_.size(); ++blockID) {
                // compute upper-bound in this block
                const SimpferBlock block = block_l_[blockID];
                float upperbound = user_sd_l_[block.userID_l[0]].norm_ * query_norm;
                assert(user_sd_l_[block.userID_l[0]].norm_ >= user_sd_l_[block.userID_l[1]].norm_);
                const int n_user_member = (int) block.userID_l.size();
                std::vector<int> tmp_result_userID_l;
                size_t tmp_ip_count = 0;
                int tmp_norm_prune = 0;
                int tmp_sample_prune = 0;
                int tmp_n_block_prune = 0;
                int tmp_result_size = 0;

                // block-level filtering
                if (upperbound > block_l_[blockID].lowerbound_array[rtk_topk - 1]) {
                    for (unsigned int memberID = 0; memberID < n_user_member; ++memberID) {
                        // get user-vec
                        SimpferData user = user_sd_l_[block.userID_l[memberID]];
                        // compute ip
                        const float ip = InnerProduct(query_item.vec_.data(), user.vec_.data(), (int) vec_dim_);
                        ++tmp_ip_count;
                        // comparison with lower-bound
                        if (ip > user.lowerbound_array_[rtk_topk - 1]) {
                            upperbound = user.norm_ * data_item_sd_l_[rtk_topk - 1].norm_;
                            // comparison with upper-bound
                            if (upperbound > ip) {
                                bool is_result = SequentialScan(user, ip, rtk_topk, tmp_ip_count);
                                if (is_result) {
                                    tmp_result_userID_l.emplace_back(user.ID_);
                                    tmp_result_size++;
                                }
                            } else {
                                tmp_norm_prune++;
                                tmp_result_userID_l.emplace_back(user.ID_);
                                tmp_result_size++;
                            }
                        } else {
                            tmp_sample_prune++;
                            tmp_norm_prune++;
                        }
                    }
                } else {
                    tmp_n_block_prune += n_user_member;
                    tmp_sample_prune += n_user_member;
                    tmp_norm_prune += n_user_member;
                }

                ip_count += tmp_ip_count;
                norm_prune += tmp_norm_prune;
                sample_prune += tmp_sample_prune;
                n_block_prune += tmp_n_block_prune;
                result_size += tmp_result_size;
#pragma omp critical
                {
                    std::for_each(tmp_result_userID_l.begin(), tmp_result_userID_l.end(), [&](int &userID) {
                        result_userID_l.push_back(userID);
                    });
                };
            }
            assert(result_size <= n_user_);
        }

        void ComputeByBruteforce(const SimpferData &query_item, const Matrix &user_matrix,
                                 const int &rtk_topk,
                                 std::vector<int> &result_userID_l,
                                 size_t &ip_count,
                                 int &result_size) {

            std::vector<VectorElement> topk_res_l(n_user_);

            sir_prune_.topK(user_matrix, rtk_topk, topk_res_l, ip_count);

            std::vector<float> queryIP_l(n_user_);
#pragma omp parallel for default(none) shared(queryIP_l, query_item, user_matrix)
            for (int userID = 0; userID < n_user_; userID++) {
                const float queryIP = InnerProduct(query_item.vec_.data(), user_matrix.getRowPtr(userID),
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
        void RTopKRetrieval(const SimpferData &query_item, Matrix &user_matrix, const int &rtk_topk,
                            std::vector<int> &result_userID_l,
                            int &n_block_prune, int &sample_prune, int &norm_prune, size_t &ip_count,
                            int &result_size) {

            result_userID_l.clear();
            n_block_prune = 0;
            sample_prune = 0;
            norm_prune = 0;
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
                               result_userID_l,
                               n_block_prune, sample_prune, norm_prune, ip_count,
                               result_size);

            }

        }

        // main operation
        void FEIXPROOnlyRetrieval(const SimpferData &query_item, Matrix &user_matrix, const int &rtk_topk,
                                  std::vector<int> &result_userID_l,
                                  int &n_block_prune, int &sample_prune, int &norm_prune, size_t &ip_count,
                                  int &result_size) {

            result_userID_l.clear();
            n_block_prune = 0;
            sample_prune = 0;
            norm_prune = 0;
            ip_count = 0;
            result_size = 0;
            assert((int) result_userID_l.size() == 0);

            if (rtk_topk > k_max_) {
                ComputeByBruteforce(query_item, user_matrix, rtk_topk,
                                    result_userID_l,
                                    ip_count,
                                    result_size);
            }

        }

        // main operation
        void SimpferOnlyRetrieval(const SimpferData &query_item, Matrix &user_matrix, const int &rtk_topk,
                                  std::vector<int> &result_userID_l,
                                  int &n_block_prune, int &sample_prune, int &norm_prune, size_t &ip_count,
                                  int &result_size) {

            result_userID_l.clear();
            n_block_prune = 0;
            sample_prune = 0;
            norm_prune = 0;
            ip_count = 0;
            result_size = 0;
            assert((int) result_userID_l.size() == 0);

            if (rtk_topk > k_max_) {
                return;
            } else {
                ComputeByIndex(query_item, rtk_topk,
                               result_userID_l,
                               n_block_prune, sample_prune, norm_prune, ip_count,
                               result_size);

            }

        }

        uint64_t IndexSizeByte() const {
            return 0;
        }
    };

}
#endif //REVERSE_K_RANKS_SIMPFERRETRIEVAL_HPP
