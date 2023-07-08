//
// Created by BianZheng on 2022/4/1.
//

#ifndef REVERSE_KRANKS_MEMORYBRUTEFORCE_HPP
#define REVERSE_KRANKS_MEMORYBRUTEFORCE_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "util/TimeMemory.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/MethodBase.hpp"
#include <vector>
#include <algorithm>
#include <cassert>
#include <queue>

namespace ReverseMIPS::MemoryBruteForce {

    class RetrievalResult : public RetrievalResultBase {
    public:
        //unit: second
        //double total_time, inner_product_time, binary_search_time
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
                                    const double &total_time, const double &inner_product_time,
                                    const double &compare_ip_time, const double &second_per_query) {
            char buff[1024];

            sprintf(buff,
                    "top%d retrieval time:\n\ttotal %.3fs, inner product %.3fs\n\tcompare inner product %.3fs, million second per query %.3fms",
                    topk, total_time, inner_product_time, compare_ip_time, second_per_query);
            std::string str(buff);
            this->config_l.emplace_back(str);
            return str;
        }

    };

    class Index : public BaseIndex {
    private:
        void ResetTime() {
            this->inner_product_time_ = 0;
            this->compare_time_ = 0;
        }

    public:
        VectorMatrix data_item_, user_;
        std::vector<double> distance_table_; // n_user_ * n_data_item_
        int n_user_, n_data_item_;

        int vec_dim_;
        int preprocess_report_every_ = 100;
        double inner_product_time_, compare_time_;
        TimeRecord preprocess_record_, inner_product_record_, compare_record_;

        Index() {}

        Index(VectorMatrix &data_item, VectorMatrix &user) {
            this->vec_dim_ = user.vec_dim_;
            this->data_item_ = std::move(data_item);
            this->user_ = std::move(user);
        }

        ~Index() {}

        void Preprocess() {
            int n_data_item = data_item_.n_vector_;
            int n_user = user_.n_vector_;
            std::vector<double> preprocess_matrix(n_user * n_data_item);
            user_.vectorNormalize();

            preprocess_record_.reset();
#pragma omp parallel for default(none) shared(n_user, preprocess_matrix, std::cout, n_data_item)
            for (int userID = 0; userID < n_user; userID++) {

                for (int itemID = 0; itemID < n_data_item; itemID++) {
                    double query_dist = InnerProduct(data_item_.getVector(itemID), user_.getVector(userID), vec_dim_);
                    preprocess_matrix[userID * n_data_item + itemID] = query_dist;
                }

                std::make_heap(preprocess_matrix.begin() + userID * n_data_item,
                               preprocess_matrix.begin() + (userID + 1) * n_data_item, std::greater<double>());
                std::sort_heap(preprocess_matrix.begin() + userID * n_data_item,
                               preprocess_matrix.begin() + (userID + 1) * n_data_item, std::greater<double>());

                if (userID % preprocess_report_every_ == 0) {
                    std::cout << "preprocessed " << userID / (0.01 * n_user) << " %, "
                              << preprocess_record_.get_elapsed_time_second() << " s/iter" << " Mem: "
                              << get_current_RSS() / 1000000 << " Mb \n";
                    preprocess_record_.reset();
                }

            }

            this->distance_table_ = preprocess_matrix;
            this->n_data_item_ = n_data_item;
            this->n_user_ = n_user;

        }

        std::vector<std::vector<UserRankElement>> Retrieval(VectorMatrix &query_item, const int &topk) override {
            if (topk > user_.n_vector_) {
                printf("top-k is larger than user, system exit\n");
                exit(-1);
            }
            ResetTime();
            int n_query_item = query_item.n_vector_;

            std::vector<std::vector<UserRankElement>> results(n_query_item);

            for (int qID = 0; qID < n_query_item; qID++) {
                double *query_item_vec = query_item.getVector(qID);
                std::vector<UserRankElement> &minHeap = results[qID];

                for (int userID = 0; userID < n_user_; userID++) {
                    inner_product_record_.reset();
                    double *user_vec = user_.getVector(userID);
                    double queryIP = InnerProduct(query_item_vec, user_vec, vec_dim_);
                    this->inner_product_time_ += inner_product_record_.get_elapsed_time_second();

                    compare_record_.reset();
                    bool is_in_res = InResult(queryIP, userID, topk);
                    this->compare_time_ += compare_record_.get_elapsed_time_second();

                    if (is_in_res) {
                        minHeap.emplace_back(userID, queryIP);
                    }
                }

            }

            return results;
        }

        bool InResult(const double &queryIP, const int &userID, const int &topk) {
            int n_data_item = data_item_.n_vector_;
            double cmp_ip = distance_table_[userID * n_data_item + topk];
            return queryIP > cmp_ip;
        }

    };

}

#endif //REVERSE_KRANKS_MEMORYBRUTEFORCE_HPP
