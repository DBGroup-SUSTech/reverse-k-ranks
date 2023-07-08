//
// Created by bianzheng on 2023/6/16.
//

#ifndef REVERSE_KRANKS_MEMORYBRUTEFORCEUPDATE_HPP
#define REVERSE_KRANKS_MEMORYBRUTEFORCEUPDATE_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "util/TimeMemory.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrixUpdate.hpp"
#include "struct/MethodBaseUpdate.hpp"
#include <vector>
#include <algorithm>
#include <cassert>
#include <queue>

namespace ReverseMIPS::MemoryBruteForceUpdate {

    class Index : public BaseUpdateIndex {
    private:
        void ResetTime() {
            this->inner_product_time_ = 0;
            this->binary_search_time_ = 0;
            this->total_retrieval_time_ = 0;
        }

    public:
        VectorMatrixUpdate data_item_, user_;
        std::vector<float> distance_table_; // n_user_ * n_data_item_
//        int n_user_, n_data_item_;

        int vec_dim_;
        int preprocess_report_every_ = 100;
        double total_retrieval_time_, inner_product_time_, binary_search_time_;
        TimeRecord total_retrieval_record_, preprocess_record_, inner_product_record_, binary_search_record_;

        Index() {}

        Index(VectorMatrixUpdate &data_item, VectorMatrixUpdate &user) {
            this->vec_dim_ = user.vec_dim_;
            this->data_item_ = std::move(data_item);
            this->user_ = std::move(user);
        }

        ~Index() {}

        void Preprocess() {
            int n_data_item = data_item_.n_vector_;
            int n_user = user_.n_vector_;
            std::vector<float> preprocess_matrix(n_user * n_data_item);

            preprocess_record_.reset();
#pragma omp parallel for default(none) shared(n_user, preprocess_matrix, n_data_item)
            for (int userID = 0; userID < n_user; userID++) {

                for (int itemID = 0; itemID < n_data_item; itemID++) {
                    float query_dist = InnerProduct(data_item_.getVector(itemID), user_.getVector(userID), vec_dim_);
                    preprocess_matrix[userID * n_data_item + itemID] = query_dist;
                }

                std::make_heap(preprocess_matrix.begin() + userID * n_data_item,
                               preprocess_matrix.begin() + (userID + 1) * n_data_item, std::greater<float>());
                std::sort_heap(preprocess_matrix.begin() + userID * n_data_item,
                               preprocess_matrix.begin() + (userID + 1) * n_data_item, std::greater<float>());

//                if (userID % preprocess_report_every_ == 0) {
//                    std::cout << "preprocessed " << userID / (0.01 * n_user) << " %, "
//                              << preprocess_record_.get_elapsed_time_second() << " s/iter" << " Mem: "
//                              << get_current_RSS() / 1000000 << " Mb \n";
//                    preprocess_record_.reset();
//                }

            }

            this->distance_table_ = preprocess_matrix;
            this->n_data_item_ = n_data_item;
            this->n_user_ = n_user;

        }

        std::vector<std::vector<UserRankElement>>
        Retrieval(const VectorMatrixUpdate &query_item, const int &topk, const int &n_execute_query,
                  std::vector<SingleQueryPerformance> &query_performance_l) override {
            if (topk > user_.n_vector_) {
                spdlog::error("top-k is larger than user, system exit");
                exit(-1);
            }
            ResetTime();

            int n_query_item = n_execute_query;
            int n_user = user_.n_vector_;

            if (n_execute_query > query_item.n_vector_) {
                spdlog::error("n_execute_query larger than n_query_item, program exit");
                exit(-1);
            }
            Preprocess();

            std::vector<std::vector<UserRankElement>> results(n_query_item, std::vector<UserRankElement>());

#pragma omp parallel for default(none) shared(n_query_item, query_item, results, topk, n_user)
            for (int qID = 0; qID < n_query_item; qID++) {
                total_retrieval_record_.reset();
                float *query_item_vec = query_item.getVector(qID);
                std::vector<UserRankElement> &minHeap = results[qID];
                minHeap.resize(topk);

                std::vector<UserRankElement> user_rank_l(n_user_);
                for (int userID = 0; userID < n_user_; userID++) {
                    float *user_vec = user_.getVector(userID);
                    float queryIP = InnerProduct(query_item_vec, user_vec, vec_dim_);
                    this->inner_product_time_ += inner_product_record_.get_elapsed_time_second();

                    binary_search_record_.reset();
                    int tmp_rank = BinarySearch(queryIP, userID);
                    this->binary_search_time_ += binary_search_record_.get_elapsed_time_second();
                    UserRankElement rankElement(userID, tmp_rank, queryIP);
                    user_rank_l[userID] = rankElement;

//                    if (qID == 84 && userID == 2974) {
//                        spdlog::info("qID: {}, userID: {}, queryIP: {}, tmp_rank: {}",
//                                     qID, userID, queryIP, tmp_rank);
//                    }
//                    if (qID == 84 && userID == 16) {
//                        spdlog::info("qID: {}, userID: {}, queryIP: {}, tmp_rank: {}",
//                                     qID, userID, queryIP, tmp_rank);
//                    }

                }
                std::sort(user_rank_l.begin(), user_rank_l.end(), std::less());

                for (int i = 0; i < topk; i++) {
                    minHeap[i] = user_rank_l[i];
                }

                const int topk_rank = user_rank_l[topk - 1].rank_;
                for (int i = topk; i < n_user_; i++) {
                    assert(user_rank_l[i].rank_ >= topk_rank);
                    if (user_rank_l[i].rank_ == topk_rank) {
                        minHeap.push_back(user_rank_l[i]);
                    } else {
                        assert(user_rank_l[i].rank_ > topk_rank);
                        break;
                    }
                }

                total_retrieval_time_ += total_retrieval_record_.get_elapsed_time_second();
            }

            return results;
        }

        int BinarySearch(float queryIP, int userID) {
            int n_data_item = data_item_.n_vector_;
            auto iter_begin = distance_table_.begin() + userID * n_data_item_;
            auto iter_end = distance_table_.begin() + (userID + 1) * n_data_item_;


            auto lb_ptr = std::lower_bound(iter_begin, iter_end, queryIP,
                                           [](const float &arrIP, float queryIP) {
                                               return arrIP > queryIP;
                                           });
            return (int) (lb_ptr - iter_begin) + 1;
        }

        std::string
        PerformanceStatistics(const std::string &info, const int &topk) override {
            // int topk;
            //double total_time,
            //          inner_product_time, binary_search_time;
            //unit: second

            char buff[1024];

            sprintf(buff,
                    "%s, top%d retrieval time:\n\ttotal %.3fs\n\tinner product %.3fs, binary search %.3fs",
                    info.c_str(), topk, total_retrieval_time_,
                    inner_product_time_, binary_search_time_);
            std::string str(buff);
            return str;
        }

        void InsertUser(const VectorMatrixUpdate &insert_user) override {
            this->user_.insert(insert_user);
            this->n_user_ = this->user_.n_vector_;
        }

        void DeleteUser(const std::vector<int> &del_userID_l) override {
            this->user_.remove(del_userID_l);
            this->n_user_ = this->user_.n_vector_;
        }

        void InsertItem(const VectorMatrixUpdate &insert_data_item) override {
            this->data_item_.insert(insert_data_item);
            this->n_data_item_ = this->data_item_.n_vector_;

        }

        void DeleteItem(const std::vector<int> &del_itemID_l) override {
            this->data_item_.remove(del_itemID_l);
            this->n_data_item_ = this->data_item_.n_vector_;

        }

        uint64_t IndexSizeByte() override {
            return sizeof(float) * n_user_ * n_data_item_;
        }

    };

    std::unique_ptr<Index> BuildIndex(VectorMatrixUpdate &data_item, VectorMatrixUpdate &user) {
        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(data_item, user);
        return index_ptr;
    }

}
#endif //REVERSE_KRANKS_MEMORYBRUTEFORCEUPDATE_HPP
