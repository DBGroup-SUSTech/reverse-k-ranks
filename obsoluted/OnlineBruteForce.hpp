#pragma once

#include "alg/SpaceInnerProduct.hpp"
#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/MethodBase.hpp"
#include "util/TimeMemory.hpp"
#include <vector>
#include <queue>

namespace ReverseMIPS::OnlineBruteForce {

    class Index : public BaseIndex {
    public:
        VectorMatrix data_item_, user_;
        int vec_dim_;
        int report_every_ = 10;

        inline Index() {}

        inline Index(VectorMatrix &data_item, VectorMatrix &user) {
            this->data_item_ = std::move(data_item);
            vec_dim_ = user.vec_dim_;
            this->user_ = std::move(user);
            this->user_.vectorNormalize();
        }

        void Preprocess() {}

        inline ~Index() {}

        std::vector<std::vector<UserRankElement>> Retrieval(const VectorMatrix &query_item, const int &topk) override {
            if (topk > user_.n_vector_) {
                spdlog::error("top-k is larger than user, system exit");
                exit(-1);
            }
            int n_query_item = query_item.n_vector_;
            int n_user = user_.n_vector_;

            std::vector<std::vector<UserRankElement>> results(n_query_item, std::vector<UserRankElement>(topk));

            TimeRecord single_query_record;
#pragma omp parallel for default(none) shared(n_query_item, query_item, results, topk, n_user, single_query_record, std::cout)
            for (int qID = 0; qID < n_query_item; qID++) {
                double *query_item_vec = query_item.getVector(qID);
                std::vector<UserRankElement> &minHeap = results[qID];

                for (int userID = 0; userID < topk; userID++) {
                    double queryIP = InnerProduct(query_item_vec, user_.getVector(userID), vec_dim_);
                    int tmp_rank = getRank(queryIP, user_.getVector(userID));
                    UserRankElement element(userID, tmp_rank, queryIP);
                    minHeap[userID] = element;
                }

                std::make_heap(minHeap.begin(), minHeap.end(), std::less<UserRankElement>());
                UserRankElement minHeapEle = minHeap.front();
                for (int userID = topk; userID < n_user; userID++) {
                    double queryIP = InnerProduct(query_item_vec, user_.getVector(userID), vec_dim_);
                    int tmp_rank = getRank(queryIP, user_.getVector(userID));
                    UserRankElement element(userID, tmp_rank, queryIP);

                    if (minHeapEle > element) {
                        std::pop_heap(minHeap.begin(), minHeap.end(), std::less<UserRankElement>());
                        minHeap.pop_back();
                        minHeap.push_back(element);
                        std::push_heap(minHeap.begin(), minHeap.end(), std::less<UserRankElement>());
                        minHeapEle = minHeap.front();
                    }
                }
                std::make_heap(minHeap.begin(), minHeap.end(), std::less<UserRankElement>());
                std::sort_heap(minHeap.begin(), minHeap.end(), std::less<UserRankElement>());

                if (qID % report_every_ == 0) {
                    std::cout << "retrieval top" << topk << " progress " << qID / (0.01 * n_query_item) << " %, "
                              << single_query_record.get_elapsed_time_second() << " s/iter" << " Mem: "
                              << get_current_RSS() / 1000000 << " Mb \n";
                    single_query_record.reset();
                }
            }

            return results;
        }

        int getRank(double queryIP, double *user_vec) {

            int n_data_item = data_item_.n_vector_;
            int rank = 1;

            for (int i = 0; i < n_data_item; i++) {
                double data_dist = InnerProduct(data_item_.getVector(i), user_vec, vec_dim_);
                rank += data_dist > queryIP ? 1 : 0;
            }

            return rank;
        }

        std::string
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &ms_per_query) override {
            // int topk;
            //double total_time,
            //double ms_per_query;
            //unit: second

            char buff[1024];

            sprintf(buff,
                    "top%d retrieval time:\n\ttotal %.3fs\n\tmillion second per query %.3fms",
                    topk, retrieval_time, ms_per_query);
            std::string str(buff);
            return str;
        }

    };

    std::unique_ptr<Index> BuildIndex(VectorMatrix &data_item, VectorMatrix &user) {
        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(data_item, user);
        index_ptr->Preprocess();
        return index_ptr;
    }

}

