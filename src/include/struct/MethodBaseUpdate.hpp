//
// Created by bianzheng on 2023/7/5.
//

#ifndef REVERSE_KRANKS_METHODBASEUPDATE_HPP
#define REVERSE_KRANKS_METHODBASEUPDATE_HPP

#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrixUpdate.hpp"
#include "util/FileIO.hpp"
#include <vector>

namespace ReverseMIPS {

    class BaseUpdateIndex {
    public:
        int n_data_item_, n_user_;

        virtual std::vector<std::vector<UserRankElement>>
        Retrieval(const VectorMatrixUpdate &query_item, const int &topk, const int &n_execute_query,
                  std::vector<SingleQueryPerformance> &query_performance_l) = 0;

        virtual void InsertUser(const VectorMatrixUpdate &insert_user) = 0;

        virtual void DeleteUser(const std::vector<int> &del_userID_l) = 0;

        virtual void InsertItem(const VectorMatrixUpdate &insert_data_item) = 0;

        virtual void DeleteItem(const std::vector<int> &del_itemID_l) = 0;

        virtual std::string
        PerformanceStatistics(const std::string &info, const int &topk) = 0;

        virtual uint64_t IndexSizeByte() = 0;

        virtual void FinishCompute() {};

        virtual ~BaseUpdateIndex() = default;

    };

}
#endif //REVERSE_KRANKS_METHODBASEUPDATE_HPP
