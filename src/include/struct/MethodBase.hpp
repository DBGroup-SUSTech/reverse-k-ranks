//
// Created by BianZheng on 2022/2/22.
//

#ifndef REVERSE_KRANKS_METHODBASE_HPP
#define REVERSE_KRANKS_METHODBASE_HPP

#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"
#include "util/FileIO.hpp"
#include <vector>

namespace ReverseMIPS {
    class BaseIndex {
    public:
        virtual std::vector<std::vector<UserRankElement>>
        Retrieval(const VectorMatrix &query_item, const int &topk, const int &n_execute_query,
                  std::vector<SingleQueryPerformance> &query_performance_l) = 0;

        virtual std::string
        PerformanceStatistics(const int &topk) = 0;

        virtual uint64_t IndexSizeByte() = 0;

        virtual void FinishCompute() {};

        virtual ~BaseIndex() = default;

    };

}


#endif //REVERSE_KRANKS_METHODBASE_HPP
