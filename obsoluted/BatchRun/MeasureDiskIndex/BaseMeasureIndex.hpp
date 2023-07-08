//
// Created by BianZheng on 2022/7/25.
//

#ifndef REVERSE_K_RANKS_BASEMEASUREINDEX_HPP
#define REVERSE_K_RANKS_BASEMEASUREINDEX_HPP

#include "struct/UserRankElement.hpp"
#include "struct/VectorMatrix.hpp"
#include <vector>

namespace ReverseMIPS {
    class BaseMeasureIndex {
    public:
        virtual void
        Retrieval(const VectorMatrix &query_item, const int &topk, const int &n_eval_query_item,
                  uint64_t *n_item_candidate_l) = 0;

        virtual std::string
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &ms_per_query) = 0;

        virtual std::string BuildIndexStatistics() {
            return "Build Index Info: none";
        };

        virtual std::string VariancePerformanceMetricName() {
            return "queryID, retrieval time, second per query";
        }

        virtual std::string VariancePerformanceStatistics(
                const double &retrieval_time, const double &second_per_query, const int &queryID) {
            char str[256];
            sprintf(str, "%d,%.3f,%.3f", queryID, retrieval_time, second_per_query);
            return str;
        };

        virtual ~BaseMeasureIndex() = default;

    };

}


#endif //REVERSE_K_RANKS_BASEMEASUREINDEX_HPP
