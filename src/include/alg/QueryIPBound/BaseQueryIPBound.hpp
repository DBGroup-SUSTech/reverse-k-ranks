//
// Created by BianZheng on 2022/5/19.
//

#ifndef REVERSE_K_RANKS_BASEIPBOUND_HPP
#define REVERSE_K_RANKS_BASEIPBOUND_HPP

#include "struct/VectorMatrix.hpp"

#include <vector>

namespace ReverseMIPS {
    class BaseQueryIPBound {
    public:
        virtual void Preprocess(const VectorMatrix &user) = 0;

        virtual void
        IPBound(const float *query_vecs, const VectorMatrix &user,
                std::vector<std::pair<float, float>> &queryIP_l, const int& n_proc_user) const = 0;

        virtual uint64_t IndexSizeByte() const = 0;

        virtual ~BaseQueryIPBound() = default;

    };
}

#endif //REVERSE_K_RANKS_BASEIPBOUND_HPP
