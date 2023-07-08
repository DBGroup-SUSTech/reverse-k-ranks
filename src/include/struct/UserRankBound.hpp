//
// Created by BianZheng on 2022/4/19.
//

#ifndef REVERSE_KRANKS_USERRANKBOUND_HPP
#define REVERSE_KRANKS_USERRANKBOUND_HPP

#include <cassert>

template<typename T>
class UserRankBound {
public:
    T rank_lb_, rank_ub_;
    bool is_assign_;

    inline UserRankBound() {
        this->rank_lb_ = 0;
        this->rank_ub_ = 0;
        this->is_assign_ = false;
    }

    inline UserRankBound(const T &rank_lb, const T rank_ub) {
        this->rank_lb_ = rank_lb;
        this->rank_ub_ = rank_ub;
    }

    void Merge(T rank) {
        assert(rank_ub_ <= rank_lb_);
        if (!is_assign_) {
            rank_lb_ = rank;
            rank_ub_ = rank;
            is_assign_ = true;
        } else {
            assert(is_assign_);
            if (rank < rank_ub_) {
                rank_ub_ = rank;
            } else if (rank > rank_lb_) {
                rank_lb_ = rank;
            }
        }
    }

    std::pair<T, T> GetRankBound(){
        return std::make_pair(rank_lb_, rank_ub_);
    }

    inline void Reset() {
        this->rank_lb_ = 0;
        this->rank_ub_ = 0;
        this->is_assign_ = false;
    }
};

#endif //REVERSE_KRANKS_USERRANKBOUND_HPP
