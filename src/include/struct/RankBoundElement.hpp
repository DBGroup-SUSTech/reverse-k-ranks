//
// Created by BianZheng on 2022/3/3.
//

#ifndef REVERSE_K_RANKS_RANKBOUNDELEMENT_HPP
#define REVERSE_K_RANKS_RANKBOUNDELEMENT_HPP

#include <string>

namespace ReverseMIPS {
    class RankBoundElement {
    public:
        int userID_, lower_rank_, upper_rank_;
        float lower_bound_, upper_bound_;

        //by default, value of upper rank is smaller than lower rank
        RankBoundElement(const int &userID, const std::pair<int, int> &bound_pair) {
            this->userID_ = userID;
            this->lower_rank_ = bound_pair.first;
            this->upper_rank_ = bound_pair.second;
        }

        RankBoundElement(const int &userID, const int &lower_rank, const int &upper_rank,
                         const float &lower_bound, const float &upper_bound) {
            this->userID_ = userID;
            this->lower_rank_ = lower_rank;
            this->upper_rank_ = upper_rank;
            this->lower_bound_ = lower_bound;
            this->upper_bound_ = upper_bound;
        }

        std::pair<float, float> IPBound() {
            return std::make_pair(lower_bound_, upper_bound_);
        }

        std::pair<int, int> rank_pair() {
            return std::make_pair(lower_rank_, upper_rank_);
        }

        RankBoundElement() {
            userID_ = -1;
            upper_rank_ = -1;
            lower_rank_ = -1;
            lower_bound_ = -1;
            upper_bound_ = -1;
        }

        ~RankBoundElement() = default;

        std::string ToString() {
            char arr[256];
            sprintf(arr, "userID %d, lower_rank %d, upper_rank %d", userID_, lower_rank_, upper_rank_);
            std::string str(arr);
            return str;
        }

        inline bool operator==(const RankBoundElement &other) const {
            if (this == &other)
                return true;
            return upper_rank_ == other.upper_rank_ && lower_rank_ == other.lower_rank_ && userID_ == other.userID_;
        };

        inline bool operator!=(const RankBoundElement &other) const {
            if (this == &other)
                return false;
            return upper_rank_ != other.upper_rank_ || lower_rank_ != other.lower_rank_ || userID_ != other.userID_;
        };

        RankBoundElement &operator=(const RankBoundElement &other) noexcept = default;

        static inline bool UpperBoundMaxHeap(const RankBoundElement &userRankBoundElement,
                                             RankBoundElement &other) {
            return userRankBoundElement.upper_rank_ < other.upper_rank_;
        }

        static inline bool LowerBoundMaxHeap(const RankBoundElement &userRankBoundElement,
                                             RankBoundElement &other) {
            return userRankBoundElement.lower_rank_ < other.lower_rank_;
        }
    };
}

#endif //REVERSE_K_RANKS_RANKBOUNDELEMENT_HPP
