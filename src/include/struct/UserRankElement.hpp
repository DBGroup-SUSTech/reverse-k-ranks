//
// Created by BianZheng on 2022/2/20.
//

#ifndef REVERSE_KRANKS_USERRANKELEMENT_HPP
#define REVERSE_KRANKS_USERRANKELEMENT_HPP

#include <string>

namespace ReverseMIPS {
    class UserRankElement {
    public:
        int userID_, rank_;
        float queryIP_;

        UserRankElement(const int &userID, const int &rank, const float &queryIP) {
            this->userID_ = userID;
            this->rank_ = rank;
            this->queryIP_ = queryIP;
        }

        UserRankElement(const int &userID, const float &queryIP) {
            this->userID_ = userID;
            this->rank_ = -1;
            this->queryIP_ = queryIP;
        }

        UserRankElement(const int &userID) {
            this->userID_ = userID;
        }

        UserRankElement() {
            userID_ = -1;
            rank_ = -1;
            queryIP_ = -1;
        }

        ~UserRankElement() = default;

        std::string ToString() {
            char arr[256];
            sprintf(arr, "userId %d, bucketID %d, queryIP %.3f", userID_, rank_, queryIP_);
            std::string str(arr);
            return str;
        }

        inline bool operator==(const UserRankElement &other) const {
            if (this == &other)
                return true;
            return rank_ == other.rank_ && userID_ == other.userID_ && queryIP_ == other.queryIP_;
        };

        inline bool operator!=(const UserRankElement &other) const {
            if (this == &other)
                return false;
            return rank_ != other.rank_ || userID_ != other.userID_ || queryIP_ != other.queryIP_;
        };

        inline bool operator<(const UserRankElement &other) const {
            if (rank_ != other.rank_) {
                return rank_ < other.rank_;
            }
            return userID_ < other.userID_;
        }

        inline bool operator<=(const UserRankElement &other) const {
            if (rank_ != other.rank_) {
                return rank_ <= other.rank_;
            }
            return userID_ <= other.userID_;

        }

        inline bool operator>(const UserRankElement &other) const {
            if (rank_ != other.rank_) {
                return rank_ > other.rank_;
            }
            return userID_ > other.userID_;
        }

        inline bool operator>=(const UserRankElement &other) const {
            if (rank_ != other.rank_) {
                return rank_ >= other.rank_;
            }
            return userID_ >= other.userID_;
        }

        static bool UserIDMinFirst(const UserRankElement &element, const UserRankElement &other) {
            return element.userID_ < other.userID_;
        }

    };
}

#endif //REVERSE_KRANKS_USERRANKELEMENT_HPP
