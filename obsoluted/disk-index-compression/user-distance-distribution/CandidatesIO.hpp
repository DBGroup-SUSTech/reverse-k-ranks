//
// Created by BianZheng on 2022/6/23.
//

#ifndef REVERSE_KRANKS_CANDIDATESIO_HPP
#define REVERSE_KRANKS_CANDIDATESIO_HPP

#include "util/StringUtil.hpp"

#include <vector>
#include <fstream>

namespace ReverseMIPS {

    class ItemCandidates {
    public:
        int userID_;
        std::vector<int> itemID_l_;
        int rank_lb_, rank_ub_;

        inline ItemCandidates() = default;

        inline ItemCandidates(const std::vector<int> &itemID_l, const int &userID,
                              const int &rank_lb, const int &rank_ub) {
            this->itemID_l_ = itemID_l;
            this->userID_ = userID;
            this->rank_lb_ = rank_lb;
            this->rank_ub_ = rank_ub;
        }

    };

    void ReadUserCandidates(const char *filename, std::vector<int> &user_cand_l) {
        std::fstream in_stream;
        in_stream.open(filename, std::ios::in); //open a file to perform read operation using file object
        if (in_stream.is_open()) { //checking whether the file is open
            std::string tp;
            while (getline(in_stream, tp)) { //read data from file object and put it into string.
                if (tp.find(':') != std::string::npos) {
                    std::vector<std::string> str_l = split(tp, ':');
                    int userID = std::stoi(str_l[0]);
                    user_cand_l.emplace_back(userID);
                }
            }
            in_stream.close(); //close the file object.
        }
    }
}
#endif //REVERSE_KRANKS_CANDIDATESIO_HPP
