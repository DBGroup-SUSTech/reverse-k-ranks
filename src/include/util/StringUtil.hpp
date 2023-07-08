//
// Created by BianZheng on 2022/1/2.
//
#include <string>
#include <vector>
#include <sstream>

#ifndef REVERSE_KRANKS_STRINGUTIL_HPP
#define REVERSE_KRANKS_STRINGUTIL_HPP

namespace ReverseMIPS {
    static void _split(const std::string &s, char delim,
                       std::vector<std::string> &elems) {
        std::stringstream ss(s);
        std::string item;

        while (std::getline(ss, item, delim)) {
            elems.push_back(item);
        }
    }

    std::vector<std::string> split(const std::string &s, char delim) {
        std::vector<std::string> elems;
        _split(s, delim, elems);
        return elems;
    }
}

#endif //REVERSE_KRANKS_STRINGUTIL_HPP
