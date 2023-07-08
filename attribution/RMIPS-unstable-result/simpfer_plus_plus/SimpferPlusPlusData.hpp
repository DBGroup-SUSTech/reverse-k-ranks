//
// Created by bianzheng on 2023/5/3.
//

#ifndef REVERSE_KRANKS_SIMPFERPLUSPLUSDATA_HPP
#define REVERSE_KRANKS_SIMPFERPLUSPLUSDATA_HPP

#include <vector>
#include <functional>
#include <algorithm>
#include <map>
#include <cfloat>
#include <eigen3/Eigen/Core>

namespace ReverseMIPS {

// definition of data
    class SimpferPlusPlusData {
    public:

        unsigned int identifier = 0;
        std::vector<float> vec;
        float norm = 0;

        std::map<float, unsigned int, std::greater<float>> topk;
        float threshold = -FLT_MAX;
        std::vector<float> lowerbound_array;

        unsigned int block_id = 0;

        // constructor
        SimpferPlusPlusData() {}

        // norm computation
        void norm_computation() {
            for (unsigned int i = 0; i < vec.size(); ++i) norm += vec[i] * vec[i];
            norm = sqrt(norm);
        }

        // k-MIPS update
        void update_topk(const float ip, const unsigned int id, const unsigned int k_) {
            if (ip > threshold) topk.insert({ip, id});

            if (topk.size() > k_) {
                auto it = topk.end();
                --it;
                topk.erase(it);
            }

            if (topk.size() == k_) {
                auto it = topk.end();
                --it;
                threshold = it->first;
            }
        }

        // make lower-bound array
        void make_lb_array() {
            auto it = topk.begin();
            while (it != topk.end()) {
                lowerbound_array.push_back(it->first);
                ++it;
            }
        }

        // init
        void init() {
            // clear top-k in pre-processing
            topk.clear();

            // clear threshold in pre-processing
            threshold = 0;
        }

        bool operator<(const SimpferPlusPlusData &d) const { return norm < d.norm; }

        bool operator>(const SimpferPlusPlusData &d) const { return norm > d.norm; }
    };


// definition of block
    class SimpferPlusPlusBlock {
    public:

        unsigned int identifier = 0;
        std::vector<SimpferPlusPlusData *> member;
        std::vector<float> lowerbound_array;
        Eigen::MatrixXf M;

        // constructor
        SimpferPlusPlusBlock(const int &k_max) {
            lowerbound_array.resize(k_max);
            for (unsigned int i = 0; i < k_max; ++i) lowerbound_array[i] = FLT_MAX;
        }

        // init
        void init(const int &k_max) {
            // increment identifier
            ++identifier;

            // clear member
            member.clear();

            // init array
            for (unsigned int i = 0; i < k_max; ++i) lowerbound_array[i] = FLT_MAX;
        }

        // make lower-bound array
        void update_lowerbound_array(const int &k_max) {
            for (unsigned int i = 0; i < k_max; ++i) {
                for (unsigned int j = 0; j < member.size(); ++j) {
                    if (lowerbound_array[i] > member[j]->lowerbound_array[i])
                        lowerbound_array[i] = member[j]->lowerbound_array[i];
                }
            }
        }

        // make matrix
        void make_matrix(const int &vec_dim) {
            M = Eigen::MatrixXf::Zero(member.size(), vec_dim);
            for (unsigned int i = 0; i < member.size(); ++i) {
                for (unsigned int j = 0; j < vec_dim; ++j) M(i, j) = member[i]->vec[j];
            }
        }
    };

}
#endif //REVERSE_KRANKS_SIMPFERPLUSPLUSDATA_HPP
