//
// Created by BianZheng on 2022/9/9.
//

#ifndef REVERSE_KRANKS_SIMPFERDATA_HPP
#define REVERSE_KRANKS_SIMPFERDATA_HPP

#include <cfloat>
#include <vector>
#include <map>
#include <cmath>

namespace ReverseMIPS {
    // definition of SimpferData
    class SimpferData {
    public:

        unsigned int ID_ = 0;
        std::vector<float> vec_;
        float norm_ = 0;

        std::map<float, unsigned int, std::greater<>> topk_;
        float threshold_ = -FLT_MAX;
        std::vector<float> lowerbound_array_;

        unsigned int block_id = 0;

        // constructor
        SimpferData() {}

        // norm computation
        void ComputeNorm() {

            for (unsigned int i = 0; i < vec_.size(); ++i) {
                norm_ += vec_[i] * vec_[i];
            }
            norm_ = std::sqrt(norm_);
        }

        // k-MIPS update
        void UpdateTopk(const float ip, const unsigned int id, const unsigned int k_) {

            if (ip > threshold_) {
                topk_.insert({ip, id});
            }

            if (topk_.size() > k_) {
                auto it = topk_.end();
                --it;
                topk_.erase(it);
            }

            if (topk_.size() == k_) {
                auto it = topk_.end();
                --it;
                threshold_ = it->first;
            }
        }

        // make lower-bound array
        void make_lb_array() {

            auto it = topk_.begin();
            while (it != topk_.end()) {
                lowerbound_array_.push_back(it->first);
                ++it;
            }
            assert(lowerbound_array_.size() == topk_.size());
        }

        // init
        void init() {

            // clear top-k in pre-processing
            topk_.clear();

            // clear threshold in pre-processing
            threshold_ = -FLT_MAX;
        }

        bool operator<(const SimpferData &d) const { return norm_ < d.norm_; }

        bool operator>(const SimpferData &d) const { return norm_ > d.norm_; }
    };

    // definition of SimpferBlock
    class SimpferBlock {
    public:
        unsigned int identifier = 0;
        std::vector<int> userID_l;
        std::vector<float> lowerbound_array;

        SimpferBlock() = default;

        // constructor
        SimpferBlock(const int &k_max) {
            userID_l.clear();

            lowerbound_array.resize(k_max);
            for (unsigned int i = 0; i < k_max; ++i) {
                lowerbound_array[i] = FLT_MAX;
            }
        }

        // init
        void init(const int &k_max) {

            // increment identifier
            ++identifier;

            // clear member
            userID_l.clear();

            // init array
            for (unsigned int i = 0; i < k_max; ++i) {
                lowerbound_array[i] = FLT_MAX;
            }
        }

        // make lower-bound array
        void UpdateLowerboundArray(const std::vector<SimpferData> &user_sd_l, const int &k_max) {

            for (unsigned int i = 0; i < k_max; ++i) {
                for (unsigned int j = 0; j < userID_l.size(); ++j) {
                    const int userID = userID_l[j];
                    if (lowerbound_array[i] > user_sd_l[userID].lowerbound_array_[i]) {
                        lowerbound_array[i] = user_sd_l[userID].lowerbound_array_[i];
                    }
                }
            }
        }
    };

}
#endif //REVERSE_KRANKS_SIMPFERDATA_HPP
