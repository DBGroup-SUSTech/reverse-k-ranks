//
// Created by BianZheng on 2022/11/3.
//

#ifndef REVERSE_K_RANKS_UNIFORMSAMPLE_HPP
#define REVERSE_K_RANKS_UNIFORMSAMPLE_HPP

#include <spdlog/spdlog.h>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <iostream>

namespace ReverseMIPS {

    void FindUniformSample(const int &n_data_item, const int &n_sample, std::vector<int> &sample_rank_l) {
        assert(sample_rank_l.size() == n_sample);

        const int end_sample_rank = (int) n_data_item - 1;
        const double delta = (end_sample_rank - 0) * 1.0 / n_sample;
        for (int sampleID = 0; sampleID < n_sample; sampleID++) {
            sample_rank_l[sampleID] = std::floor(sampleID * delta);
        }
        std::sort(sample_rank_l.data(), sample_rank_l.data() + n_sample);
        assert(0 <= sample_rank_l[0] && sample_rank_l[0] < n_data_item);
        int max_sample_every = 0;
        for (int sampleID = 1; sampleID < n_sample; sampleID++) {
            assert(0 <= sample_rank_l[sampleID] && sample_rank_l[sampleID] < n_data_item);
            assert(sample_rank_l[sampleID - 1] < sample_rank_l[sampleID]);
            max_sample_every = std::max(max_sample_every,
                                        sample_rank_l[sampleID] - sample_rank_l[sampleID - 1]);
        }
        const int max_sample_every_ = max_sample_every;
        assert(max_sample_every_ <= std::ceil(1.0 * n_data_item / n_sample));

        std::cout << "first 50 rank: ";
        const int end_rankID = std::min((int) n_sample, 50);
        for (int rankID = 0; rankID < end_rankID; rankID++) {
            std::cout << sample_rank_l[rankID] << " ";
        }
        std::cout << std::endl;
        std::cout << "last 50 rank: ";
        const int start_rankID = std::max(0, (int) n_sample - 50);
        for (int rankID = start_rankID; rankID < n_sample; rankID++) {
            std::cout << sample_rank_l[rankID] << " ";
        }
        std::cout << std::endl;

//        for (int rankID = 0; rankID < n_sample; rankID++) {
//            std::cout << sample_rank_l[rankID] << " ";
//        }
//        std::cout << std::endl;

        spdlog::info("uniform sample: max_sample_every {}, n_sample {}", max_sample_every_, n_sample);
    }
}
#endif //REVERSE_K_RANKS_UNIFORMSAMPLE_HPP
