//
// Created by BianZheng on 2022/7/26.
//

#ifndef REVERSE_K_RANKS_TOPKMAXHEAP_HPP
#define REVERSE_K_RANKS_TOPKMAXHEAP_HPP

#include <vector>
#include <cassert>
#include <algorithm>

namespace ReverseMIPS {
    class TopkMaxHeap {
        int topk_, cur_size_, heap_max_rank_;
        std::vector<int> topk_heap_;
    public:
        TopkMaxHeap() = default;

        TopkMaxHeap(const int &topk) {
            this->topk_ = topk;
            this->cur_size_ = 0;
            this->heap_max_rank_ = -1;
            this->topk_heap_.resize(topk);
        }

        void Update(const int &new_rank) {
            if (cur_size_ < topk_) {
                topk_heap_[cur_size_] = new_rank;
                cur_size_++;
                if (cur_size_ == topk_) {
                    std::make_heap(topk_heap_.begin(), topk_heap_.end(), std::less());
                    heap_max_rank_ = topk_heap_.front();
                }
            } else {
                assert(cur_size_ == topk_);
                if (heap_max_rank_ > new_rank) {
                    std::pop_heap(topk_heap_.begin(), topk_heap_.end(), std::less());
                    topk_heap_[topk_ - 1] = new_rank;
                    std::push_heap(topk_heap_.begin(), topk_heap_.end(), std::less());
                    heap_max_rank_ = topk_heap_.front();
                    assert(topk_heap_[0] == topk_heap_.front());
                }
            }
            assert(cur_size_ <= topk_);

        }

        int Front() {
            if (cur_size_ < topk_) {
                return -1;
            } else {
                assert(cur_size_ == topk_);
                return this->heap_max_rank_;
            }
        }

        void Reset() {
            this->cur_size_ = 0;
            this->heap_max_rank_ = -1;
        }
    };
}
#endif //REVERSE_K_RANKS_TOPKMAXHEAP_HPP
