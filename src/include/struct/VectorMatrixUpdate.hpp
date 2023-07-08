//
// Created by bianzheng on 2023/7/5.
//

#ifndef REVERSE_KRANKS_VECTORMATRIXUPDATE_HPP
#define REVERSE_KRANKS_VECTORMATRIXUPDATE_HPP

#include <cmath>
#include <memory>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <iostream>
#include <unordered_set>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {
    class VectorMatrixUpdate {
        std::unique_ptr<float[]> rawData_;
    public:
        int n_vector_;
        int vec_dim_;

        VectorMatrixUpdate() {
            this->rawData_ = nullptr;
            this->n_vector_ = 0;
            this->vec_dim_ = 0;
        }

        ~VectorMatrixUpdate() = default;

        [[nodiscard]] float *getVector(const int vec_idx) const {
            return rawData_.get() + vec_idx * vec_dim_;
        }

        [[nodiscard]] float *getVector(const int vec_idx, const int offset) const {
            return rawData_.get() + vec_idx * vec_dim_ + offset;
        }

        void init(std::unique_ptr<float[]> &rawData, const int n_vector, const int vec_dim) {
            this->rawData_ = std::move(rawData);
            this->n_vector_ = n_vector;
            this->vec_dim_ = vec_dim;
        }

        VectorMatrixUpdate &operator=(VectorMatrixUpdate &&other) noexcept {
            this->rawData_ = std::move(other.rawData_);
            this->n_vector_ = other.n_vector_;
            this->vec_dim_ = other.vec_dim_;
            return *this;
        }

        [[nodiscard]] float *getRawData() const {
            return this->getVector(0);
        }

        void vectorNormalize() {
#pragma omp parallel for default(none)
            for (int i = 0; i < n_vector_; i++) {
                float l2norm = 0;
                for (int j = 0; j < vec_dim_; j++) {
                    l2norm += rawData_[i * vec_dim_ + j] * rawData_[i * vec_dim_ + j];
                }
                l2norm = std::sqrt(l2norm);
                if (l2norm <= 0.001) {
                    continue;
                }
                for (int j = 0; j < vec_dim_; j++) {
                    rawData_[i * vec_dim_ + j] /= l2norm;
                }
            }
        }

        void insert(const VectorMatrixUpdate &vm) {
            int n_vector = this->n_vector_ + vm.n_vector_;
            int vec_dim = this->vec_dim_;
            std::unique_ptr<float[]> new_rawData = std::make_unique<float[]>((size_t) n_vector * vec_dim);
            std::memcpy(new_rawData.get(), this->rawData_.get(), sizeof(float) * this->n_vector_ * vec_dim);
            std::memcpy(new_rawData.get() + (size_t) this->n_vector_ * vec_dim, vm.rawData_.get(),
                        sizeof(float) * vm.n_vector_ * vec_dim);
            this->rawData_ = std::move(new_rawData);
            this->n_vector_ = n_vector;
            this->vec_dim_ = vec_dim;
        }

        void remove(const std::vector<int> &itemID_l) {
            std::unordered_set<int> itemID_s(itemID_l.begin(), itemID_l.end());

            std::vector<int> remain_itemID_l;
            for (int vecsID = 0; vecsID < this->n_vector_; vecsID++) {
                if (itemID_s.find(vecsID) == itemID_s.end()) {
                    remain_itemID_l.emplace_back(vecsID);
                }
            }
            assert(remain_itemID_l.size() + itemID_l.size() == this->n_vector_);

            int n_vector = remain_itemID_l.size();
            int vec_dim = this->vec_dim_;
            std::unique_ptr<float[]> new_rawData = std::make_unique<float[]>((size_t) n_vector * vec_dim);
            for (int i = 0; i < n_vector; i++) {
                const int itemID = remain_itemID_l[i];
                assert(0 <= itemID && itemID < this->n_vector_);
                std::memcpy(new_rawData.get() + (size_t) i * vec_dim,
                            this->rawData_.get() + (size_t) itemID * vec_dim,
                            sizeof(float) * vec_dim);
            }

            this->rawData_ = std::move(new_rawData);
            this->n_vector_ = n_vector;
            this->vec_dim_ = vec_dim;
        }

    };
}
#endif //REVERSE_KRANKS_VECTORMATRIXUPDATE_HPP
