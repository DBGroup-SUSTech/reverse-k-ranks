//
// Created by BianZheng on 2022/2/20.
//

#ifndef REVERSE_KRANKS_VECTORMATRIX_HPP
#define REVERSE_KRANKS_VECTORMATRIX_HPP

#include <cmath>
#include <memory>

namespace ReverseMIPS {
    class VectorMatrix {
        std::unique_ptr<float[]> rawData_;
    public:
        int n_vector_;
        int vec_dim_;

        VectorMatrix() {
            this->rawData_ = nullptr;
            this->n_vector_ = 0;
            this->vec_dim_ = 0;
        }

        ~VectorMatrix() = default;

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

        VectorMatrix &operator=(VectorMatrix &&other) noexcept {
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

    };
}
#endif //REVERSE_KRANKS_VECTORMATRIX_HPP