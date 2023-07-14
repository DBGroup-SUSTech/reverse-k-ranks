//
// Created by bianzheng on 2023/6/16.
//

#ifndef REVERSE_KRANKS_SVDUPDATE_HPP
#define REVERSE_KRANKS_SVDUPDATE_HPP

#include <cassert>
#include <filesystem>
#include <armadillo>
#include <memory>

#define ARMA_ALLOW_FAKE_GCC

#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrixUpdate.hpp"

namespace ReverseMIPS {
    class SVDUpdate {

        VectorMatrixUpdate transfer_data_item_;
    public:
        size_t n_user_, n_data_item_, vec_dim_;
        float SIGMA_;
        int check_dim_;
        VectorMatrixUpdate transfer_item_, transfer_user_;
        std::vector<float> data_item_cache_;

        SVDUpdate() = default;

        SVDUpdate(const VectorMatrixUpdate &user, const VectorMatrixUpdate &data_item, const float &SIGMA) {
            this->n_user_ = user.n_vector_;
            this->n_data_item_ = data_item.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->SIGMA_ = SIGMA;
            PerformSVD(user, data_item, SIGMA);
        };

        SVDUpdate(const char *index_basic_dir, const char *dataset_name) {
            LoadIndex(index_basic_dir, dataset_name);
        }

        void TransferItem(float *item_vecs, const int vec_dim) {
#pragma omp parallel for default(none) shared(item_vecs, vec_dim)
            for (int trans_dim = 0; trans_dim < vec_dim; trans_dim++) {
                float *transfer_vecs = transfer_item_.getVector(trans_dim);
                data_item_cache_[trans_dim] = InnerProduct(transfer_vecs, item_vecs, vec_dim);
            }
            memcpy(item_vecs, data_item_cache_.data(), vec_dim * sizeof(float));

        }

        void TransferQuery(const float *query_vecs, const int &vec_dim, float *query_write_vecs) {
#pragma omp parallel for default(none) shared(query_vecs, vec_dim)
            for (int trans_dim = 0; trans_dim < vec_dim; trans_dim++) {
                float *transfer_vecs = transfer_item_.getVector(trans_dim);
                data_item_cache_[trans_dim] = InnerProduct(transfer_vecs, query_vecs, vec_dim);
            }
            memcpy(query_write_vecs, data_item_cache_.data(), vec_dim * sizeof(float));

        }

        void InsertUser(const VectorMatrixUpdate &original_user, const VectorMatrixUpdate &insert_user,
                        const VectorMatrixUpdate &data_item) {
            const int n_original_user = original_user.n_vector_;
            const int n_insert_user = insert_user.n_vector_;
            std::unique_ptr<float[]> new_user_ptr = std::make_unique<float[]>(
                    (size_t) (n_original_user + n_insert_user) * vec_dim_);
            std::memcpy(new_user_ptr.get(), original_user.getRawData(),
                        sizeof(float) * n_original_user * vec_dim_);
            std::memcpy(new_user_ptr.get() + (size_t) n_original_user * vec_dim_, insert_user.getRawData(),
                        sizeof(float) * n_insert_user * vec_dim_);
            VectorMatrixUpdate new_user;
            new_user.init(new_user_ptr, n_original_user + n_insert_user, vec_dim_);
            assert(vec_dim_ == original_user.vec_dim_ && vec_dim_ == insert_user.vec_dim_);

            n_user_ += n_insert_user;
            PerformSVD(new_user, data_item, SIGMA_);
        }

        void DeleteUser(const std::vector<int> &del_userID_l) {
            transfer_user_.remove(del_userID_l);
            n_user_ -= del_userID_l.size();
        }

        void PerformSVD(const VectorMatrixUpdate &user, const VectorMatrixUpdate &data_item, const float &SIGMA) {
            const int vec_dim = user.vec_dim_; // p->colNum, m
            const int n_user = user.n_vector_; // p->rowNum, n
            const int n_data_item = data_item.n_vector_;
            data_item_cache_.resize(vec_dim);

            std::unique_ptr<float[]> transfer_ptr = std::make_unique<float[]>(vec_dim * vec_dim);
            transfer_item_.init(transfer_ptr, vec_dim, vec_dim);
            assert(transfer_item_.n_vector_ == transfer_item_.vec_dim_);

            std::unique_ptr<float[]> user_ptr = std::make_unique<float[]>((size_t) n_user * vec_dim);
            transfer_user_.init(user_ptr, n_user, vec_dim);

            std::unique_ptr<float[]> data_item_ptr = std::make_unique<float[]>((size_t) n_data_item * vec_dim);
            transfer_data_item_.init(data_item_ptr, n_data_item, vec_dim);

            //Q is item, since a new query would be added
            //U is user, since user matrix would not change
            arma::fmat P_t(user.getRawData(), user.vec_dim_, user.n_vector_, false, true);

            arma::fmat U_t;
            arma::fvec s;
            arma::fmat V;

            // see: http://arma.sourceforge.net/docs.html#svd_econ
            //	svd_econ(U_t, s, V, P_t, "both", "std");
            arma::svd_econ(U_t, s, V, P_t, "both", "std"); // P = U * sigma * V_t

            U_t = U_t.t();

            float *uData = transfer_item_.getRawData();

            for (int rowIndex = 0; rowIndex < vec_dim; rowIndex++) {
                for (int colIndex = 0; colIndex < vec_dim; colIndex++) {
                    uData[rowIndex * vec_dim + colIndex] = s[rowIndex] * U_t(rowIndex, colIndex);
                }
            }

            for (int rowIndex = 0; rowIndex < n_user; rowIndex++) {
                for (int colIndex = 0; colIndex < vec_dim; colIndex++) {
                    transfer_user_.getRawData()[rowIndex * vec_dim + colIndex] = V(rowIndex, colIndex);
                }
            }

            for (int itemID = 0; itemID < n_data_item; itemID++) {
                TransferItem(transfer_data_item_.getVector(itemID), vec_dim);
            }

            std::vector<float> sum(vec_dim);
            sum[0] = s[0];
            for (int colIndex = 1; colIndex < vec_dim; colIndex++) {
                sum[colIndex] = sum[colIndex - 1] + s[colIndex];
            }

            int check_dim = 0;
            for (int colIndex = 0; colIndex < vec_dim; colIndex++) {
                if (sum[colIndex] / sum[vec_dim - 1] >= SIGMA) {
                    check_dim = colIndex;
                    break;
                }
            }
            this->check_dim_ = check_dim;
        }

        void SaveIndex(const char *index_basic_dir, const char *dataset_name) {

            char index_path[256];
            sprintf(index_path,
                    "%s/svd_index/%s.index",
                    index_basic_dir, dataset_name);

            std::ofstream out_stream_ = std::ofstream(index_path, std::ios::binary | std::ios::out);
            if (!out_stream_) {
                spdlog::error("error in write result, not found index");
                exit(-1);
            }
            out_stream_.write((char *) &n_user_, sizeof(size_t));
            out_stream_.write((char *) &n_data_item_, sizeof(size_t));
            out_stream_.write((char *) &vec_dim_, sizeof(size_t));
            out_stream_.write((char *) &SIGMA_, sizeof(float));
            out_stream_.write((char *) &check_dim_, sizeof(int));

            out_stream_.write((char *) transfer_item_.getRawData(),
                              (int64_t) (vec_dim_ * vec_dim_ * sizeof(float)));
            out_stream_.write((char *) transfer_user_.getRawData(),
                              (int64_t) (n_user_ * vec_dim_ * sizeof(float)));
            out_stream_.write((char *) transfer_data_item_.getRawData(),
                              (int64_t) (n_data_item_ * vec_dim_ * sizeof(float)));

            out_stream_.close();
        }

        void LoadIndex(const char *index_basic_dir, const char *dataset_name) {
            char index_path[256];
            sprintf(index_path,
                    "%s/svd_index/%s.index",
                    index_basic_dir, dataset_name);
            spdlog::info("index path {}", index_path);

            std::ifstream index_stream = std::ifstream(index_path, std::ios::binary | std::ios::in);
            if (!index_stream) {
                spdlog::error("error in reading index");
                exit(-1);
            }

            index_stream.read((char *) &n_user_, sizeof(size_t));
            index_stream.read((char *) &n_data_item_, sizeof(size_t));
            index_stream.read((char *) &vec_dim_, sizeof(size_t));
            index_stream.read((char *) &SIGMA_, sizeof(float));
            index_stream.read((char *) &check_dim_, sizeof(int));

            std::unique_ptr<float[]> transfer_item_ptr = std::make_unique<float[]>(vec_dim_ * vec_dim_);
            index_stream.read((char *) transfer_item_ptr.get(), (int64_t) (sizeof(float) * vec_dim_ * vec_dim_));
            transfer_item_.init(transfer_item_ptr, vec_dim_, vec_dim_);

            std::unique_ptr<float[]> user_ptr = std::make_unique<float[]>(n_user_ * vec_dim_);
            index_stream.read((char *) user_ptr.get(), (int64_t) (sizeof(float) * n_user_ * vec_dim_));
            transfer_user_.init(user_ptr, n_user_, vec_dim_);

            std::unique_ptr<float[]> data_item_ptr = std::make_unique<float[]>(n_data_item_ * vec_dim_);
            index_stream.read((char *) data_item_ptr.get(), (int64_t) (sizeof(float) * n_data_item_ * vec_dim_));
            transfer_data_item_.init(data_item_ptr, n_data_item_, vec_dim_);

            data_item_cache_.resize(vec_dim_);

            index_stream.close();
        }

        uint64_t IndexSizeByte() const {
            const uint64_t vector_matrix_size =
                    3 * sizeof(int) * 2 + sizeof(float) * vec_dim_ * vec_dim_ +
                    sizeof(float) * n_user_ * vec_dim_ +
                    sizeof(float) * n_data_item_ * vec_dim_;
            const uint64_t cache_size = sizeof(float) * vec_dim_;
            const uint64_t single_variable_size = sizeof(int) * 4 + sizeof(float);
            return vector_matrix_size + cache_size + single_variable_size;
        }

    };


}
#endif //REVERSE_KRANKS_SVDUPDATE_HPP
