//
// Created by bianzheng on 2023/5/23.
//

#ifndef REVERSE_KRANKS_RUNGRIDINDEX_HPP
#define REVERSE_KRANKS_RUNGRIDINDEX_HPP

#include "alg/Grid.hpp"
#include "alg/SpaceInnerProduct.hpp"

namespace ReverseMIPS {

    void RunGridIndex(const VectorMatrix &user, const VectorMatrix &data_item,
                      const int &n_sample_user,
                      double &run_index_time) {
        std::unique_ptr<Grid> IPbound_ptr;

        int n_user = user.n_vector_;
        int n_data_item = data_item.n_vector_;
        int vec_dim = user.vec_dim_;

        const int min_codeword = std::floor(std::sqrt(1.0 * 80 * std::sqrt(3 * user.vec_dim_)));
        int n_codeword = 1;
        while (n_codeword < min_codeword) {
            n_codeword = n_codeword << 1;
        }
        spdlog::info("GridIndex min_codeword {}, codeword {}", min_codeword, n_codeword);

        IPbound_ptr = std::make_unique<Grid>(n_user, n_data_item, vec_dim, n_codeword);
        IPbound_ptr->Preprocess(user, data_item);

        float result = 0;

        TimeRecord record;
        record.reset();
//#pragma omp parallel for default(none) shared(n_sample_user, n_data_item, user, data_item, IPbound_ptr)
        for (int userID = 0; userID < n_sample_user; userID++) {
            for (int itemID = 0; itemID < n_data_item; itemID++) {
                const float IP_lb = IPbound_ptr->IPLowerBound(userID, itemID);
                result += IP_lb;
            }
        }

        run_index_time = record.get_elapsed_time_second();

        printf("GridIndex result value %.3f\n", result);
    }

    void RunGridIndexNoNegative(const VectorMatrix &user, const VectorMatrix &data_item,
                                const int &n_sample_user,
                                double &run_index_time) {
        std::unique_ptr<Grid> IPbound_ptr;

        int n_user = user.n_vector_;
        int n_data_item = data_item.n_vector_;
        int vec_dim = user.vec_dim_;

        const int min_codeword = std::floor(std::sqrt(1.0 * 80 * std::sqrt(3 * user.vec_dim_)));
        int n_codeword = 1;
        while (n_codeword < min_codeword) {
            n_codeword = n_codeword << 1;
        }
        spdlog::info("GridIndex min_codeword {}, codeword {}", min_codeword, n_codeword);

        IPbound_ptr = std::make_unique<Grid>(n_user, n_data_item, vec_dim, n_codeword);
        IPbound_ptr->Preprocess(user, data_item);

        float result = 0;

        TimeRecord record;
        record.reset();
//#pragma omp parallel for default(none) shared(n_sample_user, n_data_item, user, data_item, IPbound_ptr)
        for (int userID = 0; userID < n_sample_user; userID++) {
            for (int itemID = 0; itemID < n_data_item; itemID++) {
                const float IP_lb = IPbound_ptr->IPLowerBoundNoNegative(userID, itemID);
                result += IP_lb;
            }
        }

        run_index_time = record.get_elapsed_time_second();

        printf("GridIndex result value %.3f\n", result);
    }

    void RunInnerProduct(const VectorMatrix &user, const VectorMatrix &data_item,
                         const int &n_sample_user,
                         double &run_index_time) {
        std::unique_ptr<Grid> IPbound_ptr;

        int n_user = user.n_vector_;
        int n_data_item = data_item.n_vector_;
        int vec_dim = user.vec_dim_;

        float result = 0;

        TimeRecord record;
        record.reset();
//#pragma omp parallel for default(none) shared(n_sample_user, n_data_item, user, data_item, vec_dim)
        for (int userID = 0; userID < n_sample_user; userID++) {
            const float *user_vecs = user.getVector(userID);
            for (int itemID = 0; itemID < n_data_item; itemID++) {
                const float *item_vecs = data_item.getVector(itemID);
                const float ip = InnerProduct(user_vecs, item_vecs, vec_dim);
                result += ip;
            }
        }

        run_index_time = record.get_elapsed_time_second();
        printf("InnerProduct result value %.3f\n", result);

    }

}
#endif //REVERSE_KRANKS_RUNGRIDINDEX_HPP
