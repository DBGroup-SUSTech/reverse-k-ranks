//
// Created by bianzheng on 2023/5/30.
//

#ifndef REVERSE_KRANKS_RUNGRIDINDEXDOUBLE_HPP
#define REVERSE_KRANKS_RUNGRIDINDEXDOUBLE_HPP

#include "alg/Grid.hpp"
#include "alg/GridDouble.hpp"
#include "alg/SpaceInnerProduct.hpp"

namespace ReverseMIPS {

    void RunGridIndex(const std::unique_ptr<double[]> &user_ptr, const std::unique_ptr<double[]> &data_item_ptr,
                      const size_t &n_user, const size_t &n_data_item, const size_t vec_dim,
                      double &run_index_time) {
        std::unique_ptr<GridDouble> IPbound_ptr;

        const int min_codeword = std::floor(std::sqrt(1.0 * 80 * std::sqrt(3 * vec_dim)));
        int n_codeword = 1;
        while (n_codeword < min_codeword) {
            n_codeword = n_codeword << 1;
        }
        spdlog::info("GridIndex min_codeword {}, codeword {}", min_codeword, n_codeword);

        IPbound_ptr = std::make_unique<GridDouble>(n_user, n_data_item, vec_dim, n_codeword);
        IPbound_ptr->Preprocess(user_ptr.get(), data_item_ptr.get());

        double result = 0;

        TimeRecord record;
        record.reset();
//#pragma omp parallel for default(none) shared(n_sample_user, n_data_item, user, data_item, IPbound_ptr)
        for (int userID = 0; userID < n_user; userID++) {
            for (int itemID = 0; itemID < n_data_item; itemID++) {
                const double IP_lb = IPbound_ptr->IPLowerBound(userID, itemID);
                result += IP_lb;
            }
        }

        run_index_time = record.get_elapsed_time_second();

        printf("GridIndex result value %.3f\n", result);
    }

    void
    RunGridIndexNoNegative(const std::unique_ptr<double[]> &user_ptr, const std::unique_ptr<double[]> &data_item_ptr,
                           const size_t &n_user, const size_t &n_data_item, const size_t vec_dim,
                           double &run_index_time) {
        std::unique_ptr<GridDouble> IPbound_ptr;

        const int min_codeword = std::floor(std::sqrt(1.0 * 80 * std::sqrt(3 * vec_dim)));
        int n_codeword = 1;
        while (n_codeword < min_codeword) {
            n_codeword = n_codeword << 1;
        }
        spdlog::info("GridIndex min_codeword {}, codeword {}", min_codeword, n_codeword);

        IPbound_ptr = std::make_unique<GridDouble>(n_user, n_data_item, vec_dim, n_codeword);
        IPbound_ptr->Preprocess(user_ptr.get(), data_item_ptr.get());

        double result = 0;

        TimeRecord record;
        record.reset();
//#pragma omp parallel for default(none) shared(n_sample_user, n_data_item, user, data_item, IPbound_ptr)
        for (int userID = 0; userID < n_user; userID++) {
            for (int itemID = 0; itemID < n_data_item; itemID++) {
                const double IP_lb = IPbound_ptr->IPLowerBoundNoNegative(userID, itemID);
                result += IP_lb;
            }
        }

        run_index_time = record.get_elapsed_time_second();

        printf("GridIndex result value %.3f\n", result);
    }

    void RunInnerProduct(const std::unique_ptr<double[]> &user_ptr, const std::unique_ptr<double[]> &data_item_ptr,
                         const size_t &n_user, const size_t &n_data_item, const size_t vec_dim,
                         double &run_index_time) {

        double result = 0;

        TimeRecord record;
        record.reset();
//#pragma omp parallel for default(none) shared(n_sample_user, n_data_item, user, data_item, vec_dim)
        for (int userID = 0; userID < n_user; userID++) {
            const double *user_vecs = user_ptr.get() + userID * vec_dim;
            for (int itemID = 0; itemID < n_data_item; itemID++) {
                const double *item_vecs = data_item_ptr.get() + itemID * vec_dim;
                const double ip = InnerProduct(user_vecs, item_vecs, vec_dim);
                result += ip;
            }
        }

        run_index_time = record.get_elapsed_time_second();
        printf("InnerProduct result value %.3f\n", result);

    }

}
#endif //REVERSE_KRANKS_RUNGRIDINDEXDOUBLE_HPP
