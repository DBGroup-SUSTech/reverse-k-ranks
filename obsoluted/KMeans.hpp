//
// Created by BianZheng on 2022/1/20.
//

#ifndef REVERSE_KRANKS_KMEANS_HPP
#define REVERSE_KRANKS_KMEANS_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"
#include <fstream>
#include <algorithm>
#include <cfloat>
#include <armadillo>

namespace ReverseMIPS::KMeans {

    std::vector<int> Build(VectorMatrix &user, int n_merge_user) {
        arma::mat data(user.getRawData(), user.vec_dim_, user.n_vector_, false, true);
        arma::mat means;

        // Reference: http://arma.sourceforge.net/docs.html#kmeans
        bool status = arma::kmeans(means, data, n_merge_user, arma::static_spread, 50, true);
        if (!status) {
            std::cout << "clustering failed" << std::endl;
            exit(-1);
        }

//    means.print("means:");

        std::vector<std::vector<double>> centroid(n_merge_user, std::vector<double>(user.vec_dim_));
#pragma omp parallel for default(none) shared(n_merge_user, user, means, centroid)
        for (int i = 0; i < n_merge_user; i++) {
            for (int j = 0; j < user.vec_dim_; j++) {
                centroid[i][j] = means(j, i);
            }
        }

        std::vector<int> labels(user.n_vector_);
#pragma omp parallel for default(none) shared(user, n_merge_user, centroid, labels)
        for (int userID = 0; userID < user.n_vector_; userID++) {
            auto min_dist = DBL_MAX;
            int min_dist_idx = -1;
            for (int clsID = 0; clsID < n_merge_user; clsID++) {
                double dist = InnerProduct(centroid[clsID].data(), user.getVector(userID), user.vec_dim_);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_idx = clsID;
                }
            }
            labels[userID] = min_dist_idx;
        }

        return labels;
    }

}
#endif //REVERSE_KRANKS_KMEANS_HPP
