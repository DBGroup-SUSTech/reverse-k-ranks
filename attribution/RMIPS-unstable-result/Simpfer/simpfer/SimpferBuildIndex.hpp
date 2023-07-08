//
// Created by BianZheng on 2022/9/9.
//

#ifndef REVERSE_KRANKS_SIMPFERBUILDINDEX_HPP
#define REVERSE_KRANKS_SIMPFERBUILDINDEX_HPP

#include "struct/VectorMatrix.hpp"
#include "SimpferData.hpp"

namespace ReverseMIPS {
    void TransformData(const VectorMatrix &vm, std::vector<SimpferData> &simpfer_data_l) {

        const int n_vector = vm.n_vector_;
        const int vec_dim = vm.vec_dim_;

        SimpferData simpfer_data;
        for (int vecsID = 0; vecsID < n_vector; vecsID++) {
            const float *vecs = vm.getVector(vecsID);
            std::vector<float> vector_user(vec_dim);
            for (int dim = 0; dim < vec_dim; dim++) {
                vector_user[dim] = vecs[dim];
            }
            simpfer_data.vec_ = vector_user;
            simpfer_data_l.push_back(simpfer_data);
            ++simpfer_data.ID_;
        }

    }

    void ComputeNorm(std::vector<SimpferData> &simpfer_data_l) {
        const int n_data = (int) simpfer_data_l.size();
        for (int dataID = 0; dataID < n_data; dataID++) {
            simpfer_data_l[dataID].ComputeNorm();
        }
    }

    float InnerProduct(const SimpferData &q, const SimpferData &p, const int &vec_dim) {

        float ip = 0;
        for (unsigned int dim = 0; dim < vec_dim; ++dim) {
            ip += q.vec_[dim] * p.vec_[dim];
        }

        return ip;
    }

    // lower-bound computation
    void ComputeLowerbound(std::vector<SimpferData> &user_sd_l,
                           std::vector<SimpferData> &item_sd_l,
                           const int &n_user, const int &n_data_item,
                           const int &k_max, const int &n_compute_coe,
                           const int &vec_dim) {
        assert(n_user == user_sd_l.size());
        assert(n_data_item == item_sd_l.size());

        const unsigned int compute_n_item = k_max * n_compute_coe;
        for (unsigned int userID = 0; userID < n_user; ++userID) {
            for (unsigned int max_norm_itemID = 0; max_norm_itemID < compute_n_item; ++max_norm_itemID) {
                // ip comp.
                const float ip = InnerProduct(user_sd_l[userID].vec_.data(), item_sd_l[max_norm_itemID].vec_.data(),
                                               vec_dim);
                // update top-k
                user_sd_l[userID].UpdateTopk(ip, item_sd_l[max_norm_itemID].ID_, k_max);
            }
            // convert map to array
            user_sd_l[userID].make_lb_array();
        }

    }

    // blocking
    void Blocking(std::vector<SimpferData> &user_sd_l,
                  std::vector<SimpferBlock> &block_set,
                  const int &n_user,
                  const int &k_max) {

        // determine size
        const unsigned int block_size = (unsigned int) (log2(n_user) * 2);

        // make block
        SimpferBlock blk(k_max);

        // block assignment
        for (unsigned int userID = 0; userID < n_user; ++userID) {
            // assign block id
            user_sd_l[userID].block_id = blk.identifier;
            // insert into block
            blk.userID_l.push_back((int) userID);
            // init blk
            if (blk.userID_l.size() == block_size || userID == n_user - 1) {
                // update lower-bound array
                blk.UpdateLowerboundArray(user_sd_l, k_max);
                // insert into set
                block_set.push_back(blk);
                // init
                blk.init(k_max);
            }
        }

    }

}
#endif //REVERSE_KRANKS_SIMPFERBUILDINDEX_HPP
