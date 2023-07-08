//
// Created by bianzheng on 2023/5/3.
//

#ifndef REVERSE_KRANKS_SIMPFERPLUSPLUSBUILDINDEX_HPP
#define REVERSE_KRANKS_SIMPFERPLUSPLUSBUILDINDEX_HPP

#include "simpfer_plus_plus/SimpferPlusPlusData.hpp"

namespace ReverseMIPS {

    void TransformData(const VectorMatrix &vm, std::vector<SimpferPlusPlusData> &simpfer_data_l) {

        const int n_vector = vm.n_vector_;
        const int vec_dim = vm.vec_dim_;

        SimpferPlusPlusData simpfer_data;
        for (int vecsID = 0; vecsID < n_vector; vecsID++) {
            const float *vecs = vm.getVector(vecsID);
            std::vector<float> vector_user(vec_dim);
            for (int dim = 0; dim < vec_dim; dim++) {
                vector_user[dim] = vecs[dim];
            }
            simpfer_data.vec = vector_user;
            simpfer_data_l.push_back(simpfer_data);
            ++simpfer_data.identifier;
        }

    }

// ip computation
    float compute_ip(const SimpferPlusPlusData &q, const SimpferPlusPlusData &p, const int &vec_dim) {
        float ip = 0;
        for (unsigned int i = 0; i < vec_dim; ++i) ip += q.vec[i] * p.vec[i];
        return ip;
    }

// norm computation
    void compute_norm(std::vector<SimpferPlusPlusData> &user_sd_l,
                      std::vector<SimpferPlusPlusData> &item_sd_l) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

        // norm computation
        for (unsigned int i = 0; i < user_sd_l.size(); ++i) user_sd_l[i].norm_computation();
        for (unsigned int i = 0; i < item_sd_l.size(); ++i) item_sd_l[i].norm_computation();

        // sort by norm in descending order
        std::sort(user_sd_l.begin(), user_sd_l.end(), std::greater());
        std::sort(item_sd_l.begin(), item_sd_l.end(), std::greater());

        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        double time_norm_computation = (double) std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count();
        double time_pre_processing = time_norm_computation;
    }

// lower-bound computation
    void compute_lowerbound(std::vector<SimpferPlusPlusData> &user_sd_l,
                            std::vector<SimpferPlusPlusData> &item_sd_l, const int &k_max, const int &vec_dim) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

#pragma omp parallel for default(none) shared(user_sd_l, item_sd_l, vec_dim, k_max) schedule(dynamic) num_threads(omp_get_num_procs())
        for (unsigned int i = 0; i < user_sd_l.size(); ++i) {
            for (unsigned int j = 0; j < item_sd_l.size(); ++j) {
                if (user_sd_l[i].norm * item_sd_l[j].norm <= user_sd_l[i].threshold) break;

                // ip comp.
                const float ip = compute_ip(user_sd_l[i], item_sd_l[j], vec_dim);

                // update top-k
                user_sd_l[i].update_topk(ip, item_sd_l[j].identifier, k_max);
            }

            // convert map to array
            user_sd_l[i].make_lb_array();
        }

        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        double time_lower_bound_computation = (double) std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count();
        double time_pre_processing = time_lower_bound_computation;
    }

// blocking
    void blocking(std::vector<SimpferPlusPlusData> &user_sd_l,
                  std::vector<SimpferPlusPlusData> &item_sd_l,
                  std::vector<SimpferPlusPlusBlock> &block_set,
                  const int &n_user, const int &k_max, const int &vec_dim) {
        assert(user_sd_l.size() == n_user);
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

        // determine size
        const unsigned int block_size = (unsigned int) (log2(item_sd_l.size()) * 2);

        // make SimpferPlusPlusBlock
        SimpferPlusPlusBlock blk(k_max);

        // SimpferPlusPlusBlock assignment
        for (unsigned int i = 0; i < user_sd_l.size(); ++i) {
            // assign SimpferPlusPlusBlock id
            user_sd_l[i].block_id = blk.identifier;

            // insert into SimpferPlusPlusBlock
            blk.member.push_back(&user_sd_l[i]);

            // init blk
            if (blk.member.size() == block_size || i == user_sd_l.size() - 1) {

                // update lower-bound array
                blk.update_lowerbound_array(k_max);

                // make matrix
                blk.make_matrix(vec_dim);

                // insert into set
                block_set.push_back(blk);

                // init
                blk.init(k_max);
            }
        }

        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        double time_blocking = (double) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double time_pre_processing = time_blocking;
    }

// pre-processing
    void pre_processing(std::vector<SimpferPlusPlusData> &user_sd_l,
                        std::vector<SimpferPlusPlusData> &item_sd_l,
                        std::vector<SimpferPlusPlusBlock> &block_set,
                        const int &n_user,
                        const int &k_max,
                        const int &vec_dim) {
        // norm computation
        compute_norm(user_sd_l, item_sd_l);

        // lower-bound computation
        compute_lowerbound(user_sd_l, item_sd_l, k_max, vec_dim);

        // blocking
        blocking(user_sd_l, item_sd_l, block_set, n_user, k_max, vec_dim);

        // init
        for (unsigned int i = 0; i < user_sd_l.size(); ++i) user_sd_l[i].init();
    }

}
#endif //REVERSE_KRANKS_SIMPFERPLUSPLUSBUILDINDEX_HPP
