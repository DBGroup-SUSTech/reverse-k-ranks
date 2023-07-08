//
// Created by bianzheng on 2023/7/5.
//

#ifndef REVERSE_KRANKS_VECTORIOUPDATE_HPP
#define REVERSE_KRANKS_VECTORIOUPDATE_HPP

#include "struct/VectorMatrixUpdate.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    template<typename T>
    std::unique_ptr<T[]> loadVector(const char *filename, int &n_data, int &dim) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            std::cerr << "Open file error" << std::endl;
            exit(-1);
        }

        in.read((char *) &dim, 4);

        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        auto fsize = (size_t) ss;
        n_data = (int) (fsize / (sizeof(T) / 4 * dim + 1) / 4);

        std::unique_ptr<T[]> data = std::make_unique<T[]>((size_t) n_data * (size_t) dim);
        in.seekg(0, std::ios::beg);
        for (int i = 0; i < n_data; i++) {
            in.seekg(4, std::ios::cur);
            in.read((char *) (data.get() + i * dim), dim * sizeof(T));
        }
        in.close();

        return data;
    }

    std::vector<VectorMatrixUpdate>
    readDataUpdate(const char *basic_dir, const char *dataset_name, const std::string &update_type,
                   int &n_user, int &n_data_item, int &n_query_item,
                   int &n_update_user, int &n_update_item,
                   int &vec_dim) {
        n_user = 0;
        n_data_item = 0;
        n_query_item = 0;
        n_update_user = 0;
        n_update_item = 0;
        vec_dim = 0;

        char path[256];
        std::sprintf(path, "%s/%s/%s_data_item.fvecs", basic_dir, dataset_name, dataset_name);
        std::unique_ptr<float[]> data_item_ptr = loadVector<float>(path, n_data_item, vec_dim);

        sprintf(path, "%s/%s/%s_user.fvecs", basic_dir, dataset_name, dataset_name);
        std::unique_ptr<float[]> user_ptr = loadVector<float>(path, n_user, vec_dim);

        sprintf(path, "%s/%s/%s_query_item.fvecs", basic_dir, dataset_name, dataset_name);
        std::unique_ptr<float[]> query_item_ptr = loadVector<float>(path, n_query_item, vec_dim);

        static VectorMatrixUpdate user, data_item, query_item;
        user.init(user_ptr, n_user, vec_dim);
        data_item.init(data_item_ptr, n_data_item, vec_dim);
        query_item.init(query_item_ptr, n_query_item, vec_dim);
        user.vectorNormalize();


        static VectorMatrixUpdate user_update, data_item_update;
        if (update_type == "data_item") {
            std::sprintf(path, "%s/%s/%s_data_item_update.fvecs", basic_dir, dataset_name, dataset_name);
            std::unique_ptr<float[]> data_item_update_ptr = loadVector<float>(path, n_update_item, vec_dim);

            data_item_update.init(data_item_update_ptr, n_update_item, vec_dim);

        } else if (update_type == "user") {
            sprintf(path, "%s/%s/%s_user_update.fvecs", basic_dir, dataset_name, dataset_name);
            std::unique_ptr<float[]> user_update_ptr = loadVector<float>(path, n_update_user, vec_dim);

            user_update.init(user_update_ptr, n_update_user, vec_dim);
            user_update.vectorNormalize();

        } else {
            spdlog::error("do not support such update type");
        }


        std::vector<VectorMatrixUpdate> res(5);
        res[0] = std::move(user);
        res[1] = std::move(data_item);
        res[2] = std::move(query_item);
        res[3] = std::move(user_update);
        res[4] = std::move(data_item_update);
        return res;
    }

    std::vector<VectorMatrixUpdate>
    splitVectorMatrix(VectorMatrixUpdate &vm, const int &n_split_vecs, const int &n_split) {
        const int n_vector = vm.n_vector_;
        const int vec_dim = vm.vec_dim_;
        if (n_split_vecs * n_split > n_vector) {
            spdlog::error(
                    "Split VectorMaxtrix error, not enough vector to split, total vector {}, split vector {}, split {}",
                    n_vector, n_split_vecs, n_split);
            exit(1);
        }

        std::vector<VectorMatrixUpdate> res(n_split);
        for (int splitID = 0; splitID < n_split; ++splitID) {
            const int split_start_vecsID = n_split_vecs * splitID;
            std::unique_ptr<float[]> split_ptr = std::make_unique<float[]>(n_split_vecs * vec_dim);

            std::memcpy(split_ptr.get(), vm.getRawData() + split_start_vecsID * vec_dim,
                        n_split_vecs * vec_dim * sizeof(float));
            VectorMatrixUpdate split_vm;
            split_vm.init(split_ptr, n_split_vecs, vec_dim);

            res[splitID] = std::move(split_vm);
        }
        return res;
    }


    VectorMatrixUpdate &&
    splitVectorMatrix(VectorMatrixUpdate &vm, const int &n_sample_vecs) {
        const int n_vector = vm.n_vector_;
        const int vec_dim = vm.vec_dim_;
        if (n_sample_vecs > n_vector) {
            spdlog::error(
                    "Split VectorMaxtrix error, not enough vector to split, total vector {}, num sample vector {}",
                    n_vector, n_sample_vecs);
            exit(1);
        }
//        else if (n_sample_vecs == 0) {
//            spdlog::error(
//                    "Split VectorMaxtrix error, not enough vector to split, total vector {}, num sample vector {}",
//                    n_vector, n_sample_vecs);
//            exit(1);
//        }

        std::unique_ptr<float[]> split_ptr = std::make_unique<float[]>(n_sample_vecs * vec_dim);

        std::memcpy(split_ptr.get(), vm.getRawData(),
                    sizeof(float) * n_sample_vecs * vec_dim);
        static VectorMatrixUpdate split_vm;
        split_vm.init(split_ptr, n_sample_vecs, vec_dim);

        return std::move(split_vm);
    }
}
#endif //REVERSE_KRANKS_VECTORIOUPDATE_HPP
