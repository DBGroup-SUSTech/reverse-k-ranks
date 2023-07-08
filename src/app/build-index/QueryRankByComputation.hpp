//
// Created by BianZheng on 2022/11/24.
//

#ifndef REVERSE_KRANKS_QUERYRANKBYCOMPUTATION_HPP
#define REVERSE_KRANKS_QUERYRANKBYCOMPUTATION_HPP

#include "FileIO.hpp"
#include "ReadScoreTable.hpp"
#include "QueryRankBySample.hpp"
#include "struct/VectorMatrix.hpp"

#ifdef USE_GPU

#include "QueryRankGPU.hpp"

#else

#include "QueryRankCPU.hpp"

#endif

#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <queue>
#include <boost/sort/sort.hpp>
#include <vector>
#include <parallel/algorithm>
#include <thread>
#include <spdlog/spdlog.h>

namespace ReverseMIPS {

    class ComputeQueryRankBatch {
        std::vector<float> sample_item_l;
        int n_sample_item_;

#ifdef USE_GPU
        QueryRankGPU gpu;
#else
        QueryRankCPU cpu;
#endif
    public:

        ComputeQueryRankBatch() = default;

        inline ComputeQueryRankBatch(const VectorMatrix &user, const VectorMatrix &data_item,
                                     const std::vector<int> &sample_itemID_l) {
            const float *user_vecs = user.getRawData();
            const float *item_vecs = data_item.getRawData();
            const int n_user = user.n_vector_;
            const int n_data_item = data_item.n_vector_;
            const int n_sample_item = (int) sample_itemID_l.size();
            const int vec_dim = user.vec_dim_;

            this->n_sample_item_ = n_sample_item;
            sample_item_l.resize(n_sample_item * vec_dim);
            for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
                const int itemID = sample_itemID_l[sampleID];
                std::memcpy(sample_item_l.data() + sampleID * vec_dim, item_vecs + itemID * vec_dim,
                            vec_dim * sizeof(float));
            }

            assert(user.vec_dim_ == data_item.vec_dim_);
#ifdef USE_GPU
            gpu = QueryRankGPU(user_vecs, item_vecs, sample_item_l.data(),
                               n_user, n_data_item, n_sample_item_, vec_dim);
            spdlog::info("use GPU Query Rank");
#else
            cpu = QueryRankCPU(user_vecs, item_vecs, sample_item_l.data(),
                               n_user, n_data_item, n_sample_item_, vec_dim);
            spdlog::info("use CPU Query Rank");
#endif
        }

        [[nodiscard]] int ComputeBatchUser() const {

            int batch_n_user;
#ifdef USE_GPU
            batch_n_user = gpu.ComputeBatchUser();
#else
            batch_n_user = cpu.ComputeBatchUser(40);
#endif
            return (int) batch_n_user;
        }

        void init(const int &batch_n_user) {
#ifdef USE_GPU
            gpu.batch_n_user_ = batch_n_user;
            gpu.init();
#else
            cpu.batch_n_user_ = batch_n_user;
            cpu.init();
#endif
        }

        void ComputeQueryRank(const int &start_userID, const int &n_comp_user, int *sample_item_rank_l,
                              double &compute_table_time, double &sort_table_time, double &compute_query_IP_time,
                              double &compute_query_rank_time, double &transfer_time) {
#ifdef USE_GPU
            gpu.ComputeQueryRank(start_userID, n_comp_user, sample_item_rank_l,
                                 compute_table_time, sort_table_time, compute_query_IP_time,
                                 compute_query_rank_time, transfer_time);
#else
            cpu.ComputeQueryRank(start_userID, n_comp_user, sample_item_rank_l,
                                 compute_table_time, sort_table_time, compute_query_IP_time,
                                 compute_query_rank_time, transfer_time);
#endif

        }

        void FinishCompute() {
#ifdef USE_GPU
            gpu.FinishCompute();
#endif
        }
    };


    void ComputeQueryRankByComputation(const VectorMatrix &user, const VectorMatrix &data_item,
                                       const std::vector<int> &sample_itemID_l, const int64_t &n_sample_item,
                                       std::vector<int> &accu_n_user_rank_l,
                                       double &total_init_time, double &total_compute_query_rank_time,
                                       double &total_assign_rank_time,
                                       double &total_compute_accu_rank_time) {
        assert(sample_itemID_l.size() == n_sample_item);

        const int n_user = user.n_vector_;
        const int n_data_item = data_item.n_vector_;

        int64_t accu_n_element = n_sample_item * (n_data_item + 1);
        assert(accu_n_user_rank_l.size() == accu_n_element);
        accu_n_user_rank_l.assign(accu_n_element, 0);

        total_init_time = 0;
        total_compute_query_rank_time = 0;
        total_assign_rank_time = 0;
        total_compute_accu_rank_time = 0;

        double batch_compute_query_rank_time = 0;
        double batch_assign_rank_time = 0;

        double batch_compute_table_time = 0;
        double batch_sort_table_time = 0;
        double batch_compute_queryIP_time = 0;
        double batch_compute_query_rank_component_time = 0;
        double batch_transfer_time = 0;

        TimeRecord record;
        record.reset();
        ComputeQueryRankBatch cqrb(user, data_item, sample_itemID_l);
        total_init_time = record.get_elapsed_time_second();

        const uint64_t batch_n_user = cqrb.ComputeBatchUser();;
        const int remainder = n_user % batch_n_user == 0 ? 0 : 1;
        const int n_batch = n_user / (int) batch_n_user + remainder;
        spdlog::info("{} user per batch, n_batch {}", batch_n_user, n_batch);

        std::vector<int> sample_item_rank_l(batch_n_user * n_sample_item);

        record.reset();
        cqrb.init((int) batch_n_user);
        total_init_time += record.get_elapsed_time_second();

        const uint32_t report_every = 30;

        TimeRecord batch_record, operation_record;
        batch_record.reset();

        for (int batchID = 0; batchID < n_batch; batchID++) {

            double compute_table_time = 0, sort_table_time = 0, compute_query_IP_time = 0, compute_query_rank_time = 0, transfer_time = 0;

            const int start_userID = (int) batch_n_user * batchID;
            operation_record.reset();
            const int n_user_batch = n_user - start_userID > batch_n_user ? (int) batch_n_user : n_user - start_userID;
            cqrb.ComputeQueryRank(start_userID, n_user_batch, sample_item_rank_l.data(),
                                  compute_table_time, sort_table_time, compute_query_IP_time,
                                  compute_query_rank_time, transfer_time);
            const double tmp_compute_query_rank_time = operation_record.get_elapsed_time_second();

            operation_record.reset();
#pragma omp parallel for default(none) shared(n_user_batch, sample_item_rank_l, n_sample_item, n_data_item, accu_n_user_rank_l)
            for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
                for (int batch_userID = 0; batch_userID < n_user_batch; batch_userID++) {
                    const int rank = sample_item_rank_l[(int64_t) batch_userID * n_sample_item + sampleID];
                    accu_n_user_rank_l[(int64_t) sampleID * (n_data_item + 1) + rank]++;
                }
            }
            const double tmp_assign_rank_time = operation_record.get_elapsed_time_second();

            total_compute_query_rank_time += tmp_compute_query_rank_time;
            total_assign_rank_time += tmp_assign_rank_time;

            batch_compute_query_rank_time += tmp_compute_query_rank_time;
            batch_assign_rank_time += tmp_assign_rank_time;

            batch_compute_table_time += compute_table_time;
            batch_sort_table_time += sort_table_time;
            batch_compute_queryIP_time += compute_query_IP_time;
            batch_compute_query_rank_component_time += compute_query_rank_time;
            batch_transfer_time += transfer_time;

            if (batchID % report_every == 0) {
                spdlog::info(
                        "preprocessed {:.1f}%, Mem: {} Mb, {:.2f} s/iter, compute query rank time {:.4f}s, assign rank time {:.4f}s",
                        batchID / (0.01 * n_batch), get_current_RSS() / 1000000,
                        batch_record.get_elapsed_time_second(),
                        batch_compute_query_rank_time, batch_assign_rank_time);
                batch_compute_query_rank_time = 0;
                batch_assign_rank_time = 0;
                batch_record.reset();

                spdlog::info(
                        "compute_table_time {:.4f}s, sort_table_time {:.4f}s, compute_query_IP_time {:.4f}s, compute_query_rank_component_time {:.4f}s, transfer_time {:.4f}s",
                        batch_compute_table_time, batch_sort_table_time, batch_compute_queryIP_time,
                        batch_compute_query_rank_component_time, batch_transfer_time);
                batch_compute_table_time = 0;
                batch_sort_table_time = 0;
                batch_compute_queryIP_time = 0;
                batch_compute_query_rank_component_time = 0;
                batch_transfer_time = 0;
            }

        }
        cqrb.FinishCompute();

        operation_record.reset();
#pragma omp parallel for default(none) shared(n_user, n_sample_item, accu_n_user_rank_l, n_data_item)
        for (int64_t sampleID = 0; sampleID < n_sample_item; sampleID++) {
            const int n_rank = n_data_item + 1;
            for (int rank = 1; rank < n_rank; rank++) {
                accu_n_user_rank_l[sampleID * n_rank + rank] += accu_n_user_rank_l[
                        sampleID * n_rank + rank - 1];
            }
            assert(accu_n_user_rank_l[(sampleID + 1) * n_rank - 1] == n_user);
        }
        total_compute_accu_rank_time = operation_record.get_elapsed_time_second();

    }

    void BuildIndexByComputation(const VectorMatrix &user, const VectorMatrix &data_item,
                                 const char *index_path, const char *dataset_name, const char *index_dir,
                                 const int &n_sample_item, const int &sample_topk,
                                 double &sample_item_time, double &compute_query_rank_time,
                                 double &sort_kth_rank_time, double &store_index_time) {
        TimeRecord record;
        record.reset();
        const int n_data_item = data_item.n_vector_;
        std::vector<int> sample_itemID_l(n_sample_item);
        SampleItem(n_data_item, n_sample_item, sample_itemID_l);
        sample_item_time = record.get_elapsed_time_second();

        record.reset();
        const size_t size_t_n_sample_item = n_sample_item;
        std::vector<int> accu_n_user_rank_l(size_t_n_sample_item * (n_data_item + 1));
        double total_init_time, total_compute_query_rank_time, total_assign_rank_time, total_compute_accu_rank_time;
        ComputeQueryRankByComputation(user, data_item,
                                      sample_itemID_l, n_sample_item,
                                      accu_n_user_rank_l,
                                      total_init_time, total_compute_query_rank_time,
                                      total_assign_rank_time,
                                      total_compute_accu_rank_time);
        compute_query_rank_time = record.get_elapsed_time_second();

        record.reset();
        //sort the rank in ascending sort, should also influence sample_itemID_l
        std::vector<int> sort_kth_rank_l(n_sample_item);
        std::vector<int> sort_sampleID_l(n_sample_item);
        ComputeSortKthRank(accu_n_user_rank_l, n_data_item,
                           n_sample_item, sample_topk,
                           sort_kth_rank_l, sort_sampleID_l);
        sort_kth_rank_time = record.get_elapsed_time_second();

//    for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
//        printf("%d ", sort_kth_rank_l[sampleID]);
//    }
//    printf("\n");
//
//    for (int sampleID = 0; sampleID < n_sample_item; sampleID++) {
//        const int sort_sampleID = sort_sampleID_l[sampleID];
//        for (int i = 0; i < 30; i++) {
//            printf("%5d ", accu_n_user_rank_l[sort_sampleID * (n_data_item + 1) + i]);
//        }
//        printf("\n");
//    }

        record.reset();
        WriteDistributionBelowTopk(sample_itemID_l, sort_kth_rank_l,
                                   sort_sampleID_l, accu_n_user_rank_l,
                                   n_data_item, n_sample_item, sample_topk,
                                   dataset_name, index_dir);
        store_index_time = record.get_elapsed_time_second();
    }

}
#endif //REVERSE_KRANKS_QUERYRANKBYCOMPUTATION_HPP
