//
// Created by bianzheng on 2023/3/23.
//

#ifndef REVERSE_KRANKS_QUERYRANKCPU_HPP
#define REVERSE_KRANKS_QUERYRANKCPU_HPP
namespace ReverseMIPS {
    class QueryRankCPU {

        const float *user_ptr_;
        const float *data_item_ptr_;
        const float *sample_item_ptr_;
        int64_t n_user_, n_data_item_, n_sample_item_;
        int vec_dim_;

        std::vector<float> score_table_l_, sample_item_IP_l_;
    public:
        int64_t batch_n_user_;

        QueryRankCPU() = default;

        QueryRankCPU(const float *user_ptr, const float *data_item_ptr, const float *sample_item_ptr,
                     const int64_t &n_user, const int64_t &n_data_item, const int64_t &n_sample_item,
                     const int64_t &vec_dim) {
            this->user_ptr_ = user_ptr;
            this->data_item_ptr_ = data_item_ptr;
            this->sample_item_ptr_ = sample_item_ptr;

            this->n_user_ = n_user;
            this->n_data_item_ = n_data_item;
            this->n_sample_item_ = n_sample_item;
            this->vec_dim_ = (int) vec_dim;
        }

        [[nodiscard]] int ComputeBatchUser(const uint64_t &memory_size_gb) const {
            // things in memory,
            // user_ptr, sizeof(float) * n_user * vec_dim
            // item_ptr, sizeof(float) * n_data_item * vec_dim
            // sample_item_ptr, sizeof(float) * n_sample_item * vec_dim
            // score_table_ptr, sizeof(float) * n_batch_user * n_data_item
            // sample_item_ip_ptr, sizeof(float) * n_batch_user * n_sample_item
            // sample_item_rank_ptr, sizeof(float) * n_batch_user * n_sample_item
            const uint64_t dataset_size = sizeof(float) * (n_user_ + n_data_item_ + n_sample_item_) * vec_dim_;
            const uint64_t memory_size = memory_size_gb * 1024 * 1024 * 1024;
            const uint64_t batch_n_user =
                    (memory_size - dataset_size) / sizeof(float) / (n_data_item_ + n_sample_item_ * 2);
            if (batch_n_user > n_user_) {
                return 12;
            }
            return (int) batch_n_user;
        }

        void init() {
            this->score_table_l_.resize(batch_n_user_ * n_data_item_);
            this->sample_item_IP_l_.resize(batch_n_user_ * n_sample_item_);
        }

        void ComputeQueryRank(const int &start_userID, const int &n_compute_user, int *sample_item_rank_l,
                              double& compute_table_time, double& sort_table_time, double& compute_query_IP_time,
                              double& compute_query_rank_time, double& transfer_time) {

            // compute IP of each user and all data item
#pragma omp parallel for default(none) shared(n_compute_user, n_data_item_, score_table_l_, start_userID, user_ptr_, data_item_ptr_)
            for (int comp_userID = 0; comp_userID < n_compute_user; comp_userID++) {
                float *score_table_ptr = score_table_l_.data() + comp_userID * n_data_item_;
                const int userID = comp_userID + start_userID;
                const float *user_vecs = user_ptr_ + userID * vec_dim_;
                for (int itemID = 0; itemID < n_data_item_; itemID++) {
                    const float *item_vecs = data_item_ptr_ + itemID * vec_dim_;
                    const float ip = InnerProduct(user_vecs, item_vecs, vec_dim_);
                    score_table_ptr[itemID] = ip;
                }
            }

            // sort the IP of each user and all data item
            for (int comp_userID = 0; comp_userID < n_compute_user; comp_userID++) {
                float *score_table_ptr = score_table_l_.data() + comp_userID * n_data_item_;
                boost::sort::block_indirect_sort(score_table_ptr, score_table_ptr + n_data_item_, std::greater(),
                                                 std::thread::hardware_concurrency());
            }

            // compute the sample itemID
#pragma omp parallel for default(none) shared(n_compute_user, n_sample_item_, score_table_l_, start_userID, user_ptr_, data_item_ptr_)
            for (int comp_userID = 0; comp_userID < n_compute_user; comp_userID++) {
                float *sample_item_IP_ptr = sample_item_IP_l_.data() + comp_userID * n_sample_item_;
                const int userID = comp_userID + start_userID;
                const float *user_vecs = user_ptr_ + userID * vec_dim_;
                for (int itemID = 0; itemID < n_sample_item_; itemID++) {
                    const float *item_vecs = sample_item_ptr_ + itemID * vec_dim_;
                    const float ip = InnerProduct(user_vecs, item_vecs, vec_dim_);
                    sample_item_IP_ptr[itemID] = ip;
                }
            }

            // sort the query rank
#pragma omp parallel for default(none) shared(n_compute_user, n_sample_item_, sample_item_rank_l, n_data_item_, sample_item_IP_l_, score_table_l_)
            for (int comp_userID = 0; comp_userID < n_compute_user; comp_userID++) {
                float *sample_item_IP_ptr = sample_item_IP_l_.data() + comp_userID * n_sample_item_;
                float *score_table_ptr = score_table_l_.data() + comp_userID * n_data_item_;

                for (int64_t sampleID = 0; sampleID < n_sample_item_; sampleID++) {
                    const float sampleIP = sample_item_IP_ptr[sampleID];
                    float *rank_ptr = std::lower_bound(score_table_ptr, score_table_ptr + n_data_item_, sampleIP,
                                                       [](const float &arrIP, float queryIP) {
                                                           return arrIP > queryIP;
                                                       });
                    const int64_t rank = rank_ptr - score_table_ptr;
                    assert(0 <= rank && rank <= n_data_item_);

                    sample_item_rank_l[comp_userID * n_sample_item_ + sampleID] = (int) rank;
                }
            }

        }


    };
}
#endif //REVERSE_KRANKS_QUERYRANKCPU_HPP
