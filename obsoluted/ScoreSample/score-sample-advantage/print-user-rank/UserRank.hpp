//
// Created by BianZheng on 2022/5/17.
//

#ifndef REVERSE_KRANKS_ATTRIBUTION_BPLUSTREE_HPP
#define REVERSE_KRANKS_ATTRIBUTION_BPLUSTREE_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "alg/RankBoundRefinement/PruneCandidateByBound.hpp"
#include "util/VectorIO.hpp"
#include "util/TimeMemory.hpp"
#include "util/FileIO.hpp"
#include "struct/VectorMatrix.hpp"
#include "struct/MethodBase.hpp"
#include <fstream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <map>
#include <spdlog/spdlog.h>

namespace ReverseMIPS::UserRank {

    class TreeIndex {
        int BinarySearch(const double &queryIP, const int &base_rank, const int &arr_size) const {
            const double *ip_vecs = item_cache_.data();

            const double *lb_vecs = std::lower_bound(ip_vecs, ip_vecs + arr_size, queryIP,
                                                     [](const double &arrIP, double queryIP) {
                                                         return arrIP > queryIP;
                                                     });
            int offset_rank = (int) (lb_vecs - ip_vecs);

            return offset_rank + base_rank;
        }

    public:
        /**The implementation doesn't have the node, it samples the index of each column**/
        //meta information
        std::vector<double> memory_bp_tree_; // n_user_ * n_memory_element_, store the upper layer of b+ tree in memory, top -> bottom
        std::vector<int> layer_size_l_;//height_, top to bottom
        std::vector<int> memory_cum_layer_size_l_; // n_memory_layer, top to bottom
        std::vector<int> disk_cum_layer_size_l_; // n_disk_layer, top to bottom
        size_t disk_tree_size_byte_;
        size_t n_memory_element_;
        int height_;
        int n_data_item_, n_user_, node_size_, n_disk_layer_, n_memory_layer_;

        //IO
        const char *index_path_;
        std::ofstream out_stream_;
        std::ifstream in_stream_;
        size_t disk_index_size_;

        //Retrieval
        std::vector<double> item_cache_;// node_size_, cache for reading a node on the disk
        double read_disk_time_, exact_rank_refinement_time_;
        TimeRecord read_disk_record_, exact_rank_refinement_record_;
        std::vector<int> user_rank_cache_l_;

        inline TreeIndex() = default;

        inline TreeIndex(const int &n_data_item, const int &n_user,
                         const int &node_size, const int &n_disk_layer, const char *index_path) {
            if (node_size > n_data_item) {
                spdlog::error("sample number is too large, program exit");
                exit(-1);
            }
            if (n_disk_layer <= 0) {
                spdlog::error("n_disk_layer is too small, program exit");
                exit(-1);
            }

            this->n_data_item_ = n_data_item;
            this->n_user_ = n_user;
            this->node_size_ = node_size;
            this->n_disk_layer_ = n_disk_layer;

            this->index_path_ = index_path;
            out_stream_ = std::ofstream(index_path, std::ios::binary | std::ios::out);
            if (!out_stream_) {
                spdlog::error("error in write result");
                exit(-1);
            }
            this->item_cache_.resize(node_size_);
            this->user_rank_cache_l_.resize(n_user);

            GetMetaInfo();
        }

        void GetMetaInfo() {
            height_ = 1;
            layer_size_l_.clear();
            memory_cum_layer_size_l_.clear();
            disk_cum_layer_size_l_.clear();

            int layer_size = n_data_item_;
            // append the data from the bottom to the top
            layer_size_l_.emplace_back(layer_size);
            while (layer_size > node_size_) {
                int n_ip_layer = layer_size / node_size_;
                if (layer_size % node_size_ != 0) {
                    n_ip_layer++;
                }
                layer_size = n_ip_layer;
                layer_size_l_.emplace_back(layer_size);
                height_++;
            }
            n_memory_layer_ = height_ - n_disk_layer_;
            assert(height_ == layer_size_l_.size());

            if (n_disk_layer_ >= height_) {
                spdlog::error("height lower than n_disk_layer, program exit");
                exit(-1);
            }

            std::reverse(layer_size_l_.begin(), layer_size_l_.end());

            assert(height_ == layer_size_l_.size());

            memory_cum_layer_size_l_.emplace_back(layer_size_l_[0]);

            for (int layer = 1; layer < n_memory_layer_; layer++) {
                int n_ip_cum_layer = memory_cum_layer_size_l_[layer - 1] + layer_size_l_[layer];
                memory_cum_layer_size_l_.emplace_back(n_ip_cum_layer);
            }

            memory_bp_tree_.resize(memory_cum_layer_size_l_[n_memory_layer_ - 1] * n_user_);

            disk_cum_layer_size_l_.emplace_back(layer_size_l_[height_ - n_disk_layer_]);
            if (height_ - n_disk_layer_ + 1 < height_) {
                for (int layer = height_ - n_disk_layer_ + 1; layer < height_; layer++) {
                    int current_ptr = layer - (height_ - n_disk_layer_);
                    int n_ip_cum_layer = disk_cum_layer_size_l_[current_ptr - 1] + layer_size_l_[layer];
                    disk_cum_layer_size_l_.emplace_back(n_ip_cum_layer);
                }
            }
            assert(disk_cum_layer_size_l_.size() == n_disk_layer_);

            n_memory_element_ = 0;
            for (int layer = 0; layer < n_memory_layer_; layer++) {
                n_memory_element_ += layer_size_l_[layer];
            }

            disk_tree_size_byte_ = 0;
            for (int layer = height_ - n_disk_layer_; layer < height_; layer++) {
                disk_tree_size_byte_ += layer_size_l_[layer] * sizeof(double);
            }

            int test_n_node = 0;
            for (int layer = 0; layer < height_; layer++) {
                test_n_node += layer_size_l_[layer];
            }
            assert(test_n_node ==
                   memory_cum_layer_size_l_[n_memory_layer_ - 1] + disk_cum_layer_size_l_[n_disk_layer_ - 1]);

            spdlog::info(
                    "disk_tree_size_byte_ {}, n_memory_element_ {}, height_ {}, n_data_item_ {}, n_user_ {}, node_size_ {}, n_disk_layer_ {}, n_memory_layer_ {}",
                    disk_tree_size_byte_, n_memory_element_, height_, n_data_item_, n_user_, node_size_, n_disk_layer_,
                    n_memory_layer_);

            std::cout << "layer_size_l_" << std::endl;
            for (int layer = 0; layer < height_; layer++) {
                std::cout << layer_size_l_[layer] << " ";
            }
            std::cout << std::endl;

            std::cout << "memory_cum_layer_size_l_" << std::endl;
            for (int layer = 0; layer < n_memory_layer_; layer++) {
                std::cout << memory_cum_layer_size_l_[layer] << " ";
            }
            std::cout << std::endl;

            std::cout << "disk_cum_layer_size_l_" << std::endl;
            for (int layer = 0; layer < n_disk_layer_; layer++) {
                std::cout << disk_cum_layer_size_l_[layer] << " ";
            }
            std::cout << std::endl;

        }

        void BuildWriteTree(const double *IP_vecs, const int &userID) {
            //node from the bottom to the top, idx 0 is the bottom
            std::vector<std::vector<double>> layer_l;

            //build the TreeNode from the bottom
            {
                std::vector<double> layer_ip(n_data_item_);
                std::memcpy(layer_ip.data(), IP_vecs, n_data_item_ * sizeof(double));
                layer_l.push_back(layer_ip);
                assert(layer_ip.size() == layer_size_l_[height_ - 1]);
            }

            //get the IP for each layer
            for (int layer = height_ - 2; layer >= 0; layer--) {
                int layer_size = layer_size_l_[layer];
                std::vector<double> &prev_layer = layer_l[height_ - 2 - layer];
                const int prev_layer_size = (int) prev_layer.size();
                //sample
                std::vector<double> layer_ip;
                for (int ipID = node_size_ - 1; ipID < prev_layer_size; ipID += node_size_) {
                    layer_ip.emplace_back(prev_layer[ipID]);
                }
                if (prev_layer_size % node_size_ != 0) {
                    layer_ip.emplace_back(prev_layer[prev_layer_size - 1]);
                }

                assert(layer_ip.size() == layer_size);
                layer_l.push_back(layer_ip);
            }

            assert(layer_l.size() == height_);

            // write the layer to the disk
            for (int layer = height_ - n_disk_layer_; layer <= height_ - 1; layer++) {
                std::vector<double> layer_ip = layer_l[height_ - 1 - layer];
                assert(layer_ip.size() == layer_size_l_[layer]);
                out_stream_.write((char *) layer_ip.data(), sizeof(double) * layer_ip.size());
            }
            //write the layer to the memory index
            size_t offset = userID * n_memory_element_;
            for (int layer = 0; layer < n_memory_layer_; layer++) {
                std::vector<double> layer_ip = layer_l[height_ - 1 - layer];
                assert(layer_ip.size() == layer_size_l_[layer]);
                const int layer_size = layer_size_l_[layer];
                for (int eleID = 0; eleID < layer_size; eleID++) {
                    memory_bp_tree_[offset] = layer_ip[eleID];
                    offset++;
                }

            }

        }

        void RetrievalPreprocess() {
            read_disk_time_ = 0;
            exact_rank_refinement_time_ = 0;

            in_stream_ = std::ifstream(this->index_path_, std::ios::binary | std::ios::in);
            if (!in_stream_) {
                spdlog::error("error in reading index");
                exit(-1);
            }
            in_stream_.seekg(0, std::ios::end);
            std::ios::pos_type ss = in_stream_.tellg();
            in_stream_.seekg(0, std::ios::beg);
            size_t fsize = (size_t) ss;
            disk_index_size_ = fsize;
//            spdlog::info("index size %ld byte\n", fsize);
        }

        void RetrievalMemory(const std::vector<double> &queryIP_l, std::vector<int> &rank_l) {
            assert(queryIP_l.size() == n_user_ && rank_l.size() == n_user_);
            for (int userID = 0; userID < n_user_; userID++) {
                const double &queryIP = queryIP_l[userID];
                size_t base_offset = userID * n_memory_element_;
                assert(in_stream_);
                int rank = 0;

                for (int layer = 0; layer < n_memory_layer_; layer++) {
                    int base_rank = rank * node_size_;
                    if (base_rank >= layer_size_l_[layer]) {
                        base_rank = layer_size_l_[layer];
                    }
                    int layer_offset = layer == 0 ? 0 : memory_cum_layer_size_l_[layer - 1];
                    int block_offset = base_rank;
                    size_t offset = base_offset + layer_offset + block_offset;

                    int normal_read_count = node_size_;
                    int remain_read_count = layer_size_l_[layer] - base_rank;
                    int read_count = std::min(normal_read_count, remain_read_count);
                    assert(offset + read_count <= (userID + 1) * n_memory_element_);

                    memcpy(item_cache_.data(), memory_bp_tree_.data() + offset, read_count * sizeof(double));
                    rank = BinarySearch(queryIP, base_rank, read_count);

                    assert(0 <= rank && rank <= layer_size_l_[layer]);
                }
                assert(in_stream_);
                assert(rank >= 0);
                rank_l[userID] = rank;
            }

        }

        std::vector<int>
        RetrievalDisk(const std::vector<double> &queryIP_l, const std::vector<int> &rank_l) {
            assert(queryIP_l.size() == n_user_);
            int n_candidate = 0;
            for (int userID = 0; userID < n_user_; userID++) {
                const double &queryIP = queryIP_l[userID];
                size_t base_offset = userID * disk_tree_size_byte_;
                assert(in_stream_);
                int rank = rank_l[userID];

                for (int disk_layer = 0; disk_layer < n_disk_layer_; disk_layer++) {
                    int base_rank = rank * node_size_;
                    if (base_rank >= layer_size_l_[n_memory_layer_ + disk_layer]) {
                        base_rank = layer_size_l_[n_memory_layer_ + disk_layer];
                    }
                    int layer_offset = disk_layer == 0 ? 0 : disk_cum_layer_size_l_[disk_layer - 1];
                    int block_offset = base_rank;
                    size_t offset = base_offset + (layer_offset + block_offset) * sizeof(double);

                    int normal_read_count = node_size_;
                    int remain_read_count = layer_size_l_[n_memory_layer_ + disk_layer] - base_rank;
                    int read_byte = std::min(normal_read_count, remain_read_count) * sizeof(double);

                    assert(offset + read_byte <= (userID + 1) * disk_tree_size_byte_);
                    read_disk_record_.reset();
                    in_stream_.seekg((int64_t) offset, std::ios::beg);
                    in_stream_.read((char *) item_cache_.data(), (int64_t) read_byte);
                    read_disk_time_ += read_disk_record_.get_elapsed_time_second();

                    exact_rank_refinement_record_.reset();
                    int arr_size = (int) (read_byte / sizeof(double));
                    rank = BinarySearch(queryIP, base_rank, arr_size);

                    assert(0 <= rank && rank <= layer_size_l_[n_memory_layer_ + disk_layer]);
                    exact_rank_refinement_time_ += exact_rank_refinement_record_.get_elapsed_time_second();
                }
                assert(in_stream_);
                assert(rank >= 0);
                rank += 1;

                user_rank_cache_l_[userID] = rank;
            }

            return user_rank_cache_l_;
        }

        void FinishRetrieval() {
            this->in_stream_.close();
        }

    };

    class Index {
        void ResetTimer() {
            inner_product_time_ = 0;
            rank_bound_prune_time_ = 0;
            read_disk_time_ = 0;
            exact_rank_refinement_time_ = 0;

            rank_prune_ratio_ = 0;
        }

    public:
        TreeIndex tree_ins_;
        VectorMatrix user_;
        int n_user_, n_data_item_, vec_dim_;
        double inner_product_time_, rank_bound_prune_time_, read_disk_time_, exact_rank_refinement_time_;
        TimeRecord inner_product_record_, rank_bound_prune_record_;
        double rank_prune_ratio_;

        Index() {}

        Index(TreeIndex &tree_ins, const int n_data_item, VectorMatrix &user) {
            this->tree_ins_ = std::move(tree_ins);
            this->n_user_ = user.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->user_ = std::move(user);
            this->n_data_item_ = n_data_item;
        }

        std::vector<std::vector<int>> GetAllRank(const VectorMatrix &query_item, std::vector<double> &prune_ratio_l) {
            ResetTimer();
            assert(prune_ratio_l.size() == query_item.n_vector_);

            tree_ins_.RetrievalPreprocess();

            int n_query_item = query_item.n_vector_;

            std::vector<std::vector<int>> query_rank_l(n_query_item, std::vector<int>(n_user_));

            std::vector<double> queryIP_l(n_user_);
            std::vector<int> rank_l(n_user_);
            std::vector<bool> prune_l(n_user_);

            for (int qID = 0; qID < n_query_item; qID++) {
                prune_l.assign(n_user_, false);

                //calculate distance
                double *query_item_vec = query_item.getVector(qID);
                for (int userID = 0; userID < n_user_; userID++) {
                    double *user_vec = user_.getVector(userID);
                    double queryIP = InnerProduct(query_item_vec, user_vec, vec_dim_);
                    queryIP_l[userID] = queryIP;
                }

                tree_ins_.RetrievalMemory(queryIP_l, rank_l);

                //read disk and fine binary search
                std::vector<int> user_rank_l = tree_ins_.RetrievalDisk(queryIP_l, rank_l);

                for (int quserID = 0; quserID < n_user_; quserID++) {
                    query_rank_l[qID][quserID] = user_rank_l[quserID];
                }

            }

            tree_ins_.FinishRetrieval();
            return query_rank_l;
        }

        std::string
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &second_per_query) {
            // int topk;
            //double total_time,
            //          inner_product_time, read_disk_time, binary_search_time;
            //double second_per_query;
            //unit: second

            char buff[1024];

            sprintf(buff,
                    "top%d retrieval time:\n\ttotal %.3fs\n\tinner product %.3fs, rank bound pruning %.3fs\n\tread disk %.3fs, exact rank refinement %.3fs\n\trank prune ratio %.4f\n\tmillion second per query %.3fms",
                    topk, retrieval_time,
                    inner_product_time_, rank_bound_prune_time_,
                    read_disk_time_, exact_rank_refinement_time_,
                    rank_prune_ratio_,
                    second_per_query);
            std::string str(buff);
            return str;
        }

    };

    const int write_every_ = 30000;
    const int report_batch_every_ = 5;

    /*
     * bruteforce index
     * shape: 1, type: int, n_data_item
     * shape: n_user * n_data_item, type: double, the distance pair for each user
     */

    std::unique_ptr<Index>
    BuildIndex(VectorMatrix &data_item, VectorMatrix &user, const char *index_path, const int &node_size) {
        const int n_user = user.n_vector_;
        const int n_data_item = data_item.n_vector_;
        std::vector<double> distance_cache(write_every_ * n_data_item);
        const int vec_dim = data_item.vec_dim_;
        const int n_batch = user.n_vector_ / write_every_;
        const int n_remain = user.n_vector_ % write_every_;
        user.vectorNormalize();

        const int n_disk_layer = 1;
        TreeIndex tree_ins(n_data_item, n_user, node_size, n_disk_layer, index_path);

        TimeRecord batch_report_record;
        batch_report_record.reset();
        for (int i = 0; i < n_batch; i++) {
#pragma omp parallel for default(none) shared(i, data_item, user, distance_cache) shared(write_every_, n_data_item, vec_dim)
            for (int cacheID = 0; cacheID < write_every_; cacheID++) {
                int userID = write_every_ * i + cacheID;
                for (int itemID = 0; itemID < n_data_item; itemID++) {
                    double ip = InnerProduct(data_item.getVector(itemID), user.getVector(userID), vec_dim);
                    distance_cache[cacheID * n_data_item + itemID] = ip;
                }
                std::sort(distance_cache.begin() + cacheID * n_data_item,
                          distance_cache.begin() + (cacheID + 1) * n_data_item, std::greater<double>());
            }

            for (int cacheID = 0; cacheID < write_every_; cacheID++) {
                int userID = write_every_ * i + cacheID;
                const double *distance_vecs = distance_cache.data() + cacheID * n_data_item;
                tree_ins.BuildWriteTree(distance_vecs, userID);
            }

            if (i % report_batch_every_ == 0) {
                spdlog::info("preprocessed {}%, {} s/iter Mem: {} Mb", i / (0.01 * n_batch),
                             batch_report_record.get_elapsed_time_second(), get_current_RSS() / 1000000);
                batch_report_record.reset();
            }

        }

        if (n_remain != 0) {
            for (int cacheID = 0; cacheID < n_remain; cacheID++) {
                int userID = cacheID + write_every_ * n_batch;
                for (int itemID = 0; itemID < data_item.n_vector_; itemID++) {
                    double ip = InnerProduct(data_item.getRawData() + itemID * vec_dim,
                                             user.getRawData() + userID * vec_dim, vec_dim);
                    distance_cache[cacheID * data_item.n_vector_ + itemID] = ip;
                }

                std::sort(distance_cache.begin() + cacheID * n_data_item,
                          distance_cache.begin() + (cacheID + 1) * n_data_item, std::greater<double>());
            }

            for (int cacheID = 0; cacheID < n_remain; cacheID++) {
                int userID = cacheID + write_every_ * n_batch;
                const double *distance_vecs = distance_cache.data() + cacheID * n_data_item;
                tree_ins.BuildWriteTree(distance_vecs, userID);
            }
        }
        tree_ins.out_stream_.close();

        std::unique_ptr<Index> index_ptr = std::make_unique<Index>(tree_ins, n_data_item, user);
        return index_ptr;
    }

}
#endif //REVERSE_KRANKS_ATTRIBUTION_BPLUSTREE_HPP
