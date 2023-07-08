//
// Created by BianZheng on 2022/3/28.
//

#ifndef REVERSE_KRANKS_BPLUSTREE_HPP
#define REVERSE_KRANKS_BPLUSTREE_HPP

//the following are UBUNTU/LINUX, and MacOS ONLY terminal color codes.
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

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

namespace ReverseMIPS::BPlusTree {

    class TreeIndex {
        int BinarySearch(const double &queryIP, const int &arr_size) const {
            const double *ip_vecs = item_cache_.data();

            const double *lb_vecs = std::lower_bound(ip_vecs, ip_vecs + arr_size, queryIP,
                                                     [](const double &arrIP, double queryIP) {
                                                         return arrIP > queryIP;
                                                     });
            int offset_rank = (int) (lb_vecs - ip_vecs);

            return offset_rank;
        }

    public:
        /**The implementation doesn't have the node, it samples the index of each column**/
        //for build record
        std::vector<int> tree_height_l_;//n_data_item, record the layer that it should have, from bottom to top
        //meta information
        int height_;
        int n_data_item_, n_user_, node_size_, n_disk_layer_, n_memory_layer_;
        std::vector<int> layer_size_l_;//height_, top to bottom
        std::vector<int> memory_cum_layer_size_l_; // n_memory_layer, top to bottom
        std::vector<int> disk_cum_layer_size_l_; // n_disk_layer, top to bottom
        size_t n_memory_element_;
        size_t disk_tree_size_byte_;
        std::vector<int> layer_rank_bound_l_;//height, stores the bound for a difference in a layer, top to bottom , value decrease from top to bottom
        std::vector<double> memory_bp_tree_; // n_user_ * n_memory_element_, store the upper layer of b+ tree in memory, top -> bottom

        //IO
        const char *index_path_;
        std::ofstream out_stream_;
        std::ifstream in_stream_;
        size_t disk_index_size_;

        //Retrieval
        std::vector<double> item_cache_;// node_size_, cache for reading a node on the disk
        std::vector<int> node_offset_l_;//n_user, stores the offset in each node of a layer
        std::vector<bool> is_all_last_node_l_;//n_user, whether it is the last node

        double read_disk_time_, exact_rank_refinement_time_;
        TimeRecord read_disk_record_, exact_rank_refinement_record_;
        std::vector<UserRankElement> user_topk_cache_l_;

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
            this->item_cache_.resize(2 * node_size_ - 1);
            this->user_topk_cache_l_.resize(n_user);
            this->node_offset_l_.resize(n_user);
            this->is_all_last_node_l_.resize(n_user);

            GetMetaInfo();
        }

        void GetMetaInfo() {
            layer_size_l_.clear();
            memory_cum_layer_size_l_.clear();
            disk_cum_layer_size_l_.clear();

            if (n_data_item_ <= node_size_ * 2) {
                spdlog::error("# data item is too small, program exit");
                exit(-1);
            }
            //insert the element in descending order
            //n = 2 * \tau pointer
            //leaf node, get a node for every \tau elements, the maximum element of last node is 2 * \tau - 1, minimum is \tau
            //non-leaf node, sample a node for every (\tau + 1) elements, the maximum element of the last node is 2 * \tau - 1, minimum is \tau - 1

            // the step for bottom
            tree_height_l_.resize(n_data_item_);
            tree_height_l_.assign(n_data_item_, 0);

            height_ = 1;

            //the step for the first sampling layer
            int n_sample = n_data_item_ / node_size_ - 1;
            assert(n_sample >= 1);
            int n_cand = n_data_item_;
            std::vector<int> cand_l(n_data_item_); // record which candidates are remained in this layer
            for (int sampleID = 0, itemID = node_size_; sampleID < n_sample; sampleID++, itemID += node_size_) {
                cand_l[sampleID] = itemID;
                tree_height_l_[itemID] = 1;
                assert(itemID < n_cand);
            }
            n_cand = n_sample;

            int next_n_sample = n_sample / (node_size_ + 1);
            int next_n_remain = n_sample % (node_size_ + 1);
            if (next_n_remain != node_size_ - 1 && next_n_remain != node_size_) {
                next_n_sample--;
            }
            n_sample = next_n_sample;

            height_++;
            while (n_sample > 0) {
                //sample to the next layer
                std::vector<int> tmp_cand_l(n_data_item_); // record which candidates are remained in this layer
                for (int sampleID = 0, candID = node_size_; sampleID < n_sample; sampleID++, candID += node_size_ + 1) {
                    int itemID = cand_l[candID];
                    tmp_cand_l[sampleID] = itemID;
                    tree_height_l_[itemID] = height_;
                    assert(candID < n_cand);
                }
                //copy to the data
                for (int sampleID = 0; sampleID < n_sample; sampleID++) {
                    int itemID = tmp_cand_l[sampleID];
                    cand_l[sampleID] = itemID;
                }
                n_cand = n_sample;

                next_n_sample = n_sample / (node_size_ + 1);
                next_n_remain = n_sample % (node_size_ + 1);
                if (next_n_remain != node_size_ - 1 && next_n_remain != node_size_) {
                    next_n_sample--;
                }
                n_sample = next_n_sample;

                height_++;
            }

            n_memory_layer_ = height_ - n_disk_layer_;

            layer_size_l_.resize(height_);
            layer_size_l_.assign(height_, 0);
            layer_size_l_[height_ - 1] = n_data_item_;
            for (int itemID = 0; itemID < n_data_item_; itemID++) {
                int rev_height = tree_height_l_[itemID];
                int height = height_ - 1 - rev_height;
                if (height != height_ - 1) {
                    layer_size_l_[height]++;
                }
            }

            if (n_disk_layer_ >= height_) {
                spdlog::error("height lower than n_disk_layer, program exit");
                exit(-1);
            }

            assert(height_ == layer_size_l_.size());

            memory_cum_layer_size_l_.emplace_back(layer_size_l_[0]);
            for (int layer = 1; layer < n_memory_layer_; layer++) {
                int n_ip_cum_layer = memory_cum_layer_size_l_[layer - 1] + layer_size_l_[layer];
                memory_cum_layer_size_l_.emplace_back(n_ip_cum_layer);
            }
            memory_bp_tree_.resize(memory_cum_layer_size_l_[n_memory_layer_ - 1] * n_user_);

            disk_cum_layer_size_l_.emplace_back(layer_size_l_[n_memory_layer_]);
            for (int layer = 1; layer < n_disk_layer_; layer++) {
                int current_tree_layer = n_memory_layer_ + layer;
                int n_ip_cum_layer = disk_cum_layer_size_l_[layer - 1] + layer_size_l_[current_tree_layer];
                disk_cum_layer_size_l_.emplace_back(n_ip_cum_layer);
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

            layer_rank_bound_l_.resize(height_);
            layer_rank_bound_l_.assign(height_, 1);
            for (int layer = 0; layer < height_ - 1; layer++) {
                layer_rank_bound_l_[layer] = node_size_;
                for (int add_layer = layer + 1; add_layer < height_ - 1; add_layer++) {
                    layer_rank_bound_l_[layer] *= (node_size_ + 1);
                }
            }

            std::cout << "total height: " << height_ << std::endl;

            //print tree height
//            std::cout << "height 0: all data item" << std::endl;
//            for (int heightID = 1; heightID < height_; heightID++) {
//                std::cout << "height " << heightID << ":" << std::endl;
//                for (int itemID = 0; itemID < n_data_item_; itemID++) {
//                    if (tree_height_l_[itemID] == heightID) {
//                        std::cout << itemID << " ";
//                    } else if (tree_height_l_[itemID] > heightID) {
//                        std::cout << RED << itemID << RESET << " ";
//                    }
//                }
//                std::cout << std::endl;
//            }

            std::cout << "layer_size_l_: ";
            for (int layer = 0; layer < height_; layer++) {
                std::cout << layer_size_l_[layer] << " ";
            }
            std::cout << std::endl;

            std::cout << "layer_rank_bound_l_: ";
            for (int layer = 0; layer < height_; layer++) {
                std::cout << layer_rank_bound_l_[layer] << " ";
            }
            std::cout << std::endl;

            spdlog::info(
                    "disk_tree_size_byte_ {}, n_memory_element_ {}, height_ {}, n_data_item_ {}, n_user_ {}, node_size_ {}, n_disk_layer_ {}, n_memory_layer_ {}",
                    disk_tree_size_byte_, n_memory_element_, height_, n_data_item_, n_user_, node_size_, n_disk_layer_,
                    n_memory_layer_);

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

            int test_n_node = 0;
            for (int layer = 0; layer < height_; layer++) {
                test_n_node += layer_size_l_[layer];
            }
            assert(test_n_node ==
                   memory_cum_layer_size_l_[n_memory_layer_ - 1] + disk_cum_layer_size_l_[n_disk_layer_ - 1]);

        }

        void BuildWriteTree(const double *IP_vecs, const int &userID) {
            //node from the bottom to the top, idx 0 is the bottom
            std::vector<std::vector<double>> layer_l;

            for (int layer = 0; layer < height_; layer++) {
                int n_layer_ele = layer_size_l_[layer];
                std::vector<double> layer_ele(n_layer_ele);
                if (layer == height_ - 1) {
                    //assign the bottom
                    assert(n_layer_ele == n_data_item_);
                    layer_ele.assign(IP_vecs, IP_vecs + n_data_item_);
                } else {
                    int n_cpy = 0;
                    for (int itemID = 0; itemID < n_data_item_; itemID++) {
                        if (height_ - 1 - tree_height_l_[itemID] == layer) {
                            layer_ele[n_cpy] = IP_vecs[itemID];
                            n_cpy++;
                        }
                    }
                    assert(n_cpy == n_layer_ele);
                }

                layer_l.push_back(layer_ele);
            }

            assert(layer_l.size() == height_);

            //write the layer to the memory index
            size_t offset = userID * n_memory_element_;
            for (int layer = 0; layer < n_memory_layer_; layer++) {
                std::vector<double> layer_ip = layer_l[layer];
                assert(layer_ip.size() == layer_size_l_[layer]);
                const int layer_size = layer_size_l_[layer];
                for (int eleID = 0; eleID < layer_size; eleID++) {
                    memory_bp_tree_[offset] = layer_ip[eleID];
                    offset++;
                }

            }

            // write the layer to the disk
            for (int layer = n_memory_layer_; layer < height_; layer++) {
                std::vector<double> layer_ip = layer_l[layer];
                assert(layer_ip.size() == layer_size_l_[layer]);
                out_stream_.write((char *) layer_ip.data(), sizeof(double) * layer_ip.size());
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
//            spdlog::info("index size {} byte", fsize);
        }

        void
        RetrievalMemory(const std::vector<double> &queryIP_l, std::vector<int> &rank_lb_l, std::vector<int> &rank_ub_l,
                        int queryID) {
            assert(queryIP_l.size() == n_user_ && rank_lb_l.size() == n_user_);
            is_all_last_node_l_.assign(n_user_, true);

            for (int userID = 0; userID < n_user_; userID++) {
                const double &queryIP = queryIP_l[userID];
                size_t base_offset = userID * n_memory_element_;
                assert(in_stream_);
                node_offset_l_[userID] = 0;

                for (int layer = 0; layer < n_memory_layer_; layer++) {
                    int layer_offset = layer == 0 ? 0 : memory_cum_layer_size_l_[layer - 1];
                    int block_offset = node_offset_l_[userID];
                    size_t offset = base_offset + layer_offset + block_offset;

                    int normal_read_count = node_size_;
                    int remain_read_count = layer_size_l_[layer] - node_offset_l_[userID];
                    assert(remain_read_count >= 0);
                    int read_count = normal_read_count;
                    if (normal_read_count > layer_size_l_[layer]) {
                        read_count = remain_read_count;
                    } else if (node_size_ - 1 <= remain_read_count && remain_read_count <= 2 * node_size_ - 1) {
                        read_count = remain_read_count;
                    }
                    assert(read_count > 0);
                    assert(0 <= block_offset + read_count && block_offset + read_count <= layer_size_l_[layer]);

                    assert(offset + read_count <= (userID + 1) * n_memory_element_);
                    memcpy(item_cache_.data(), memory_bp_tree_.data() + offset, read_count * sizeof(double));
                    int loc_rk = BinarySearch(queryIP, read_count);
                    rank_ub_l[userID] = rank_ub_l[userID] + loc_rk * layer_rank_bound_l_[layer];
                    if (loc_rk != read_count) {
                        is_all_last_node_l_[userID] = false;
                    }
                    if (is_all_last_node_l_[userID]) {
                        rank_lb_l[userID] = n_data_item_;
                    } else {
                        rank_lb_l[userID] = std::min(rank_ub_l[userID] + layer_rank_bound_l_[layer],
                                                     rank_lb_l[userID]);
                    }

                    if (n_disk_layer_ == 1 && layer == n_memory_layer_ - 1) {
                        //the whole last layer, case in memory
                        node_offset_l_[userID] += loc_rk;
                    } else {
                        node_offset_l_[userID] = node_offset_l_[userID] * (node_size_ + 1) + loc_rk * node_size_;
                    }

                    assert(0 <= rank_ub_l[userID] &&
                           rank_ub_l[userID] <= rank_lb_l[userID] &&
                           rank_lb_l[userID] <= n_data_item_);

                    assert(0 <= node_offset_l_[userID] && node_offset_l_[userID] <= layer_size_l_[layer + 1]);
                }

                assert(in_stream_);
                assert(node_offset_l_[userID] >= 0);
            }

        }

        void
        RetrievalDisk(const std::vector<double> &queryIP_l, const std::vector<bool> &prune_l,
                      const std::vector<int> &rank_lb_l, const std::vector<int> &rank_ub_l, int queryID) {
            assert(queryIP_l.size() == n_user_);
            int n_candidate = 0;
            for (int userID = 0; userID < n_user_; userID++) {
                if (prune_l[userID]) {
                    continue;
                }
                const double &queryIP = queryIP_l[userID];
                size_t base_offset = userID * disk_tree_size_byte_;
                assert(in_stream_);
                int rank_lb = rank_lb_l[userID];
                int rank_ub = rank_ub_l[userID];

                for (int disk_layer = 0; disk_layer < n_disk_layer_ - 1; disk_layer++) {
                    int layer_offset = disk_layer == 0 ? 0 : disk_cum_layer_size_l_[disk_layer - 1];
                    int block_offset = node_offset_l_[userID];
                    size_t offset = base_offset + (layer_offset + block_offset) * sizeof(double);

                    int normal_read_count = node_size_;
                    int remain_read_count = layer_size_l_[n_memory_layer_ + disk_layer] - node_offset_l_[userID];
                    assert(remain_read_count >= 0);
                    int read_count = normal_read_count;
                    if (normal_read_count > layer_size_l_[n_memory_layer_ + disk_layer]) {
                        read_count = remain_read_count;
                    } else if (node_size_ - 1 <= remain_read_count && remain_read_count <= 2 * node_size_ - 1) {
                        read_count = remain_read_count;
                    }
                    assert(read_count > 0);

                    int read_byte = read_count * sizeof(double);
                    assert(offset + read_byte <= (userID + 1) * disk_tree_size_byte_);
                    read_disk_record_.reset();
                    in_stream_.seekg((int64_t) offset, std::ios::beg);
                    in_stream_.read((char *) item_cache_.data(), (int64_t) read_byte);
                    read_disk_time_ += read_disk_record_.get_elapsed_time_second();

                    exact_rank_refinement_record_.reset();
                    int loc_rk = BinarySearch(queryIP, read_count);
                    rank_ub = rank_ub + loc_rk * layer_rank_bound_l_[disk_layer + n_memory_layer_];
                    if (loc_rk != read_count) {
                        is_all_last_node_l_[userID] = false;
                    }
                    if (is_all_last_node_l_[userID]) {
                        rank_lb = n_data_item_;
                    } else {
                        rank_lb = std::min(rank_ub + layer_rank_bound_l_[disk_layer + n_memory_layer_], rank_lb);
                    }

                    if (disk_layer == n_disk_layer_ - 2) {
                        //the whole last layer, case in memory
                        node_offset_l_[userID] += loc_rk;
                    } else {
                        node_offset_l_[userID] = node_offset_l_[userID] * (node_size_ + 1) + loc_rk * node_size_;
                    }

                    exact_rank_refinement_time_ += exact_rank_refinement_record_.get_elapsed_time_second();
                    assert(0 <= node_offset_l_[userID] &&
                           node_offset_l_[userID] < layer_size_l_[disk_layer + 1 + n_memory_layer_]);

                }

                assert(0 <= rank_ub && rank_ub <= rank_lb && rank_lb <= n_data_item_);
                int rank = rank_ub;
                int read_count = rank_lb - rank_ub;
                {
                    int layer_offset = n_disk_layer_ - 1 == 0 ? 0 : disk_cum_layer_size_l_[n_disk_layer_ - 2];
                    int block_offset = rank;
                    size_t offset = base_offset + (layer_offset + block_offset) * sizeof(double);

                    assert(read_count <= item_cache_.size());

                    int read_byte = read_count * sizeof(double);
                    assert(offset + read_byte <= (userID + 1) * disk_tree_size_byte_);
                    read_disk_record_.reset();
                    in_stream_.seekg((int64_t) offset, std::ios::beg);
                    in_stream_.read((char *) item_cache_.data(), (int64_t) read_byte);
                    read_disk_time_ += read_disk_record_.get_elapsed_time_second();

                    exact_rank_refinement_record_.reset();
                    int arr_size = (int) (read_byte / sizeof(double));
                    rank += BinarySearch(queryIP, arr_size);

                    assert(0 <= rank && rank <= layer_size_l_[n_memory_layer_ + n_disk_layer_ - 1]);
                    exact_rank_refinement_time_ += exact_rank_refinement_record_.get_elapsed_time_second();
                }

                assert(in_stream_);
                assert(rank >= 0);
                rank += 1;

                user_topk_cache_l_[n_candidate] = UserRankElement(userID, rank, queryIP);
                n_candidate++;
            }

            std::sort(user_topk_cache_l_.begin(), user_topk_cache_l_.begin() + n_candidate,
                      std::less());

        }

        void FinishRetrieval() {
            this->in_stream_.close();
        }

    };

    class Index : public BaseIndex {
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

        std::vector<std::vector<UserRankElement>> Retrieval(const VectorMatrix &query_item, const int &topk) override {
            ResetTimer();

            if (topk > user_.n_vector_) {
                spdlog::error("top-k is too large, program exit");
                exit(-1);
            }

            tree_ins_.RetrievalPreprocess();

            int n_query_item = query_item.n_vector_;

            std::vector<std::vector<UserRankElement>> query_heap_l(n_query_item);
            for (int qID = 0; qID < n_query_item; qID++) {
                query_heap_l[qID].reserve(topk);
            }

            std::vector<double> queryIP_l(n_user_);
            std::vector<int> rank_lb_l(n_user_);
            std::vector<int> rank_ub_l(n_user_);
            std::vector<int> topk_lb_heap(topk);
            std::vector<bool> prune_l(n_user_);

            for (int qID = 0; qID < n_query_item; qID++) {
                prune_l.assign(n_user_, false);
                rank_lb_l.assign(n_user_, n_data_item_);
                rank_ub_l.assign(n_user_, 0);

                //calculate distance
                double *query_item_vec = query_item.getVector(qID);
                inner_product_record_.reset();
                for (int userID = 0; userID < n_user_; userID++) {
                    double *user_vec = user_.getVector(userID);
                    double queryIP = InnerProduct(query_item_vec, user_vec, vec_dim_);
                    queryIP_l[userID] = queryIP;
                }
                this->inner_product_time_ += inner_product_record_.get_elapsed_time_second();

                rank_bound_prune_record_.reset();
                tree_ins_.RetrievalMemory(queryIP_l, rank_lb_l, rank_ub_l, qID);
                PruneCandidateByBound(rank_lb_l, rank_ub_l,
                                      n_user_, topk,
                                      prune_l, topk_lb_heap);
                rank_bound_prune_time_ += rank_bound_prune_record_.get_elapsed_time_second();
                int n_candidate = 0;
                for (int userID = 0; userID < n_user_; userID++) {
                    if (!prune_l[userID]) {
                        n_candidate++;
                    }
                }
                assert(n_candidate >= topk);
                rank_prune_ratio_ += 1.0 * (n_user_ - n_candidate) / n_user_;

                //read disk and fine binary search
                tree_ins_.RetrievalDisk(queryIP_l, prune_l, rank_lb_l, rank_ub_l, qID);

                for (int candID = 0; candID < topk; candID++) {
                    query_heap_l[qID].emplace_back(tree_ins_.user_topk_cache_l_[candID]);
                }
                assert(query_heap_l[qID].size() == topk);
            }

            read_disk_time_ = tree_ins_.read_disk_time_;
            exact_rank_refinement_time_ = tree_ins_.exact_rank_refinement_time_;

            tree_ins_.FinishRetrieval();
            rank_prune_ratio_ /= n_query_item;
            return query_heap_l;
        }

        std::string VariancePerformanceMetricName() override {
            return "queryID, retrieval time, second per query, rank prune ratio";
        }

        std::string VariancePerformanceStatistics(
                const double &retrieval_time, const double &second_per_query, const int &queryID) override {
            char str[256];
            sprintf(str, "%d,%.3f,%.3f,%.3f", queryID, retrieval_time, second_per_query, rank_prune_ratio_);
            return str;
        };

        std::string
        PerformanceStatistics(const int &topk, const double &retrieval_time, const double &ms_per_query) override {
            // int topk;
            //double total_time,
            //          inner_product_time, read_disk_time, binary_search_time;
            //double ms_per_query;
            //unit: second

            char buff[1024];

            sprintf(buff,
                    "top%d retrieval time:\n\ttotal %.3fs\n\tinner product %.3fs, rank bound pruning %.3fs\n\tread disk %.3fs, exact rank refinement %.3fs\n\trank prune ratio %.4f\n\tmillion second per query %.3fms",
                    topk, retrieval_time,
                    inner_product_time_, rank_bound_prune_time_,
                    read_disk_time_, exact_rank_refinement_time_,
                    rank_prune_ratio_,
                    ms_per_query);
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

#endif //REVERSE_KRANKS_BPLUSTREE_HPP
