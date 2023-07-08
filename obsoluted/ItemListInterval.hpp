//
// Created by BianZheng on 2022/1/25.
//

#ifndef REVERSE_KRANKS_ITEMLISTINTERVAL_HPP
#define REVERSE_KRANKS_ITEMLISTINTERVAL_HPP

#include "struct/DistancePair_cpy.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "KMeans.hpp"
#include <vector>
#include <cfloat>
#include <cstring>
#include <fstream>

namespace ReverseMIPS {
    class ItemInfo {
    public:
        //bound variable stores the rank, lower_bound means the rank of this item should be at least this value; same as upper bound
        int lower_bound_, upper_bound_, itemID_;

        inline ItemInfo(int lb, int ub, int itemID) {
            this->lower_bound_ = lb;
            this->upper_bound_ = ub;
            this->itemID_ = itemID;
        }

        inline bool operator==(const ItemInfo &other) const {
            if (this == &other)
                return true;
            return lower_bound_ == other.lower_bound_ && upper_bound_ == other.upper_bound_ && itemID_ == other.itemID_;
        };

        inline bool operator!=(const ItemInfo &other) const {
            if (this == &other)
                return false;
            return lower_bound_ != other.lower_bound_ || upper_bound_ != other.upper_bound_ || itemID_ != other.itemID_;
        };

        inline bool operator<(const ItemInfo &other) const {
            if (upper_bound_ != other.upper_bound_) {
                return upper_bound_ < other.upper_bound_;
            } else {
                return lower_bound_ < other.lower_bound_;
            }
        }

        inline bool operator<=(const ItemInfo &other) const {
            if (upper_bound_ != other.upper_bound_) {
                return upper_bound_ <= other.upper_bound_;
            } else {
                return lower_bound_ <= other.lower_bound_;
            }
        }

        inline bool operator>(const ItemInfo &other) const {
            if (upper_bound_ != other.upper_bound_) {
                return upper_bound_ > other.upper_bound_;
            } else {
                return lower_bound_ > other.lower_bound_;
            }
        }

        inline bool operator>=(const ItemInfo &other) const {
            if (upper_bound_ != other.upper_bound_) {
                return upper_bound_ >= other.upper_bound_;
            } else {
                return lower_bound_ >= other.lower_bound_;
            }
        }

        std::string ToString() {
            char arr[256];
            sprintf(arr, "lbidx %d ubidx %d itemid %d", lower_bound_, upper_bound_, itemID_);
            std::string str(arr);
            return str;
        }
    };

    class ItemListInterval {
    public:
        double lower_bound_inner_product_, upper_bound_inner_product_;
        double interval_width_;
        int n_interval_, n_data_item_;
        //left is the lower bound, right is the upper bound
        std::vector<ItemInfo> item_list_;

        inline ItemListInterval() = default;

        inline ItemListInterval(const double lb_ip, const double ub_ip, const int n_interval, const int n_data_item) {
            this->n_interval_ = n_interval;
            this->n_data_item_ = n_data_item;
            this->lower_bound_inner_product_ = lb_ip - 0.01;
            this->upper_bound_inner_product_ = ub_ip + 0.01;
            this->interval_width_ = (upper_bound_inner_product_ - lower_bound_inner_product_) / n_interval;

            this->item_list_.reserve(n_data_item);
            for (int i = 0; i < n_data_item; i++) {
                this->item_list_.emplace_back(n_data_item + 1, -1, i);
            }

        }

        void UpdateItem(const DistancePair &dp) {
            int itemID = dp.ID_;
            double inner_product = dp.dist_;
            int idx = std::floor((upper_bound_inner_product_ - inner_product) / interval_width_);
            this->item_list_[itemID].lower_bound_ = std::min(this->item_list_[itemID].lower_bound_, idx);
            this->item_list_[itemID].upper_bound_ = std::max(this->item_list_[itemID].upper_bound_, idx);
        }

        void SortItemAndCountSize() {
            //the update function should be stopped
            //TODO.txt check whether it is all updated
            std::sort(item_list_.begin(), item_list_.end(), std::less<ItemInfo>());

//            for(int i=0;i<n_data_item_;i++){
//                std::cout << item_list_[i].ToString() << std::endl;
//            }
        }

        [[nodiscard]] std::vector<int> GetCandidateByIP(double queryIP) const {
            int idx = std::floor((upper_bound_inner_product_ - queryIP) / interval_width_);
            if (idx < 0 || idx >= n_interval_) {
                return std::vector<int>{};
            }

            std::vector<int> res;
            for (int i = 0; i < n_data_item_; i++) {
                if (item_list_[i].lower_bound_ <= idx && idx <= item_list_[i].upper_bound_) {
                    res.emplace_back(item_list_[i].itemID_);
                }
            }
            return res;
        }

        [[nodiscard]] int GetPruneSizeByIP(double inner_product) const {
            int index = std::floor((upper_bound_inner_product_ - inner_product) / interval_width_);
            if (index < 0) {
                return 0;
            } else if (index >= n_interval_) {
                return n_data_item_;
            } else {
                int size = 0;
                for (int i = 0; i < n_data_item_; i++) {
                    if (item_list_[i].upper_bound_ < index) {
                        size++;
                    }
                }
                return size;

//                auto lb_ptr = std::lower_bound(item_list_.begin(), item_list_.end(), index,
//                                               [](const ItemInfo &info, int value) {
//                                                   return info.upper_bound_ < value;
//                                               });
//                return (int) (lb_ptr - item_list_.begin());
            }
        }

        inline ~ItemListInterval() = default;
    };

    class ItemListIntervalIndex {
    public:

        std::vector<int> user_merge_idx_l_; // shape: n_user, record which user belongs to which ItemListInterval
        std::vector<ItemListInterval> it_ls_itv_l_; // shape: n_merge_user
        // ItemListInterval shape: n_data_item
        int n_user_, n_merge_user_, n_data_item_;
        VectorMatrix user_, data_item_;

        void setUserItemMatrix(VectorMatrix &user, VectorMatrix &data_item) {
            this->user_ = user;
            this->data_item_ = data_item;
        }

        inline ItemListIntervalIndex(std::vector<int> &user_merge_idx_l, std::vector<ItemListInterval> &it_ls_itv_l) {
            this->user_merge_idx_l_ = user_merge_idx_l;
            this->it_ls_itv_l_ = it_ls_itv_l;

            this->n_user_ = (int) user_merge_idx_l.size();
            this->n_merge_user_ = (int) it_ls_itv_l.size();
            this->n_data_item_ = it_ls_itv_l[0].n_data_item_;
        }

        [[nodiscard]] std::vector<std::vector<RankElement>> Retrieval(VectorMatrix &query_item, int topk) const {
            if (topk > user_.n_vector_) {
                printf("top-k is larger than user, system exit\n");
                exit(-1);
            }
            std::vector<std::vector<RankElement>> result(query_item.n_vector_, std::vector<RankElement>());
            int n_query = query_item.n_vector_;
            int vec_dim = query_item.vec_dim_;
            for (int qID = 0; qID < n_query; qID++) {
                double *query_vec = query_item.getVector(qID);

                std::vector<RankElement> &minHeap = result[qID];
                minHeap.resize(topk);

                for (int userID = 0; userID < topk; userID++) {
                    int rank = GetRank(userID, query_vec, vec_dim);
                    RankElement rankElement(userID, rank);
                    minHeap[userID] = rankElement;
                }

                std::make_heap(minHeap.begin(), minHeap.end(), std::less<RankElement>());

                RankElement minHeapEle = minHeap.front();
                for (int userID = topk; userID < n_user_; userID++) {
                    int tmpRank = GetRank(userID, query_vec, vec_dim);

                    RankElement rankElement(userID, tmpRank);
                    if (minHeapEle.rank_ > rankElement.rank_) {
                        std::pop_heap(minHeap.begin(), minHeap.end(), std::less<RankElement>());
                        minHeap.pop_back();
                        minHeap.push_back(rankElement);
                        std::push_heap(minHeap.begin(), minHeap.end(), std::less<RankElement>());
                        minHeapEle = minHeap.front();
                    }

                }
                std::make_heap(minHeap.begin(), minHeap.end(), std::less<RankElement>());
                std::sort_heap(minHeap.begin(), minHeap.end(), std::less<RankElement>());
            }
            return result;
        }

        [[nodiscard]] int RelativeRankInInterval(const std::vector<int> &candidate_l, double queryIP,
                                                 int userID, int vec_dim) const {
            //calculate all the IP, then get the lower bound
            //make different situation by the information
            int interval_size = (int) candidate_l.size();
            int rank = 0;

            for (int i = 0; i < interval_size; i++) {
                double data_dist = InnerProduct(data_item_.getVector(candidate_l[i]), user_.getVector(userID), vec_dim);
                rank += data_dist > queryIP? 1 : 0;
            }

            return rank;
        }

        [[nodiscard]] int GetRank(const int userID, const double *query_vec, const int vec_dim) const {
            double *user_vec = this->user_.getVector(userID);
            double queryIP = InnerProduct(query_vec, user_vec, vec_dim);

            int merge_l_idx = user_merge_idx_l_[userID];
            std::vector<int> candidate_l = it_ls_itv_l_[merge_l_idx].GetCandidateByIP(queryIP);
            int prune_size = it_ls_itv_l_[merge_l_idx].GetPruneSizeByIP(queryIP);
            int loc_rk = RelativeRankInInterval(candidate_l, queryIP, userID, vec_dim);

            int rank = loc_rk + prune_size;
            return rank + 1;
        }

    };

    /*
     * bruteforce index
     * shape: 1, type: int, n_user
     * shape: 1, type: int, n_data_item
     * shape: n_user * n_data_item, type: DistancePair, the distance pair for each user
     */
    void BuildSaveBruteForceIndex(const VectorMatrix &user, const VectorMatrix &data_item, const char *index_path,
                                  const std::vector<int> &user_merge_idx_l,
                                  std::vector<std::pair<double, double>> &ip_bound_l) {
        const int write_every_ = 10000;
        const int report_batch_every_ = 5;

        std::ofstream out(index_path, std::ios::binary | std::ios::out);
        if (!out) {
            std::printf("error in write result\n");
        }
        const int n_data_item = data_item.n_vector_;
        std::vector<DistancePair> distance_cache(write_every_ * n_data_item);
        const int vec_dim = data_item.vec_dim_;
        const int n_batch = user.n_vector_ / write_every_;
        const int n_remain = user.n_vector_ % write_every_;
        out.write((char *) &user.n_vector_, sizeof(int));
        out.write((char *) &n_data_item, sizeof(int));

        TimeRecord batch_report_record;
        batch_report_record.reset();
        for (int i = 0; i < n_batch; i++) {
            for (int cacheID = 0; cacheID < write_every_; cacheID++) {
                int userID = write_every_ * i + cacheID;
                for (int itemID = 0; itemID < n_data_item; itemID++) {
                    double ip = InnerProduct(data_item.getVector(itemID), user.getVector(userID), vec_dim);
                    distance_cache[cacheID * n_data_item + itemID] = DistancePair(ip, itemID);
                }
                std::sort(distance_cache.begin() + cacheID * n_data_item,
                          distance_cache.begin() + (cacheID + 1) * n_data_item, std::greater<DistancePair>());

                int merge_idx = user_merge_idx_l[userID];
                double ip_lower_bound = std::min(distance_cache[(cacheID + 1) * n_data_item - 1].dist_,
                                                 ip_bound_l[merge_idx].first);
                double ip_upper_bound = std::max(distance_cache[cacheID * n_data_item].dist_,
                                                 ip_bound_l[merge_idx].second);
                ip_bound_l[merge_idx].first = ip_lower_bound;
                ip_bound_l[merge_idx].second = ip_upper_bound;
            }
            out.write((char *) distance_cache.data(), (long) (distance_cache.size() * sizeof(DistancePair)));

            if (i % report_batch_every_ == 0) {
                std::cout << "preprocessed " << i / (0.01 * n_batch) << " %, "
                          << batch_report_record.get_elapsed_time_second() << " s/iter" << " Mem: "
                          << get_current_RSS() / 1000000 << " Mb \n";
                batch_report_record.reset();
            }

        }

        for (int cacheID = 0; cacheID < n_remain; cacheID++) {
            int userID = cacheID + write_every_ * n_batch;
            for (int itemID = 0; itemID < data_item.n_vector_; itemID++) {
                double ip = InnerProduct(data_item.rawData_ + itemID * vec_dim,
                                         user.rawData_ + userID * vec_dim, vec_dim);
                distance_cache[cacheID * data_item.n_vector_ + itemID] = DistancePair(ip, itemID);
            }

            std::sort(distance_cache.begin() + cacheID * n_data_item,
                      distance_cache.begin() + (cacheID + 1) * n_data_item, std::greater<DistancePair>());

            int merge_idx = user_merge_idx_l[userID];
            double ip_lower_bound = std::min(distance_cache[(cacheID + 1) * n_data_item - 1].dist_,
                                             ip_bound_l[merge_idx].first);
            double ip_upper_bound = std::max(distance_cache[cacheID * n_data_item].dist_,
                                             ip_bound_l[merge_idx].second);
            ip_bound_l[merge_idx].first = ip_lower_bound;
            ip_bound_l[merge_idx].second = ip_upper_bound;
        }

        out.write((char *) distance_cache.data(),
                  (int) n_remain * data_item.n_vector_ * sizeof(DistancePair));
    }

    void BuildItemListInterval(const std::vector<int> &user_merge_idx_l, const char *index_path,
                               std::vector<ItemListInterval> &ili_merge_l, const int n_interval) {
        std::ifstream index_stream_ = std::ifstream(index_path, std::ios::binary | std::ios::in);
        if (!index_stream_) {
            std::printf("error in writing index\n");
        }
        int n_user, n_data_item;
        index_stream_.read((char *) &n_user, sizeof(int));
        index_stream_.read((char *) &n_data_item, sizeof(int));

        const int n_cache = std::min(n_user, 10000);
        const int report_user_every_ = 10000;

        std::vector<DistancePair> distance_cache(n_cache * n_data_item);
        const int n_batch = n_user / n_cache;
        const int n_remain = n_user % n_cache;

        TimeRecord batch_report_record;
        batch_report_record.reset();
        for (int batchID = 0; batchID < n_batch; batchID++) {
            index_stream_.read((char *) distance_cache.data(), n_cache * n_data_item * sizeof(DistancePair));
            for (int cacheID = 0; cacheID < n_cache; cacheID++) {
                int userID = batchID * n_cache + cacheID;
                int merge_l_idx = user_merge_idx_l[userID];
                ItemListInterval &tmp_it_ls_itv = ili_merge_l[merge_l_idx];

                for (int d_itemID = 0; d_itemID < n_data_item; d_itemID++) {
                    int dp_idx = cacheID * n_data_item + d_itemID;
                    const DistancePair &dp = distance_cache[dp_idx];
                    tmp_it_ls_itv.UpdateItem(dp);
                }

                if (userID % report_user_every_ == 0) {
                    std::cout << "read and process interval vector " << userID / (0.01 * n_user) << " %, "
                              << batch_report_record.get_elapsed_time_second() << " s/iter" << " Mem: "
                              << get_current_RSS() / 1000000 << " Mb \n";
                    batch_report_record.reset();
                }
            }

        }

        index_stream_.read((char *) distance_cache.data(), n_remain * n_data_item * sizeof(DistancePair));
        for (int cacheID = 0; cacheID < n_remain; cacheID++) {
            int userID = n_batch * n_cache + cacheID;
            int merge_l_idx = user_merge_idx_l[userID];
            ItemListInterval &tmp_it_ls_itv = ili_merge_l[merge_l_idx];

            for (int d_itemID = 0; d_itemID < n_data_item; d_itemID++) {
                int dp_idx = cacheID * n_data_item + d_itemID;
                const DistancePair &dp = distance_cache[dp_idx];
                tmp_it_ls_itv.UpdateItem(dp);
            }

            if (userID % report_user_every_ == 0) {
                std::cout << "read and process interval vector " << userID / (0.01 * n_user) << " %, "
                          << batch_report_record.get_elapsed_time_second() << " s/iter" << " Mem: "
                          << get_current_RSS() / 1000000 << " Mb \n";
                batch_report_record.reset();
            }
        }

        for (auto &tmp_it_ls_itv: ili_merge_l) {
            tmp_it_ls_itv.SortItemAndCountSize();
        }

    }


    ItemListIntervalIndex
    BuildIndex(VectorMatrix &user, VectorMatrix &data_item, int n_merge_user, const char *dataset_name,
               std::vector<double> &component_time_l) {

        int n_user = user.n_vector_;
        int n_data_item = data_item.n_vector_;
        int vec_dim = user.vec_dim_;

        int n_interval = std::min(n_data_item / 10, 5);

        //perform Kmeans for user vector, the label start from 0, indicates where the rank should come from
        printf("n_merge_user %d\n", n_merge_user);
        std::vector<int> user_merge_idx_l = BuildKMeans(user, n_merge_user);

        char index_path[256];
        sprintf(index_path, "../index/%s.it_ls_itv_index", dataset_name);

        //left: lower bound, right: upper bound
        std::vector<std::pair<double, double>> ip_bound_l(n_merge_user, std::pair<double, double>(DBL_MAX, -DBL_MAX));

        TimeRecord record;
        BuildSaveBruteForceIndex(user, data_item, index_path, user_merge_idx_l, ip_bound_l);
        double bruteforce_index_time = record.get_elapsed_time_second();
        component_time_l.push_back(bruteforce_index_time);

        std::vector<ItemListInterval> it_ls_itv_l;
        for (std::pair<double, double> tmp_bound: ip_bound_l) {
            double lb = tmp_bound.first;
            double ub = tmp_bound.second;
            it_ls_itv_l.emplace_back(lb, ub, n_interval, n_data_item);
        }
        BuildItemListInterval(user_merge_idx_l, index_path, it_ls_itv_l, n_interval);

        printf("n_interval %d\n", n_interval);

        ItemListIntervalIndex itemListIntervalIndex(user_merge_idx_l, it_ls_itv_l);
        itemListIntervalIndex.setUserItemMatrix(user, data_item);

        return itemListIntervalIndex;
    }
}
#endif //REVERSE_KRANKS_ITEMLISTINTERVAL_HPP
