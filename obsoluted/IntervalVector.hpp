//
// Created by BianZheng on 2022/1/25.
//

#ifndef REVERSE_KRANKS_INTERVALVECTOR_HPP
#define REVERSE_KRANKS_INTERVALVECTOR_HPP

#include "alg/SpaceInnerProduct.hpp"
#include "KMeans.hpp"
#include "struct/DistancePair.hpp"
#include "struct/MethodBase.hpp"
#include <vector>
#include <cfloat>
#include <cstring>
#include <set>
#include <algorithm>
#include <fstream>

namespace ReverseMIPS::IntervalVector {

    class RetrievalResult {
    public:
        //unit: second
        double total_time, inner_product_time, brute_force_search_time, second_per_query;
        int topk;

        inline RetrievalResult(double total_time, double inner_product_time, double brute_force_search_time,
                               double second_per_query, int topk) {
            this->total_time = total_time;
            this->inner_product_time = inner_product_time;
            this->brute_force_search_time = brute_force_search_time;
            this->second_per_query = second_per_query;

            this->topk = topk;
        }

        void AddMap(std::map<std::string, std::string> &performance_m) {
            char buff[256];
            sprintf(buff, "top%d total retrieval time", topk);
            std::string str1(buff);
            performance_m.emplace(str1, double2string(total_time));

            sprintf(buff, "top%d inner product time", topk);
            std::string str2(buff);
            performance_m.emplace(str2, double2string(inner_product_time));

            sprintf(buff, "top%d brute force search time", topk);
            std::string str3(buff);
            performance_m.emplace(str3, double2string(brute_force_search_time));

            sprintf(buff, "top%d second per query time", topk);
            std::string str4(buff);
            performance_m.emplace(str4, double2string(second_per_query));
        }

        [[nodiscard]] std::string ToString() const {
            char arr[256];
            sprintf(arr,
                    "top%d retrieval time:\n\ttotal %.3fs, inner product search %.3fs\n\tbrute force search time %.3fs, million second per query %.3fms",
                    topk, total_time, inner_product_time, brute_force_search_time, second_per_query * 1000);
            std::string str(arr);
            return str;
        }

    };

    class IntervalVector {
    public:
        std::vector<std::set<int>> tmp_set_vector_;
        std::vector<std::vector<int>> set_vector_;
        int n_interval_;
        double lower_bound_inner_product_, upper_bound_inner_product_;
        double interval_width_;

        inline IntervalVector() = default;

        inline IntervalVector(const int dimension, const double lb_ip, const double ub_ip) {
            n_interval_ = dimension;
            this->lower_bound_inner_product_ = lb_ip - 0.01;
            this->upper_bound_inner_product_ = ub_ip + 0.01;
            this->set_vector_.resize(dimension);
            this->tmp_set_vector_.resize(dimension);
            this->interval_width_ = (upper_bound_inner_product_ - lower_bound_inner_product_) / n_interval_;
        }

        inline ~IntervalVector() = default;

        [[nodiscard]] std::vector<int> GetIntervalByIP(double inner_product) const {
            int index = std::floor((upper_bound_inner_product_ - inner_product) / interval_width_);
            if (index < 0 || index >= n_interval_) {
                return std::vector<int>{};
            }
            return set_vector_[index];
        }

        [[nodiscard]] int GetIntervalIndexByIP(double inner_product) const {
            int index = std::floor((upper_bound_inner_product_ - inner_product) / interval_width_);
            if (index < 0) {
                return 0;
            }
            if (index >= n_interval_) {
                return n_interval_;
            }
            return index;
        }

        [[nodiscard]] double GetUpperBoundByIP(double inner_product) const {
            int index = std::floor((upper_bound_inner_product_ - inner_product) / interval_width_);
            if (index < 0) {
                return DBL_MAX;
            } else if (index >= n_interval_) {
                return lower_bound_inner_product_;
            } else {
                return upper_bound_inner_product_ - interval_width_ * index;
            }
        }

        int AddUniqueElement(double inner_product, int itemID) {
            int idx = std::floor((upper_bound_inner_product_ - inner_product) / interval_width_);
            if (tmp_set_vector_[idx].find(itemID) == tmp_set_vector_[idx].end()) {
                tmp_set_vector_[idx].insert(itemID);
            }
            return idx;
        }

        void StopAddUniqueElement() {
            for (int i = 0; i < n_interval_; i++) {
                std::set<int> &tmp_itemID_set = tmp_set_vector_[i];
                std::vector<int> &tmp_itemID_arr = set_vector_[i];
                for (auto &itemID: tmp_itemID_set) {
                    tmp_itemID_arr.emplace_back(itemID);
                }

                std::sort(tmp_itemID_arr.begin(), tmp_itemID_arr.end(), std::greater<int>());
                tmp_itemID_set.clear();
            }

        }


    };

    class Index : public BaseIndex {
    private:
        void ResetTime() {
            self_inner_product_time_ = 0;
            brute_force_search_time_ = 0;
        }

    public:
        std::vector<int> user_merge_idx_l_; // shape: n_user, record which user belongs to which IntervalVector
        std::vector<IntervalVector> interval_vector_l_; // shape: n_merge_user
        // IntervalVector shape: n_interval * unknown_size
        std::vector<int> interval_size_l_;
        /*
         * shape: n_user * (n_interval - 1), for each user, record the lower bound of rank in each interval
         * start record from the second interval since the first is always 0
         */
        int n_user_, n_merge_user_, n_interval_;
        VectorMatrix user_, data_item_;
        double self_inner_product_time_, brute_force_search_time_;
        TimeRecord record;

        inline Index(const std::vector<int> &user_merge_idx_l,
                     const std::vector<IntervalVector> &interval_vector_l,
                     const std::vector<int> &interval_size_l) {
            this->user_merge_idx_l_ = user_merge_idx_l;
            this->interval_vector_l_ = interval_vector_l;
            this->n_user_ = (int) user_merge_idx_l.size();
            this->n_merge_user_ = (int) interval_vector_l.size();
            this->n_interval_ = interval_vector_l[0].n_interval_;
            this->interval_size_l_ = interval_size_l;
        }

        void setUserItemMatrix(VectorMatrix &user, VectorMatrix &data_item) {
            this->user_ = user;
            this->data_item_ = data_item;
        }

        [[nodiscard]] std::vector<std::vector<UserRankElement>> Retrieval(VectorMatrix &query_item, int topk) override {
            if (topk > user_.n_vector_) {
                printf("top-k is larger than user, system exit\n");
                exit(-1);
            }
            ResetTime();
            std::vector<std::vector<UserRankElement>> result(query_item.n_vector_, std::vector<UserRankElement>());
            int n_query = query_item.n_vector_;
            int vec_dim = query_item.vec_dim_;
            for (int qID = 0; qID < n_query; qID++) {
                double *query_vec = query_item.getVector(qID);

                std::vector<UserRankElement> &minHeap = result[qID];
                minHeap.resize(topk);

                for (int userID = 0; userID < topk; userID++) {
                    int rank = GetRank(userID, query_vec, vec_dim);
                    UserRankElement rankElement(userID, rank);
                    minHeap[userID] = rankElement;
                }

                std::make_heap(minHeap.begin(), minHeap.end(), std::less<UserRankElement>());

                UserRankElement minHeapEle = minHeap.front();
                for (int userID = topk; userID < n_user_; userID++) {
                    int tmpRank = GetRank(userID, query_vec, vec_dim);

                    UserRankElement rankElement(userID, tmpRank);
                    if (minHeapEle.rank_ > rankElement.rank_) {
                        std::pop_heap(minHeap.begin(), minHeap.end(), std::less<UserRankElement>());
                        minHeap.pop_back();
                        minHeap.push_back(rankElement);
                        std::push_heap(minHeap.begin(), minHeap.end(), std::less<UserRankElement>());
                        minHeapEle = minHeap.front();
                    }

                }
                std::make_heap(minHeap.begin(), minHeap.end(), std::less<UserRankElement>());
                std::sort_heap(minHeap.begin(), minHeap.end(), std::less<UserRankElement>());
            }
            return result;
        }

        [[nodiscard]] int RelativeRankInInterval(const std::vector<int> &candidate_l, double queryIP,
                                                 int userID, int vec_dim, double upper_bound_ip) const {
            //calculate all the IP, then get the lower bound
            //make different situation by the information
            int interval_size = (int) candidate_l.size();
            int rank = 0;

            for (int i = 0; i < interval_size; i++) {
                double data_dist = InnerProduct(data_item_.getVector(candidate_l[i]), user_.getVector(userID), vec_dim);
                rank += data_dist > queryIP && data_dist < upper_bound_ip ? 1 : 0;
            }

            return rank;
        }

        [[nodiscard]] int GetRank(const int userID, const double *query_vec, const int vec_dim) {
            double *user_vec = this->user_.getVector(userID);
            record.reset();
            double queryIP = InnerProduct(query_vec, user_vec, vec_dim);
            self_inner_product_time_ += record.get_elapsed_time_second();

            int interval_vector_idx = user_merge_idx_l_[userID];
            std::vector<int> candidate_l = interval_vector_l_[interval_vector_idx].GetIntervalByIP(queryIP);
            double upper_bound_ip = interval_vector_l_[interval_vector_idx].GetUpperBoundByIP(queryIP);
            int intervalID = interval_vector_l_[interval_vector_idx].GetIntervalIndexByIP(queryIP);
            record.reset();
            int loc_rk = RelativeRankInInterval(candidate_l, queryIP, userID, vec_dim, upper_bound_ip);
            brute_force_search_time_ += record.get_elapsed_time_second();

            int rank;
            if (intervalID == n_interval_) {
                rank = data_item_.n_vector_;
            } else {
                rank = intervalID == 0 ? loc_rk : interval_size_l_[userID * (n_interval_ - 1) + intervalID - 1] +
                                                  loc_rk;
            }
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
            out.write((char *) distance_cache.data(), distance_cache.size() * sizeof(DistancePair));

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
                  n_remain * data_item.n_vector_ * sizeof(DistancePair));
    }

    void BuildIntervalVectorIndex(const std::vector<int> &user_merge_idx_l, const char *index_path,
                                  std::vector<IntervalVector> &itv_vec_l, std::vector<int> &interval_size_l,
                                  const int n_interval) {
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
                int itv_l_idx = user_merge_idx_l[userID];
                IntervalVector &tmp_itv_vec = itv_vec_l[itv_l_idx];

                // get index of every data item, used for calculate the size of interval in each user
                std::vector<int> itv_idx_l(n_data_item);
                std::memset(itv_idx_l.data(), 0, sizeof(int) * n_data_item);
                for (int d_itemID = 0; d_itemID < n_data_item; d_itemID++) {
                    int dp_idx = cacheID * n_data_item + d_itemID;
                    const DistancePair &dp = distance_cache[dp_idx];
                    int itv_idx = tmp_itv_vec.AddUniqueElement(dp.dist_, dp.ID_);
                    itv_idx_l[d_itemID] = itv_idx;
                }

                int size_count = 0;
                int itv_ptr = 0;
                int d_itemID = 0;
                if (itv_idx_l[d_itemID] != itv_ptr) {
                    for (; itv_ptr < itv_idx_l[d_itemID]; itv_ptr++) {
                        interval_size_l[userID * (n_interval - 1) + itv_ptr] = size_count;
                    }
                } else {
                    size_count++;
                }
                for (d_itemID = 1; d_itemID < n_data_item; d_itemID++) {
                    if (itv_ptr < itv_idx_l[d_itemID]) {
                        interval_size_l[userID * (n_interval - 1) + itv_ptr] = size_count;
                        itv_ptr++;
                    }
                    size_count++;
                }
                for (; itv_ptr < n_interval - 1; itv_ptr++) {
                    interval_size_l[userID * (n_interval - 1) + itv_ptr] = size_count;
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
            int itv_l_idx = user_merge_idx_l[userID];
            IntervalVector &tmp_itv_vec = itv_vec_l[itv_l_idx];
            // get index of every data item, used for calculate the size of interval in each user
            std::vector<int> itv_idx_l(n_data_item);
            std::memset(itv_idx_l.data(), 0, sizeof(int) * n_data_item);

            for (int d_itemID = 0; d_itemID < n_data_item; d_itemID++) {
                int dp_idx = cacheID * n_data_item + d_itemID;
                const DistancePair &dp = distance_cache[dp_idx];
                int itv_idx = tmp_itv_vec.AddUniqueElement(dp.dist_, dp.ID_);
                itv_idx_l[d_itemID] = itv_idx;
            }

            int size_count = 0;
            int itv_ptr = 0;
            int d_itemID = 0;
            if (itv_idx_l[d_itemID] != itv_ptr) {
                for (; itv_ptr < itv_idx_l[d_itemID]; itv_ptr++) {
                    interval_size_l[userID * (n_interval - 1) + itv_ptr] = size_count;
                }
            } else {
                size_count++;
            }
            for (d_itemID = 1; d_itemID < n_data_item; d_itemID++) {
                if (itv_ptr < itv_idx_l[d_itemID]) {
                    interval_size_l[userID * (n_interval - 1) + itv_ptr] = size_count;
                    itv_ptr++;
                }
                size_count++;
            }
            for (; itv_ptr < n_interval - 1; itv_ptr++) {
                interval_size_l[userID * (n_interval - 1) + itv_ptr] = size_count;
            }

            if (userID % report_user_every_ == 0) {
                std::cout << "read and process interval vector " << userID / (0.01 * n_user) << " %, "
                          << batch_report_record.get_elapsed_time_second() << " s/iter" << " Mem: "
                          << get_current_RSS() / 1000000 << " Mb \n";
                batch_report_record.reset();
            }
        }

        for (auto &tmp_itv_vec: itv_vec_l) {
            tmp_itv_vec.StopAddUniqueElement();
        }

    }


    Index BuildIndex(VectorMatrix &user, VectorMatrix &data_item, int n_merge_user, const char *dataset_name,
                     std::vector<double> &component_time_l) {

        int n_user = user.n_vector_;
        int n_data_item = data_item.n_vector_;
        int vec_dim = user.vec_dim_;

        int n_interval = std::min(n_data_item / 2, 20000);

        //perform Kmeans for user vector, the label start from 0, indicates where the rank should come from
        printf("n_merge_user %d\n", n_merge_user);
        std::vector<int> user_merge_idx_l = BuildKMeans(user, n_merge_user);

        char index_path[256];
        sprintf(index_path, "../index/%s.itv_vec_idx", dataset_name);

        //left: lower bound, right: upper bound
        std::vector<std::pair<double, double>> ip_bound_l(n_merge_user, std::pair<double, double>(DBL_MAX, -DBL_MAX));

        TimeRecord record;
        BuildSaveBruteForceIndex(user, data_item, index_path, user_merge_idx_l, ip_bound_l);
        double bruteforce_index_time = record.get_elapsed_time_second();
        component_time_l.push_back(bruteforce_index_time);

        std::vector<IntervalVector> itv_vec_l;
        for (std::pair<double, double> tmp_bound: ip_bound_l) {
            double lb = tmp_bound.first;
            double ub = tmp_bound.second;
            itv_vec_l.emplace_back(n_interval, lb, ub);
        }
        std::vector<int> interval_size_l(n_user * (n_interval - 1));
        BuildIntervalVectorIndex(user_merge_idx_l, index_path, itv_vec_l, interval_size_l, n_interval);

        printf("n_interval %d\n", n_interval);

        Index index(user_merge_idx_l, itv_vec_l, interval_size_l);
        index.setUserItemMatrix(user, data_item);

        return index;
    }
}
#endif //REVERSE_KRANKS_INTERVALVECTOR_HPP
