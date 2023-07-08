//
// Created by BianZheng on 2022/5/17.
//

#ifndef REVERSE_KRANKS_GREEDYMERGEMINCLUSTERSIZE_HPP
#define REVERSE_KRANKS_GREEDYMERGEMINCLUSTERSIZE_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>
#include <spdlog/spdlog.h>
#include <queue>

#include "alg/SpaceEuclidean.hpp"
#include "struct/VectorMatrix.hpp"

namespace ReverseMIPS::GreedyMergeMinClusterSize {

    class PQCluster {
    public:
        int clusterID_;
        int cluster_size_;

        inline PQCluster() {
            this->clusterID_ = -1;
            this->cluster_size_ = -1;
        }

        inline PQCluster(const int &clusterID, const int &user_size) {
            this->clusterID_ = clusterID;
            this->cluster_size_ = user_size;
        }

    };

    std::pair<int, int>
    SelectCenterUser(const VectorMatrix &user, const int user_size, const std::vector<int> &candidate_l,
                     const int &vec_dim) {
        /**select the promising pair, return the userID of center point**/
        assert(user_size == candidate_l.size());
        /**对一整个数组进行partition, 分成两份, 将user数量最多的那一份放到最远的地方**/

        //choose two userID as the center, the first it randomly chose, then choose the user of the two polar point
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<> distrib(0, user_size - 1);
        int cuserID = candidate_l[distrib(gen)]; // center userID
        double max_dist = -1;
        int cuserID2 = -1;
        for (int candID = 0; candID < user_size; candID++) {
            int userID = candidate_l[candID];
            double dist = EuclideanDistance(user.getVector(userID), user.getVector(cuserID), vec_dim);
            if (max_dist < dist) {
                max_dist = dist;
                cuserID2 = userID;
            }
        }

        max_dist = -1;
        cuserID = -1;
        for (int candID = 0; candID < user_size; candID++) {
            int userID = candidate_l[candID];
            double dist = EuclideanDistance(user.getVector(userID), user.getVector(cuserID2), vec_dim);
            if (max_dist < dist) {
                max_dist = dist;
                cuserID = userID;
            }
        }
        assert(cuserID2 != -1 && cuserID != -1);
        return std::make_pair(cuserID, cuserID2);
    }

    void PartitionCandidate(const VectorMatrix &user, const std::vector<int> &candidate_l,
                            const std::pair<int, int> &cuserID_pair, const int &vec_dim,
                            std::vector<int> &candidate_l1, std::vector<int> &candidate_l2) {
        //candidate with larger user size should be the first
        int user_size = (int) candidate_l.size();
        int cuserID1 = cuserID_pair.first;
        int cuserID2 = cuserID_pair.second;
        int cuser1_size = 0;
        int cuser2_size = 0;
        for (int candID = 0; candID < user_size; candID++) {
            int userID = candidate_l[candID];
            const double *user_vecs = user.getVector(userID);
            const double *cuser1_vecs = user.getVector(cuserID1);
            const double *cuser2_vecs = user.getVector(cuserID2);
            double dist1 = EuclideanDistance(user_vecs, cuser1_vecs, vec_dim);
            double dist2 = EuclideanDistance(user_vecs, cuser2_vecs, vec_dim);
            if (dist1 > dist2) {
                cuser2_size++;
                candidate_l2.emplace_back(userID);
            } else {
                cuser1_size++;
                candidate_l1.emplace_back(userID);
            }
        }
        assert(candidate_l1.size() + candidate_l2.size() == candidate_l.size());
    }

    /*
     * self-implement greedy merge interface
     */
    std::vector<uint32_t> ClusterLabel(const VectorMatrix &vm, const int &n_cluster) {
        const int n_vector = vm.n_vector_;
        std::vector<uint32_t> label_l(n_vector, -1);

        /**先将所有的点分配到node, 计算半径, 赋值圆心**/
        //to store the candidate for each node
        auto PQNodeMaxHeap = [](const PQCluster &node, const PQCluster &other) {
            return node.cluster_size_ < other.cluster_size_;
        };

        std::priority_queue<PQCluster, std::vector<PQCluster>, decltype(PQNodeMaxHeap)> node_max_heap{PQNodeMaxHeap};
        node_max_heap.push(PQCluster(0, n_vector));
        PQCluster max_node = node_max_heap.top();

        std::vector<std::vector<int>> cluster_cand_l; //store the vecsID for a vector
        std::vector<int> vecs_idx_l(n_vector);
        std::iota(vecs_idx_l.begin(), vecs_idx_l.end(), 0);
        cluster_cand_l.push_back(vecs_idx_l);

        for (int cur_n_cluster = 1; cur_n_cluster < n_cluster; cur_n_cluster++) {
            assert(node_max_heap.size() == cluster_cand_l.size());
            int clusterID = max_node.clusterID_;
            std::vector<int> candidate_l = cluster_cand_l[clusterID];

            std::pair<int, int> cuserID_pair = SelectCenterUser(vm, max_node.cluster_size_, candidate_l, vm.vec_dim_);
            std::vector<int> candidate_l1;
            std::vector<int> candidate_l2;
            PartitionCandidate(vm, candidate_l, cuserID_pair, vm.vec_dim_, candidate_l1, candidate_l2);
            int nodeID1 = clusterID;
            int nodeID2 = (int) cluster_cand_l.size();
            cluster_cand_l[clusterID] = candidate_l1;
            cluster_cand_l.push_back(candidate_l2);

            node_max_heap.pop();
            node_max_heap.emplace(nodeID1, candidate_l1.size());
            node_max_heap.emplace(nodeID2, candidate_l2.size());
            max_node = node_max_heap.top();
        }

        assert(cluster_cand_l.size() == node_max_heap.size());
        //assign the result
        assert(n_cluster == cluster_cand_l.size());
        assert(n_cluster < n_vector);

        for (int clusterID = 0; clusterID < n_cluster; clusterID++) {
            const std::vector<int> &candidate_l = cluster_cand_l[clusterID];
            int n_cand = (int) candidate_l.size();
            for (int candID = 0; candID < n_cand; candID++) {
                int vecsID = candidate_l[candID];
                assert(0 <= vecsID && vecsID < n_vector);
                label_l[vecsID] = clusterID;
            }
        }

        for (int vecsID = 0; vecsID < n_vector; vecsID++) {
            assert(label_l[vecsID] != -1);
        }

        spdlog::info("Finish GreedyMerge");
        return label_l;
    }

}

#endif //REVERSE_KRANKS_GREEDYMERGEMINCLUSTERSIZE_HPP
