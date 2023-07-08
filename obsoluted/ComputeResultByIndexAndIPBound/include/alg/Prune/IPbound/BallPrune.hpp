//
// Created by BianZheng on 2022/3/22.
//

#ifndef REVERSE_KRANKS_BALLPRUNE_HPP
#define REVERSE_KRANKS_BALLPRUNE_HPP

#include <random>
#include <algorithm>
#include <vector>
#include <memory>
#include <queue>
#include <spdlog/spdlog.h>
#include "alg/SpaceEuclidean.hpp"
#include "alg/SpaceInnerProduct.hpp"
#include "struct/VectorMatrix.hpp"

namespace ReverseMIPS {

    class PQNode {
    public:
        int nodeID_;
        int user_size_;

        inline PQNode() {
            this->nodeID_ = -1;
            this->user_size_ = -1;
        }

        inline PQNode(const int &nodeID, const int &user_size) {
            this->nodeID_ = nodeID;
            this->user_size_ = user_size;
        }

    };


    /**
     * Preprocess: 先选择一个两个最远的点进行划分, 然后选择归类, 判断其半径大小
     * 讲半径更大的进行拆分, 直到总共的长度不超过N
     * **/
    class BallPrune {
    public:
        // a node contains many users
        int n_user_, vec_dim_, n_node_; // n_node is determined after the tree have built

        //IP bound
        std::unique_ptr<int[]> user2node_l_;  // n_user_, i-th position stores the nodeID it belongs to
        std::vector<double> center_l_; // n_node_ * vec_dim_, stores the center
        std::vector<double> radius_l_; // n_node_, stores the radius of each node

        //used for retrieval
        std::vector<bool> eval_l_; // n_node_, stores if evaluate this node
        std::vector<std::pair<double, double>> node_ip_bound_l;  // n_node_, stores, the IP bound of a node

        inline BallPrune() = default;

        //make bound from offset_dim to vec_dim
        void Preprocess(const VectorMatrix &user, const int node_threshold) {
            this->n_user_ = user.n_vector_;
            this->vec_dim_ = user.vec_dim_;
            this->n_node_ = 0;

            if (node_threshold > n_user_) {
                spdlog::error("the number of threshold larger than size of user, program exit");
                exit(-1);
            }

            user2node_l_ = std::make_unique<int[]>(n_user_);
            int *user_idx_vecs = user2node_l_.get();
            std::iota(user_idx_vecs, user_idx_vecs + n_user_, 0);

            BuildBall(user, node_threshold);

            spdlog::info("node_threshold {}, n_node {}", node_threshold, n_node_);

            std::vector<bool> check_l(n_node_);
            check_l.assign(n_user_, false);
            for (int userID = 0; userID < n_user_; userID++) {
                int nodeID = user2node_l_[userID];
                check_l[nodeID] = true;
            }
            for (int nodeID = 0; nodeID < n_node_; nodeID++) {
                assert(check_l[nodeID] == true);
            }

            for (int nodeID = 0; nodeID < n_node_; nodeID++) {
                assert(0 <= radius_l_[nodeID]);
            }
        }

        void BuildBall(const VectorMatrix &user, const int &node_threshold) {

            /**先将所有的点分配到node, 计算半径, 赋值圆心**/
            //to store the candidate for each node
            auto PQNodeMaxHeap = [](const PQNode &node, const PQNode &other) {
                return node.user_size_ < other.user_size_;
            };

            std::priority_queue<PQNode, std::vector<PQNode>, decltype(PQNodeMaxHeap)> node_max_heap{PQNodeMaxHeap};
            node_max_heap.push(PQNode(0, n_user_));
            PQNode max_node = node_max_heap.top();

            std::vector<std::vector<int>> ball_candidate_l; //store the userID for a vector
            std::vector<int> user_idx(n_user_);
            std::iota(user_idx.begin(), user_idx.end(), 0);
            ball_candidate_l.push_back(user_idx);

            n_node_ = 1;

            while (max_node.user_size_ > node_threshold) {
                assert(node_max_heap.size() == ball_candidate_l.size());
                int nodeID = max_node.nodeID_;
                std::vector<int> candidate_l = ball_candidate_l[nodeID];

                std::pair<int, int> cuserID_pair = SelectCenterUser(user, max_node.user_size_, candidate_l);
                std::vector<int> candidate_l1;
                std::vector<int> candidate_l2;
                PartitionCandidate(user, candidate_l, cuserID_pair, candidate_l1, candidate_l2);
                int nodeID1 = nodeID;
                int nodeID2 = (int) ball_candidate_l.size();
                ball_candidate_l[nodeID] = candidate_l1;
                ball_candidate_l.push_back(candidate_l2);

                node_max_heap.pop();
                node_max_heap.emplace(nodeID1, candidate_l1.size());
                node_max_heap.emplace(nodeID2, candidate_l2.size());
                max_node = node_max_heap.top();
                n_node_++;
            }

            assert(ball_candidate_l.size() == node_max_heap.size());
            //assign the result

            assert(n_node_ == ball_candidate_l.size());
            assert(n_node_ < n_user_);

            radius_l_.resize(n_node_);
            center_l_.resize(n_node_ * vec_dim_);
            eval_l_.resize(n_node_);
            node_ip_bound_l.resize(n_node_);
            for (int nodeID = 0; nodeID < n_node_; nodeID++) {
                std::vector<int> candidate_l = ball_candidate_l[nodeID];
                int user_size = (int) candidate_l.size();

                for (int start_idx = 0; start_idx < user_size; start_idx++) {
                    int userID = candidate_l[start_idx];
                    user2node_l_[userID] = nodeID;
                }

                double *center_vecs = center_l_.data() + nodeID * vec_dim_;
                std::memset(center_vecs, 0, vec_dim_ * sizeof(double));
                for (int candID = 0; candID < user_size; candID++) {
                    int userID = candidate_l[candID];
                    const double *user_vecs = user.getVector(userID);
                    for (int dim = 0; dim < vec_dim_; dim++) {
                        center_vecs[dim] += user_vecs[dim];
                    }
                }
                for (int dim = 0; dim < vec_dim_; dim++) {
                    center_vecs[dim] /= user_size;
                }

                double max_dist = -1;
                for (int candID = 0; candID < user_size; candID++) {
                    int userID = candidate_l[candID];
                    const double *user_vecs = user.getVector(userID);
                    double dist = EuclideanDistance(user_vecs, center_vecs, vec_dim_);
                    max_dist = dist > max_dist ? dist : max_dist;
                }
                assert(max_dist >= 0);
                radius_l_[nodeID] = max_dist;

            }
        }

        void PartitionCandidate(const VectorMatrix &user, const std::vector<int> &candidate_l,
                                const std::pair<int, int> &cuserID_pair,
                                std::vector<int> &candidate_l1, std::vector<int> &candidate_l2) const {
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
                double dist1 = EuclideanDistance(user_vecs, cuser1_vecs, vec_dim_);
                double dist2 = EuclideanDistance(user_vecs, cuser2_vecs, vec_dim_);
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

        [[nodiscard]] std::pair<int, int>
        SelectCenterUser(const VectorMatrix &user, const int user_size, const std::vector<int> &candidate_l) const {
            /**select the promising pair, return the userID of center point**/
            assert(user_size == candidate_l.size());
            /**对一整个数组进行partition, 分成两份, 将user数量最多的那一份放到最远的地方**/

            //choose two userID as the temp center
            std::random_device rd;  //Will be used to obtain a seed for the random number engine
            std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
            std::uniform_int_distribution<> distrib(0, user_size - 1);
            int cuserID = candidate_l[distrib(gen)]; // center userID
            double max_dist = -1;
            int cuserID2 = -1;
            for (int candID = 0; candID < user_size; candID++) {
                int userID = candidate_l[candID];
                double dist = EuclideanDistance(user.getVector(userID), user.getVector(cuserID), vec_dim_);
                if (max_dist < dist) {
                    max_dist = dist;
                    cuserID2 = userID;
                }
            }

            max_dist = -1;
            cuserID = -1;
            for (int candID = 0; candID < user_size; candID++) {
                int userID = candidate_l[candID];
                double dist = EuclideanDistance(user.getVector(userID), user.getVector(cuserID2), vec_dim_);
                if (max_dist < dist) {
                    max_dist = dist;
                    cuserID = userID;
                }
            }
            assert(cuserID2 != -1 && cuserID != -1);
            return std::make_pair(cuserID, cuserID2);
        }

        void IPBound(const double *query_vecs, const VectorMatrix &user, const std::vector<bool> &prune_l,
                     std::vector<std::pair<double, double>> &ip_bound_l) {
            //算出query和各个中心的IP, 通过IP bound转换成一个batch的rank bound
            assert(ip_bound_l.size() == n_user_);
            eval_l_.assign(n_node_, false);

            double qnorm = InnerProduct(query_vecs, query_vecs, vec_dim_);
            qnorm = std::sqrt(qnorm);

            for (int userID = 0; userID < n_user_; userID++) {
                int nodeID = user2node_l_[userID];
                if (eval_l_[nodeID]) {
                    ip_bound_l[userID] = node_ip_bound_l[nodeID];
                    continue;
                }

                const double *center_vecs = center_l_.data() + nodeID * vec_dim_;
                double radius = radius_l_[nodeID];
                double baseIP = InnerProduct(query_vecs, center_vecs, vec_dim_);
                double radiusIP = qnorm * radius;

                std::pair<double, double> IPbound_pair(baseIP - radiusIP, baseIP + radiusIP);
                node_ip_bound_l[nodeID] = IPbound_pair;
                ip_bound_l[userID] = IPbound_pair;
            }
        }

    };

}
#endif //REVERSE_KRANKS_BALLPRUNE_HPP
