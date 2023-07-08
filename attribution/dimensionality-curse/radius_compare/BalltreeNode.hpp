//
// Created by bianzheng on 2023/4/26.
//

#ifndef REVERSE_KRANKS_BALLTREENODE_HPP
#define REVERSE_KRANKS_BALLTREENODE_HPP

#include <vector>
#include <map>

#include "struct/VectorMatrix.hpp"

#include "Balltree/structs/Matrix.h"
#include "Balltree/structs/BallTreeNode.h"
#include "Balltree/alg/tree/BallTreeSearch.h"


namespace ReverseMIPS::BalltreeNode {

    std::vector<std::pair<int, double>>
    inRank1(BallTreeNode *root) {
        std::queue<std::pair<BallTreeNode *, int>> queue;
        queue.emplace(root, 0);
        std::vector<std::pair<int, double>> node_radius_l;

        /*BFS*/
        while (!queue.empty()) {
            BallTreeNode *node = queue.front().first;
            const int level = queue.front().second;
            queue.pop();

            const double radius = node->getConstrain();
            const int node_size = node->getSize();
            node_radius_l.emplace_back(node_size, radius);

            if (!node->isLeafNode()) { //non leaf

                const int new_level = level + 1;
                BallTreeNode *left = node->getLeftNode();
                BallTreeNode *right = node->getRightNode();
                if (left) {
                    queue.emplace(left, new_level);
                }
                if (right) {
                    queue.emplace(right, new_level);
                }
            }
        }// while BFS search

        std::sort(node_radius_l.begin(), node_radius_l.end(),
                  [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
                      return a.first > b.first;
                  });

        return node_radius_l;
    }


    std::vector<std::pair<int, double>> HeightSizeCurve(const VectorMatrix &vm) {
        int n_vecs = vm.n_vector_;
        int vec_dim = vm.vec_dim_;

        const int nodeSize = 20;

        // Step1 (offline): build ball tree
        Matrix p;
        p.init(n_vecs, vec_dim);
        std::memcpy(p.rawData, vm.getRawData(), (size_t) n_vecs * vec_dim * sizeof(float));

        vector<int> pointIDs(p.rowNum, 0);
        vector<const float *> pointPtrs(p.rowNum, NULL);
        for (int id = 0; id < p.rowNum; id++) {
            pointIDs[id] = id;
            pointPtrs[id] = p.getRowPtr(id);
        }

        BallTreeNode root(pointIDs, pointPtrs, p.colNum);
        makeBallTree(&root, p, nodeSize);

        auto res = inRank1(&root);
        return res;
    }
}
#endif //REVERSE_KRANKS_BALLTREENODE_HPP
