//
// Created by bianzheng on 2023/4/24.
//

#ifndef REVERSE_KRANKS_RTREEHEIGHT_HPP
#define REVERSE_KRANKS_RTREEHEIGHT_HPP

#include <vector>
#include <map>

#include "TreeNodeInfo.hpp"

#include "struct/VectorMatrix.hpp"
#include "rtree/dyy_RTK_method.hpp"

namespace ReverseMIPS::RtreeHeight {

    std::vector<TreeNodeInfo> inRank1(const dyy::RStarTree &tree) {
        std::queue<std::pair<dyy::Node_P, int>> queue;
        queue.emplace(tree.root, 0);
        std::vector<TreeNodeInfo> node_radius_l;

        /*BFS*/
        while (!queue.empty()) {
            dyy::Node_P e = queue.front().first;
            const int height = queue.front().second;
            queue.pop();
//            spdlog::info("level: {}, aggregate: {}, radius: {}", e->level, e->aggregate, e->mbrn.getRadius());
            const double radius = e->mbrn.getRadius();
            const int node_size = e->aggregate;
            node_radius_l.emplace_back(height, node_size, radius);

            if (e->level) { //non leaf

                dyy::Node_P_V &children = *e->children;
                for (size_t ic = 0; ic < children.size(); ic++) {
                    dyy::Node_P childptr = children.at(ic);
                    queue.emplace(childptr, height + 1);
                }
            }
        }// while BFS search

        std::sort(node_radius_l.begin(), node_radius_l.end(),
                  [](const TreeNodeInfo &a, const TreeNodeInfo &b) {
                      if (a.height != b.height) {
                          return a.height > b.height;
                      }
                      return a.n_element > b.n_element;
                  });

        return node_radius_l;
    }


    std::vector<TreeNodeInfo> HeightSizeCurve(const VectorMatrix &vm) {
        int n_vecs = vm.n_vector_;
        int vec_dim = vm.vec_dim_;

        dyy::Point::DIM = vec_dim;
        dyy::Mbr::DIM = vec_dim;
        dyy::RTreeNode::DIM = vec_dim;

        dyy::Data data;

        std::vector<dyy::Point> data_item_data(n_vecs);
        for (int vecsID = 0; vecsID < n_vecs; vecsID++) {
            dyy::Point point(vm.getVector(vecsID), vec_dim, vecsID);
            data_item_data[vecsID] = point;
        }
        data.Products = data_item_data;

        dyy::Data::buildTree(data.Products, data.entriesP, &data.RtreeP);

        std::vector<TreeNodeInfo> res = inRank1(data.RtreeP);

        return res;
    }
}
#endif //REVERSE_KRANKS_RTREEHEIGHT_HPP
