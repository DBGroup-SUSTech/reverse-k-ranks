//
// Created by bianzheng on 2023/4/26.
//

#ifndef REVERSE_KRANKS_RTREENODE_HPP
#define REVERSE_KRANKS_RTREENODE_HPP

#include <vector>

#include "struct/VectorMatrix.hpp"
#include "rtree/dyy_RTK_method.hpp"
#include <map>

namespace ReverseMIPS::RtreeNode {

    std::vector<std::pair<int, double>> inRank1(const dyy::RStarTree &tree) {
        std::queue<dyy::Node_P> queue;
        queue.push(tree.root);
        std::vector<std::pair<int, double>> node_radius_l;

        /*BFS*/
        while (!queue.empty()) {
            dyy::Node_P e = queue.front();
            queue.pop();
//            spdlog::info("level: {}, aggregate: {}, radius: {}", e->level, e->aggregate, e->mbrn.getRadius());
            const double radius = e->mbrn.getRadius();
            const int node_size = e->aggregate;
            node_radius_l.emplace_back(node_size, radius);

            if (e->level) { //non leaf

                dyy::Node_P_V &children = *e->children;
                for (size_t ic = 0; ic < children.size(); ic++) {
                    dyy::Node_P childptr = children.at(ic);
                    queue.push(childptr);
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

        std::vector<std::pair<int, double>> res = inRank1(data.RtreeP);

        return res;
    }
}
#endif //REVERSE_KRANKS_RTREENODE_HPP
