#include "dyy_RTK_method.hpp"
#include <stack>
#include <queue>
#include <spdlog/spdlog.h>

namespace dyy {

    RTK::ARankResult
    RTK::inRank(const RStarTree &tree, const Point &w, const int &minRank, const Point &q, size_t &ip_cost) {
        std::queue<Node_P> queue;
        queue.push(tree.root);
        int rnk = 0;
        float q_score = dot(q, w);
        ip_cost++;
        ARankResult AR;
        AR.isBetter = false;

        /*BFS*/
        while (!queue.empty()) {
            Node_P e = queue.front();
            queue.pop();
            if (e->level) { //non leaf
                Node_P_V &children = *e->children;
                for (size_t ic = 0; ic < children.size(); ic++) {
                    Node_P childptr = children.at(ic);
                    if (dotMbrUp(w, childptr->mbrn) > q_score) {
                        if (dotMbrLow(w, childptr->mbrn) > q_score) {
                            rnk += childptr->aggregate;
                            if (rnk > minRank)
                                return AR;
                        } else {
                            queue.push(childptr);
                        }
                    }
                }
            } else { //leaf
                Entry_P_V &entries = *e->entries;
                for (size_t ie = 0; ie < entries.size(); ie++) {
                    Entry_P entryPtr = entries.at(ie);
                    //mbre is a point
                    float score_p = dotMbrLow(w, entryPtr->mbre);
                    ip_cost++;
                    if (score_p > q_score) {
                        rnk++;
                        if (rnk > minRank) {
                            return AR;
                        }
                    }
                }
            }
        }// while BFS search

        if (rnk <= minRank) {
            AR.isBetter = true;
            AR.rank = rnk;
        }

        return AR;
    }


    RTK::ARankResult
    RTK::inRankPW(const RStarTree &tree, const Mbr &Ew, const int &minRank, const Point &q) {
        std::queue<Node_P> queue;
        queue.push(tree.root);
        int rnk = 0;
        float score_qLow = dotMbrLow(q, Ew);
        float score_qUp = dotMbrUp(q, Ew);

        ARankResult AR;
        AR.flag = -1;

        /*BFS*/
        while (!queue.empty()) {
            Node_P e = queue.front();
            queue.pop();
            if (e->level) { //non leaf
                Node_P_V &children = *e->children;
                for (size_t ic = 0; ic < children.size(); ic++) {
                    Node_P childptr = children.at(ic);
                    if (score_qLow < dotMMUp(childptr->mbrn, Ew)) {
                        if (score_qUp < dotMMLow(childptr->mbrn, Ew)) {
                            rnk += childptr->aggregate;
                            if (rnk > minRank)
                                return AR;
                        } else {
                            queue.push(childptr);
                        }
                    }
                }
            } else { //leaf
                Entry_P_V &entries = *e->entries;
                for (size_t ie = 0; ie < entries.size(); ie++) {
                    Entry_P entryPtr = entries.at(ie);
                    float score_p_low = dotMMLow(entryPtr->mbre, Ew);
                    if (score_p_low > score_qUp) {
                        if (score_p_low > score_qLow) {
                            rnk++;
                            if (rnk > minRank)
                                return AR;
                        }
                    }
                }
            }
        }// BFS

        /*not sure*/
        if (rnk <= minRank)
            AR.flag = 1;
        else
            AR.flag = 0;
        return AR;
    }


    RTK::BUFFER
    RTK::rkrmethod(const RStarTree &treeP, const RStarTree &treeW, const Point &q, const int &k, size_t &ip_cost,
                   const double &pass_time, const double &stop_time, size_t &n_proc_user) {
        std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();
        n_proc_user = 0;
        ip_cost = 0;
        BUFFER buffer;
        int threshold = std::numeric_limits<int>::max();
        std::stack<std::pair<Node_P, bool>> queue;

        queue.emplace(treeW.root, false);

        /*DFS W*/
        while (!queue.empty()) {
            std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = time_end - time_begin;
            if (diff.count() + pass_time > stop_time) {
                spdlog::info("total retrieval time larger than stop time inside Rtree method, retrieval exit");
                return buffer;
            }
            Node_P e = queue.top().first;
            bool in = queue.top().second;
            queue.pop();
            if (buffer.size() == k) {
                threshold = buffer.top().first;
            }
            if (e->level) { //non leaf
                if (in) {
                    Node_P_V &children = *e->children;
                    for (size_t ic = 0; ic < children.size(); ic++) {
                        Node_P childptr = children.at(ic);
                        queue.emplace(childptr, in);
                    }

                } else {
                    Node_P_V &children = *e->children;
                    for (size_t ic = 0; ic < children.size(); ic++) {
                        Node_P childptr = children.at(ic);
                        if (threshold == std::numeric_limits<int>::max()) {
                            queue.emplace(childptr, false);
                        } else {
                            ARankResult ar = inRankPW(treeP, childptr->mbrn, threshold, q);
                            if (ar.flag == 0) {
                                queue.emplace(childptr, false);
                            } else if (ar.flag == 1) {
                                queue.emplace(childptr, true);
                            } else {
                                n_proc_user += childptr->aggregate;
                            }
                        }
                    }
                }
            } else { //leaf
                Entry_P_V &entries = *e->entries;
                for (size_t ie = 0; ie < entries.size(); ie++) {
                    Entry_P entryPtr = entries.at(ie);
                    Point *w = static_cast<Point *>(entryPtr->data);
                    ARankResult ar = inRank(treeP, *w, threshold, q, ip_cost);
                    n_proc_user++;
                    if (ar.isBetter) {
                        if (buffer.size() == k) {
                            update_map(buffer, ar.rank, w->id);
                        } else {
                            buffer.push(std::make_pair(ar.rank, w->id));
                        }
                    }
                }
            }
        }// loop BFS W

        return buffer;
    }

    RTK::BUFFER
    RTK::rkrmethod(const RStarTree &treeP, const std::vector<Point> &tree_user_l, const Point &q, const int &topk,
                   size_t &ip_cost) {
        std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();
        ip_cost = 0;
        BUFFER buffer;

        const int n_user = (int) tree_user_l.size();

#pragma omp parallel for default(none) shared(topk, tree_user_l, treeP, q, ip_cost, buffer)
        for (int userID = 0; userID < topk; userID++) {
            Point w = tree_user_l[userID];
            const int max_int = std::numeric_limits<int>::max();
            ARankResult ar = inRank(treeP, w, max_int, q, ip_cost);
#pragma omp critical
            {
                if (ar.isBetter) {
                    buffer.emplace(ar.rank, w.id);
                }
            };

        }
#pragma omp parallel for default(none) shared(n_user, topk, buffer, tree_user_l, treeP, q, ip_cost)
        for (int userID = topk; userID < n_user; userID++) {
            const int threshold = buffer.top().first;
            Point w = tree_user_l[userID];
            ARankResult ar = inRank(treeP, w, threshold, q, ip_cost);
#pragma omp critical
            {
                if (ar.isBetter) {
                    update_map(buffer, ar.rank, w.id);
                }
            };
        }

        return buffer;
    }

}
