#ifndef _DYY_RTK_METHOD_HPP_
#define _DYY_RTK_METHOD_HPP_

#include "dyy_data.hpp"
#include <map>
#include <queue>
#include <vector>
#include <iterator>

namespace dyy {


    class RTK {
    public:

        /*dot procut*/
        static float dot(const Point &a, const Point &b);

        static float dotMbrLow(const Point &point, const Mbr &mbr);

        static float dotMbrUp(const Point &point, const Mbr &mbr);

        static float dotMMLow(const Mbr &a, const Mbr &b);

        static float dotMMUp(const Mbr &a, const Mbr &b);

        /*Reverse k-Rank method*/
        class ARankResult {
        public:
            bool isBetter;
            int rank;
            /*
              1:All in
              flag =    0:Need check children
              -1:All out
            */
            int flag;
        };

        static ARankResult
        inRank(const RStarTree &tree, const Point &w, const int &minRank, const Point &q, size_t &ip_cost);

        static ARankResult inRankPW(const RStarTree &tree, const Mbr &Ew, const int &minRank, const Point &q);

        struct cmp {
            bool operator()(const std::pair<int, int> &a, const std::pair<int, int> &b) const {
                if (a.first != b.first) {
                    return a.first < b.first;
                } else {
                    return a.second < b.second;
                }
            }
        };

        typedef std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, cmp> BUFFER;

        static void update_map(BUFFER &map, int key, int value);

        static BUFFER
        rkrmethod(const RStarTree &treeP, const RStarTree &treeW, const Point &q, const int &k, size_t &ip_cost,
                  const double &pass_time, const double &stop_time, size_t &n_proc_user);

        static BUFFER
        rkrmethod(const RStarTree &treeP, const std::vector<Point> &tree_user_l, const Point &q, const int &k,
                  size_t &ip_cost);

    };


    inline float RTK::dot(const Point &a, const Point &b) {
        float score = 0;
        for (size_t dim = 0; dim < Point::DIM; dim++)
            score += a.coords[dim] * b.coords[dim];
        return score;
    }

    inline float RTK::dotMbrLow(const Point &point, const Mbr &mbr) {
        float score = 0;
        for (size_t dim = 0; dim < Point::DIM; dim++) {
            const float min_val = std::min(point.coords[dim] * mbr.coord[dim][0],
                                           point.coords[dim] * mbr.coord[dim][1]);
            score += min_val;
        }

        return score;
    }

    inline float RTK::dotMbrUp(const Point &point, const Mbr &mbr) {
        float score = 0;
        for (size_t dim = 0; dim < Point::DIM; dim++) {
            const float max_val = std::max(point.coords[dim] * mbr.coord[dim][0],
                                           point.coords[dim] * mbr.coord[dim][1]);
            score += max_val;
        }
        return score;
    }

    inline float RTK::dotMMLow(const Mbr &a, const Mbr &b) {
        float score = 0;
        for (size_t dim = 0; dim < Point::DIM; dim++) {
            const float min_val = std::min(a.coord[dim][0] * b.coord[dim][0],
                                           a.coord[dim][0] * b.coord[dim][1]);
            const float min_val2 = std::min(a.coord[dim][1] * b.coord[dim][0],
                                            a.coord[dim][1] * b.coord[dim][1]);
            score += std::min(min_val, min_val2);
        }
        return score;
    }

    inline float RTK::dotMMUp(const Mbr &a, const Mbr &b) {
        float score = 0;
        for (size_t dim = 0; dim < Point::DIM; dim++) {
            const float max_val = std::max(a.coord[dim][0] * b.coord[dim][0],
                                           a.coord[dim][0] * b.coord[dim][1]);
            const float max_val2 = std::max(a.coord[dim][1] * b.coord[dim][0],
                                            a.coord[dim][1] * b.coord[dim][1]);
            score += std::max(max_val, max_val2);
        }
        return score;
    }

    inline void RTK::update_map(BUFFER &map, int key, int value) {
        const int threshold = map.top().first;
        const int threshold_userID = map.top().second;
        if ((key < threshold) || (key == threshold && value < threshold_userID)) {
            map.pop();
            map.emplace(key, value);
        }
    }


}

#endif /*_DYY_RTK_METHOD_HPP_*/
