#ifndef _DYY_DATA_HPP_
#define _DYY_DATA_HPP_

#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <cstring>
#include "const.h"
#include "dyy_RstarTree.hpp"

/*
  Data model used in the research of YuyangDong
 */

namespace dyy {
/*********************************************************************
 * Point
 ********************************************************************/
    class Point {
    public:
        static size_t DIM;
        Coord_V coords;
        int id;

        Point() {};

        Point(const Coord_V &v, const int &ID = -1) {
            coords = v;
            this->id = ID;
        }

        Point(const Coord *vecs, const int &vec_dim, const int &ID = -1) {
            this->id = ID;
            assert(vec_dim == DIM);
            coords.resize(vec_dim);
            std::memcpy(coords.data(), vecs, sizeof(Coord) * vec_dim);
        }

        friend std::istream &operator>>(std::istream &in, Point &p);

        bool operator<(const Point &p) const;

    public:
        void print();
    };


/*********************************************************************
 * Data
 ********************************************************************/

    typedef std::vector<Point> Point_V;
    typedef std::vector<LeafNodeEntry> Entry_V;

    class Data {
    public:

        Point_V Products;
        Point_V Weights;

        RStarTree RtreeP;
        RStarTree RtreeW;

        Data() {};

        ~Data() {};

        Entry_V entriesP;
        Entry_V entriesW;

        Data(Data &&x)
                : Products(std::move(x.Products)), Weights(std::move(x.Weights)),
                  entriesP(std::move(x.entriesP)), entriesW(std::move(x.entriesW)),
                  RtreeP(std::move(x.RtreeP)),
                  RtreeW(std::move(x.RtreeW)) {
        }

        static void loadPoint(std::string fileName, Point_V &v);

        static void buildTree(Point_V &points, Entry_V &entries, RStarTree *tree);

    };


}


#endif /*_DYY_DATA_HPP_*/
