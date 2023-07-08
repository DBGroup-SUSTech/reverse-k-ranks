#include <chrono>
#include "dyy_data.hpp"
#include "omp.h"

namespace dyy {

/*********************************************************************
 * Point
 ********************************************************************/

    size_t Point::DIM = 2;

    std::istream &operator>>(std::istream &in, Point &p) {
        p.coords.resize(Point::DIM);
        for (size_t d = 0; d < Point::DIM; d++)
            in >> p.coords[d];
        return in;
    }

    bool Point::operator<(const dyy::Point &p) const {
        for (size_t dim = 0; dim < DIM; dim++)
            if (coords[dim] != p.coords[dim])
                return coords[dim] < p.coords[dim];
        return true;
    }

    void Point::print() {
        std::cout << "<" << coords[0];
        for (size_t dim = 1; dim < DIM; dim++)
            std::cout << "," << coords[dim];
        std::cout << ">" << std::endl;
    }

/*********************************************************************
 * Data
 ********************************************************************/

    void Data::loadPoint(std::string data_file, Point_V &points) {
        points.clear();
        std::ifstream in(data_file.c_str());
        assert(in.is_open());
        int id = 0;
        while (true) {
            Point point;
            in >> point;
            if (in) {
                point.id = id++;
                points.push_back(point);
            } else
                break;
        }
        in.close();
    }

    void Data::buildTree(Point_V &points, Entry_V &entries, RStarTree *tree) {
        entries.clear();
        for (size_t ip = 0; ip < points.size(); ip++) {
            Data_P datap = &points.at(ip);
            Mbr mbr(points.at(ip).coords);
            LeafNodeEntry entry(mbr, datap);
            entries.push_back(entry);
        }

        std::cout << entries.size() << " entries created" << std::endl;

        std::chrono::steady_clock::time_point batch_time_begin = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point total_time_begin = std::chrono::steady_clock::now();

        for (size_t ie = 0; ie < entries.size(); ie++) {
            tree->insertData(&entries.at(ie));
            if (ie % 3000 == 0 && ie != 0) {
                std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
                std::chrono::duration<double> diff_batch = time_end - batch_time_begin;
                std::chrono::duration<double> diff_total = time_end - total_time_begin;
                const double batch_time = diff_batch.count();
                const double total_time = diff_total.count();
                batch_time_begin = std::chrono::steady_clock::now();
                std::cout << ie << " entries inserted, progress " << 1.0 * ie / entries.size() * 100
                          << "%, batch time " << batch_time << "s, passed time " << total_time << "s, predicted finish time "
                          << total_time / (ie + 1) * entries.size() << "s"
                          << std::endl;

            }
        }

        std::cout << tree->root->aggregate << " entries created" << std::endl;
    }


}
