// Copyright 2020 Roger Peralta Aranibar Advanced Data Estructures
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <map>
#include <fstream>
#include <cmath>
#include <chrono>

#include "rectangle.hpp"
#include "xtree.hpp"

#define DIM 14
#define POINTS 170653
#define PATH "spotify_dataset.csv"

typedef std::pair<int, std::string> data_type; // {year, name}

std::map<size_t, float> normalizer;

template<typename>
class Timer;

template<typename R, typename... T>
class Timer<R(T...)> {
public:
    typedef R (*function_type)(T...);

    function_type function;

    explicit Timer(function_type function, std::string process_name = "")
            : function_(function), process_name_(process_name) {}

    R operator()(T... args) {
        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
        start = std::chrono::high_resolution_clock::now();

        R result = function_(std::forward<T>(args)...);

        end = std::chrono::high_resolution_clock::now();
        int64_t duration =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                        .count();

        std::cout << std::setw(10) << process_name_ << std::setw(30)
                  << "Duration: " + std::to_string(duration) + " ns\n";
        return result;
    }

private:
    function_type function_;
    std::string process_name_;
};


int build_data_structure(Xtree<data_type, 10> &Cake) {
    std::ifstream points(PATH);
    float value;
    int year;
    std::string info;
    getline(points, info); // ignore 1st line

    for (size_t p = 0; p < POINTS; ++p) {
        Point pt(DIM);
        // Get point data
        for (size_t i = 0; i < DIM; ++i) {
            points >> pt[i];
        }
        for (std::pair<size_t, float> norm: normalizer) {
            pt[norm.first] /= norm.second;
        }
        // Get song data
        points >> year;
        getline(points, info);
        Cake.insert(pt, std::make_pair(year, info));
    }
    return 0;
}

std::vector<std::vector<float>> query_knn(Xtree<data_type, 10> &Cake, std::vector<float> query, int k) {
    Point query_point(DIM);
    for (size_t i = 0; i < DIM; ++i) {
        query_point[i] = query[i];
    }
    std::vector<std::pair<Point, data_type> > knnData = Cake.KNNquery(query_point, k);
    for (size_t i = 0; i < knnData.size(); ++i) {
        std::cout << "   #" << i + 1 << " -> " << knnData[i].second.second << std::endl;
    }
    std::vector<std::vector<float> > result;
    for (size_t i = 0; i < knnData.size(); ++i) {
        result.push_back(knnData[i].first.get_vector());
    }
    return result;
}

int main() {
    Xtree<data_type, 10>::vec_dim_ = 14;
    Xtree<data_type, 10> Cake;

    // Normalize values
    normalizer[3] = 350000.0; // duration_ms
    normalizer[7] = 11.0; // key
    normalizer[9] = 60.0; // loudness
    normalizer[11] = 100.0; // popularity
    normalizer[13] = 245.0; // tempo

    std::cout << "*---------------------------------------------*" << std::endl;
    std::cout << "*---------- X Tree by Joaquin Palma ----------*" << std::endl;
    std::cout << "*---------------------------------------------*" << std::endl;
    std::cout << "|> Inserting Points ... " << std::endl;

    Timer<int(Xtree<data_type, 10> &)> timed_built(build_data_structure, "Index");
    timed_built(Cake);

    std::cout << "\n*-------- K Nearest Neighbors Queries --------*\n" << std::endl;
    while (true) {
        Timer<std::vector<std::vector<float>>(Xtree<data_type, 10> &, std::vector<float>, int)> timed_query(
                query_knn, "Query KNN");
        std::vector<float> query(DIM);
        int k;
//        std::cout << "|> Enter k: "; //std::cin>>k;
        k = 10;
//        std::cout << "|> Enter point coordinates: ";
        query[0] = 0.831;
        query[1] = 0.372;
        query[2] = 0.794;
        query[3] = 174901;
        query[4] = 0.845;
        query[5] = 1;
        query[6] = 0;
        query[7] = 1;
        query[8] = 0.124;
        query[9] = -6.118;
        query[10] = 1;
        query[11] = 66;
        query[12] = 0.387;
        query[13] = 93.939;

        std::cout << "k: " << k << std::endl;
        std::cout << "point coordinates: ";
        for (size_t i = 0; i < DIM; ++i) {
            std::cout << query[i] << " ";
        }
        std::cout << std::endl;

        for (std::pair<size_t, float> norm: normalizer) {
            query[norm.first] /= norm.second;
        }

        std::vector<std::vector<float>> result = timed_query(Cake, query, k);

        for (int i = 0; i < k; ++i) {
            // get normalized dist
            float dist = 0;
            for (size_t j = 0; j < DIM; ++j) {
                dist += (result[i][j] - query[j]) * (result[i][j] - query[j]);
            }
            dist = sqrt(dist);
            std::cout << "   #" << i + 1 << " [";
            std::cout << std::fixed << std::setprecision(5) << dist;
            std::cout << "] -> { ";
            for (std::pair<size_t, float> norm: normalizer) {
                result[i][norm.first] *= norm.second;
            }
            for (int j = 0; j < DIM; ++j)
                std::cout << std::fixed << std::setprecision(4) << result[i][j] << " ";
            std::cout << "}\n";
        }
        std::cout << '\n';
        break;
    }


    std::cout << "\n*-------- K Nearest Neighbors Queries --------*\n" << std::endl;
    while (true) {
        Timer<std::vector<std::vector<float>>(Xtree<data_type, 10> &, std::vector<float>, int)> timed_query(
                query_knn, "Query KNN");
        std::vector<float> query(DIM);
        int k;
//        std::cout << "|> Enter k: "; //std::cin>>k;
        k = 10;
//        std::cout << "|> Enter point coordinates: ";
//        for (size_t i = 0; i < DIM; ++i) {
//            std::cout << query[i] << " ";
//        }
        std::cout << std::endl;
        query[0] = 0.235;
        query[1] = 0.00849;
        query[2] = 0.62;
        query[3] = 238560;
        query[4] = 0.61;
        query[5] = 0;
        query[6] = 0.162;
        query[7] = 11;
        query[8] = 0.205;
        query[9] = -8.329;
        query[10] = 0;
        query[11] = 71;
        query[12] = 0.0373;
        query[13] = 127.052;

        std::cout << "k: " << k << std::endl;
        std::cout << "point coordinates: ";
        for (size_t i = 0; i < DIM; ++i) {
            std::cout << query[i] << " ";
        }
        std::cout << std::endl;

        for (std::pair<size_t, float> norm: normalizer) {
            query[norm.first] /= norm.second;
        }

        std::vector<std::vector<float>> result = timed_query(Cake, query, k);

        for (int i = 0; i < k; ++i) {
            // get normalized dist
            float dist = 0;
            for (size_t j = 0; j < DIM; ++j) {
                dist += (result[i][j] - query[j]) * (result[i][j] - query[j]);
            }
            dist = sqrt(dist);
            std::cout << "   #" << i + 1 << " [";
            std::cout << std::fixed << std::setprecision(5) << dist;
            std::cout << "] -> { ";
            for (std::pair<size_t, float> norm: normalizer) {
                result[i][norm.first] *= norm.second;
            }
            for (int j = 0; j < DIM; ++j)
                std::cout << std::fixed << std::setprecision(4) << result[i][j] << " ";
            std::cout << "}\n";
        }
        std::cout << '\n';
        break;
    }
}
