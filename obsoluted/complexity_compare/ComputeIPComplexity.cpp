//
// Created by BianZheng on 2022/11/30.
//

#include "alg/SpaceInnerProduct.hpp"
#include "util/TimeMemory.hpp"
#include "ComplexityCompareUtil.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <string>

using namespace std;
using namespace ReverseMIPS;

int main(int argc, char **argv) {

    const int n_try_dim = 15;

    const size_t n_user = 1000000;
    const size_t n_query = 1000;

    std::vector<std::pair<size_t, double>> time_use_l(n_try_dim);

    for (int try_dim = 1; try_dim <= n_try_dim; try_dim++) {
        const int vec_dim = try_dim;

        unique_ptr<double[]> user_vecs_l = GenRandom(n_user, vec_dim);
        unique_ptr<double[]> query_vecs_l = GenRandom(n_query, vec_dim);

        spdlog::info("ComputeIP, n_user {}, n_query {}, vec_dim {}", n_user, n_query, vec_dim);

        int64_t sum = 0;

        TimeRecord record;
        record.reset();
        for (size_t queryID = 0; queryID < n_query; queryID++) {
            const double *query_vecs = query_vecs_l.get() + queryID * vec_dim;
            for (size_t userID = 0; userID < n_user; userID++) {
                const double *user_vecs = user_vecs_l.get() + userID * vec_dim;
                sum += (int64_t) InnerProduct(query_vecs, user_vecs, vec_dim);
            }
        }
        double ip_compute_time = record.get_elapsed_time_second();
        spdlog::info("try_dim {}, time {}s, sum {}", vec_dim, ip_compute_time, sum);
        time_use_l[try_dim - 1] = std::make_pair(try_dim, ip_compute_time);
    }

    WritePerformance(time_use_l, "ComputeIP", n_user, n_query);

    return 0;
}
