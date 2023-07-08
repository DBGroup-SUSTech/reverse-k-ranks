//
// Created by BianZheng on 2022/11/30.
//

#include "alg/SpaceInnerProduct.hpp"
#include "ComplexityCompareUtil.hpp"
#include "util/TimeMemory.hpp"

#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>

using namespace std;
using namespace ReverseMIPS;

int main(int argc, char **argv) {

    const size_t n_user = 1000000;
    const size_t n_query = 1000;
    const int n_try_dim = 15;

    std::vector<std::pair<size_t, double>> time_use_l(n_try_dim);

    for (int try_dim = 1; try_dim <= n_try_dim; try_dim++) {
        const size_t tau = 2 << try_dim;

        unique_ptr<double[]> data_ip_l = GenRandom(n_user, tau);
        SortArray(data_ip_l.get(), n_user, tau);
        unique_ptr<double[]> query_ip_l = GenRandom(n_query, 1);

        spdlog::info("BinarySearch, n_user {}, n_query {}, tau {}", n_user, n_query, tau);

        int64_t sum = 0;

        TimeRecord record;
        record.reset();
        for (size_t queryID = 0; queryID < n_query; queryID++) {
            const double queryIP = query_ip_l[queryID];
            for (size_t userID = 0; userID < n_user; userID++) {
                const double *user_vecs = data_ip_l.get() + userID * tau;
                const double *ptr = std::lower_bound(user_vecs, user_vecs + tau, queryIP, std::less());
                sum += ptr - user_vecs;
            }
        }
        double binary_search_time = record.get_elapsed_time_second();
        spdlog::info("compute tau {}, time {}s, sum {}", tau, binary_search_time, sum);
        time_use_l[try_dim - 1] = std::make_pair(try_dim, binary_search_time);

    }

    WritePerformance(time_use_l, "BinarySearch", n_user, n_query);
    return 0;
}
