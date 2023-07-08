//
// Created by BianZheng on 2022/3/20.
//
#include <vector>
#include <cassert>
#include <algorithm>
#include <random>
#include <fstream>
#include <set>
#include <spdlog/spdlog.h>
#include "util/TimeMemory.hpp"

using namespace ReverseMIPS;

void
PruneSingle(const std::vector<int> &lb_l, const std::vector<int> &ub_l,
            const int &n_user, const int &topk,
            std::vector<bool> &prune_l, std::vector<int> &topk_lb_heap) {
    assert(lb_l.size() == n_user);
    assert(ub_l.size() == n_user);
    assert(prune_l.size() == n_user);

    for (int userID = 0; userID < topk; userID++) {
        topk_lb_heap[userID] = lb_l[userID];
    }
    std::make_heap(topk_lb_heap.begin(), topk_lb_heap.end(), std::less());
    int global_lb = topk_lb_heap.front();

    int topk_1 = topk - 1;
    for (int userID = topk; userID < n_user; userID++) {
        int tmp_lb = lb_l[userID];
        if (global_lb > tmp_lb) {
            std::pop_heap(topk_lb_heap.begin(), topk_lb_heap.end(), std::less());
            topk_lb_heap[topk_1] = tmp_lb;
            std::push_heap(topk_lb_heap.begin(), topk_lb_heap.end(), std::less());
            global_lb = topk_lb_heap.front();
        }
    }

    for (int userID = 0; userID < n_user; userID++) {
        int tmp_ub = ub_l[userID];
        if (global_lb < tmp_ub) {
            prune_l[userID] = true;
        }
    }

}

void
PruneSingleUpgrade(const std::vector<int> &lb_l, const std::vector<int> &ub_l, const int &topk,
                   std::set<int> &user_idx_s,
                   std::vector<int> &topk_lb_heap) {
    int n_candidate = (int) user_idx_s.size();
    assert(topk <= n_candidate);
    assert(lb_l.size() == ub_l.size());

    auto iter = user_idx_s.begin();
    for (int i = 0; i < topk; i++) {
        int userID = *iter;
        topk_lb_heap[i] = lb_l[userID];
        iter++;
    }

    std::make_heap(topk_lb_heap.begin(), topk_lb_heap.end(), std::less());
    int global_lb = topk_lb_heap.front();

    int topk_1 = topk - 1;
    for (int i = topk; i < n_candidate; i++) {
        int userID = *iter;
        int tmp_lb = lb_l[userID];
        if (global_lb > tmp_lb) {
            std::pop_heap(topk_lb_heap.begin(), topk_lb_heap.end(), std::less());
            topk_lb_heap[topk_1] = tmp_lb;
            std::push_heap(topk_lb_heap.begin(), topk_lb_heap.end(), std::less());
            global_lb = topk_lb_heap.front();
        }
        iter++;
    }
    assert(iter == user_idx_s.end());

    for (iter = user_idx_s.begin(); iter != user_idx_s.end();) {
        int userID = *iter;
        int tmp_ub = ub_l[userID];
        if (global_lb < tmp_ub) {
            iter = user_idx_s.erase(iter);
        } else {
            ++iter;
        }
    }
}

void PruneIntegrate(const std::vector<int> &lb_l, const std::vector<int> &ub_l, const int &n_user,
                    const int &topk, std::vector<int> &user_idx_l, std::vector<bool> &prune_l, int &n_candidate) {
    assert(user_idx_l.size() == prune_l.size());
    assert(user_idx_l.size() >= n_candidate);
    assert(user_idx_l.size() > topk);
    assert(user_idx_l.size() == lb_l.size());
    assert(lb_l.size() == ub_l.size());
    assert(user_idx_l.size() == n_user);

    const auto ArgLBMaxHeap = [lb_l](const int &a, const int &b) noexcept -> bool {
        return lb_l[a] < lb_l[b];
    };

    const auto ArgUBMaxHeap = [ub_l](const int &a, const int &b) noexcept -> bool {
        return ub_l[a] < ub_l[b];
    };

    auto topk_iter = user_idx_l.begin() + topk;
    // max heap for lower rank
    std::make_heap(user_idx_l.begin(), topk_iter, ArgLBMaxHeap);
    int top_userID = user_idx_l.front();
    int top_lb = lb_l[top_userID];
    int top_ub = ub_l[top_userID];

    int n_remain = 0;
    const int topk_1 = topk - 1;
    const int candidate_size = n_candidate;
    for (int candID = topk; candID < candidate_size; candID++) {
        if (prune_l[candID]) {
            continue;
        }

        int this_lb = lb_l[candID];
        int this_ub = ub_l[candID];
        assert(this_lb >= this_ub);

        if (top_lb > this_lb) {
            //current lower bound smaller than topk lower bound
            //pop current tok heap
            int pre_top_ub = top_ub;

            std::pop_heap(user_idx_l.begin(), topk_iter, ArgLBMaxHeap);
            std::swap(user_idx_l[topk_1], user_idx_l[candID]);
            std::push_heap(user_idx_l.begin(), topk_iter, ArgLBMaxHeap);
            top_userID = user_idx_l.front();
            top_lb = lb_l[top_userID];
            top_ub = ub_l[top_userID];
            //update the remain_heap
            if (pre_top_ub <= top_lb) {
                //popped element can be push into remain_heap
                std::swap(user_idx_l[candID], topk_iter[n_remain]);
                n_remain++;
                std::push_heap(topk_iter, topk_iter + n_remain, ArgUBMaxHeap);
            }
            //delete the upper bound in the remain_heap
            int remain_ub = ub_l[topk_iter[0]];
            while (n_remain > 0 and top_lb < remain_ub) {
                std::pop_heap(topk_iter, topk_iter + n_remain, ArgUBMaxHeap);
                n_remain--;
                if (n_remain > 0) {
                    remain_ub = ub_l[topk_iter[0]];
                }
            }

        } else if (top_lb >= this_ub) {
            std::swap(user_idx_l[candID], topk_iter[n_remain]);
            n_remain++;
            std::push_heap(topk_iter, topk_iter + n_remain, ArgUBMaxHeap);
        }
    }

    n_candidate = n_remain + topk;

}


void GenRandomPair(const int &n_user, const int &n_item,
                   std::vector<int> &lb_l, std::vector<int> &ub_l) {

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(1, n_item);

    lb_l.resize(n_user);
    ub_l.resize(n_user);
    for (int userID = 0; userID < n_user; userID++) {
        int lb = distrib(gen);
        int ub = distrib(gen);
        //lb should bigger than ub in rank
        if (lb < ub) {
            int tmp = ub;
            ub = lb;
            lb = tmp;
        }
        lb_l[userID] = lb;
        ub_l[userID] = ub;
        assert(lb >= ub);
    }

}

void AttributionWrite(const std::vector<std::tuple<double, double, double>> &result_l, const std::vector<int> &topk_l) {

    char resPath[256];
    std::sprintf(resPath, "../../result/attribution/Prune/PruneImplementation.txt");
    std::ofstream file(resPath);
    if (!file) {
        std::printf("error in write result\n");
    }

    assert(result_l.size() == topk_l.size());
    int size = (int) topk_l.size();

    for (int i = 0; i < size; i++) {
        file << "top-" << topk_l[i] << ", single time " << std::to_string(std::get<0>(result_l[i]))
             << "s, single upgrade time " << std::to_string(std::get<1>(result_l[i]))
             << "s, integrate time " << std::to_string(std::get<2>(result_l[i])) << "s" << std::endl;
    }

    file.close();
}

using namespace std;

int main(int argc, char **argv) {
    const int n_user = 100000;
    const int n_item = 10000;

    spdlog::info("PruneImplementation n_user {}, n_item {}", n_user, n_item);

    std::vector<std::pair<int, int>> random_pair_l;

    std::vector<int> lb_l;
    std::vector<int> ub_l;
    GenRandomPair(n_user, n_item, lb_l, ub_l);

    std::vector<tuple<double, double, double>> result_l;
    const vector<int> topk_l{10, 30, 50, 70, 90};

    for (const int &topk: topk_l) {

        std::vector<int> user_idx_l(n_user);

        std::vector<bool> single_prune_l(n_user);
        single_prune_l.assign(n_user, false);
        std::vector<int> topk_lb_heap(topk);// store the lower bound

        TimeRecord record;
        record.reset();
        PruneSingle(lb_l, ub_l, n_user, topk, single_prune_l, topk_lb_heap);
        double single_time = record.get_elapsed_time_second();

        std::set<int> user_idx_s;
        for (int userID = 0; userID < n_user; userID++) {
            user_idx_s.emplace(userID);
        }
        record.reset();
        PruneSingleUpgrade(lb_l, ub_l, topk, user_idx_s, topk_lb_heap);
        double single_upgrade_time = record.get_elapsed_time_second();

        int n_candidate = n_user;
        std::iota(user_idx_l.begin(), user_idx_l.end(), 0);
        std::vector<bool> integrate_prune_l(n_user);
        integrate_prune_l.assign(n_user, false);
        record.reset();
        PruneIntegrate(lb_l, ub_l, n_user, topk, user_idx_l, integrate_prune_l, n_candidate);
        double integrate_time = record.get_elapsed_time_second();

        result_l.emplace_back(single_time, single_upgrade_time, integrate_time);

        /**test**/
        int single_candidate = 0;
        for (int userID = 0; userID < n_user; userID++) {
            if (!single_prune_l[userID]) {
                single_candidate++;
            }
        }
        assert(single_candidate == n_candidate && single_candidate == user_idx_s.size());
        spdlog::info("top-{}, prune ratio {}", topk, 1.0 * (n_user - n_candidate) / n_user);

        for (int userID = 0; userID < n_user; userID++) {
            if (!single_prune_l[userID]) {
                bool has_found = false;
                for (int candID = 0; candID < n_candidate; candID++) {
                    if (userID == user_idx_l[candID]) {
                        has_found = true;
                    }
                }
                if (!has_found) {
                    printf("not found\n");
                }
                assert(has_found);

                has_found = false;
                for (const int &user_idx_: user_idx_s) {
                    if (userID == user_idx_) {
                        has_found = true;
                    }
                }
                if (!has_found) {
                    printf("not found\n");
                }
                assert(has_found);
            }
        }
    }
    int n_topk = (int) topk_l.size();
    for (int i = 0; i < n_topk; i++) {
        spdlog::info("top-{} single time {}s, single upgrade time {}s, integrate time {}s", topk_l[i],
                     std::get<0>(result_l[i]), std::get<1>(result_l[i]), std::get<2>(result_l[i]));
    }

    AttributionWrite(result_l, topk_l);

}