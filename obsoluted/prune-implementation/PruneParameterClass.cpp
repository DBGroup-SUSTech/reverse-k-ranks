//
// Created by BianZheng on 2022/3/20.
//

/**
 * 对比剪枝时传递参数采用的方式, 是使用一个大的类还是使用很多个小的类
 * 画出数据量---运行时间的图像
 * 剪枝实现采用设置prune_l的方法
 * 数组随机生成即可
 * **/

#include <cassert>
#include <random>
#include <algorithm>
#include <spdlog/spdlog.h>
#include <fstream>
#include "util/TimeMemory.hpp"

using namespace ReverseMIPS;

class RankBoundElement {
public:
    int userID_, lower_rank_, upper_rank_;
    bool prune_;

    //by default, value of upper rank is smaller than lower rank
    RankBoundElement(const int &userID, const int &lb, const int &ub) {
        this->userID_ = userID;
        this->lower_rank_ = lb;
        this->upper_rank_ = ub;
        this->prune_ = false;
    }

    std::pair<int, int> rank_pair() {
        return std::make_pair(lower_rank_, upper_rank_);
    }

    RankBoundElement() {
        userID_ = -1;
        upper_rank_ = -1;
        lower_rank_ = -1;
        prune_ = false;
    }

    ~RankBoundElement() = default;

    std::string ToString() {
        char arr[256];
        sprintf(arr, "userID %d, lower_rank %d, upper_rank %d", userID_, lower_rank_, upper_rank_);
        std::string str(arr);
        return str;
    }

    RankBoundElement &operator=(const RankBoundElement &other) noexcept = default;

    static inline bool UpperBoundMaxHeap(const RankBoundElement &userRankBoundElement,
                                         RankBoundElement &other) {
        return userRankBoundElement.upper_rank_ < other.upper_rank_;
    }
};

void GenRandomPair(const int &n_user, const int &n_item,
                   std::vector<int> &lb_l, std::vector<int> &ub_l,
                   std::vector<RankBoundElement> &random_ele_l) {

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

    random_ele_l.resize(n_user);
    for (int userID = 0; userID < n_user; userID++) {
        random_ele_l[userID] = RankBoundElement(userID, lb_l[userID], ub_l[userID]);
    }
}

double
PruneSingle(const std::vector<int> &lb_l, const std::vector<int> &ub_l,
            const int &n_user, const int &n_item, const int &topk,
            std::vector<bool> &prune_l) {
    assert(lb_l.size() == n_user);
    assert(ub_l.size() == n_user);
    assert(prune_l.size() == n_user);
    prune_l.assign(n_user, false);
    std::vector<int> topk_lb_heap(topk);// store the lower bound

    TimeRecord record;
    record.reset();

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

    return record.get_elapsed_time_second();

}

double
PruneIntegrate(const int &n_user, const int &n_item, const int &topk,
               std::vector<RankBoundElement> &ele_l) {
    assert(ele_l.size() == n_user);
    for (int userID = 0; userID < n_user; userID++) {
        ele_l[userID].prune_ = false;
    }

    std::vector<int> topk_lb_heap(topk);// store the lower bound

    TimeRecord record;
    record.reset();

    for (int userID = 0; userID < topk; userID++) {
        topk_lb_heap[userID] = ele_l[userID].lower_rank_;
    }
    std::make_heap(topk_lb_heap.begin(), topk_lb_heap.end(), std::less());
    int global_lb = topk_lb_heap.front();

    int topk_1 = topk - 1;
    for (int userID = topk; userID < n_user; userID++) {
        int tmp_lb = ele_l[userID].lower_rank_;
        if (global_lb > tmp_lb) {
            std::pop_heap(topk_lb_heap.begin(), topk_lb_heap.end(), std::less());
            topk_lb_heap[topk_1] = tmp_lb;
            std::push_heap(topk_lb_heap.begin(), topk_lb_heap.end(), std::less());
            global_lb = topk_lb_heap.front();
        }
    }

    for (int userID = 0; userID < n_user; userID++) {
        int tmp_ub = ele_l[userID].upper_rank_;
        if (global_lb < tmp_ub) {
            ele_l[userID].prune_ = true;
        }
    }

    return record.get_elapsed_time_second();

}

void AttributionWrite(const std::vector<std::pair<double, double>> &result_l, const std::vector<int> &topk_l) {

    char resPath[256];
    std::sprintf(resPath, "../../result/attribution/Prune/PruneParameterClass.txt");
    std::ofstream file(resPath);
    if (!file) {
        std::printf("error in write result\n");
    }

    assert(result_l.size() == topk_l.size());
    int size = (int) topk_l.size();

    for (int i = 0; i < size; i++) {
        file << "top-" << topk_l[i] << ", single class time " << std::to_string(result_l[i].first)
             << "s, integrate time " << std::to_string(result_l[i].second) << "s" << std::endl;
    }

    file.close();
}

using namespace std;

int main(int argc, char **argv) {
    const int n_user = 100000000;
    const int n_item = 100000;

    spdlog::info("PruneParameterClass n_user {}, n_item {}", n_user, n_item);

    std::vector<std::pair<int, int>> random_pair_l;

    std::vector<int> lb_l;
    std::vector<int> ub_l;
    std::vector<RankBoundElement> ele_l;
    GenRandomPair(n_user, n_item, lb_l, ub_l, ele_l);

    std::vector<std::pair<double, double>> result_l;
    const vector<int> topk_l{10, 30, 50, 70, 90};
    std::vector<bool> prune_l(n_user);
    for (const int &topk: topk_l) {
        double single_time = PruneSingle(lb_l, ub_l, n_user, n_item, topk, prune_l);
        double integrate_time = PruneIntegrate(n_user, n_item, topk, ele_l);

        result_l.emplace_back(single_time, integrate_time);

        for (int userID = 0; userID < n_user; userID++) {
            assert(prune_l[userID] == ele_l[userID].prune_);
        }
    }
    int n_topk = (int) topk_l.size();
    for (int i = 0; i < n_topk; i++) {
        spdlog::info("top-{} single time {}s, integrate time {}s", topk_l[i], result_l[i].first, result_l[i].second);
    }

    AttributionWrite(result_l, topk_l);


}