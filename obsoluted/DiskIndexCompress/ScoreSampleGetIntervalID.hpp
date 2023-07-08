//
// Created by BianZheng on 2022/7/15.
//

#ifndef REVERSE_KRANKS_SCORESAMPLEGETINTERVALID_HPP
#define REVERSE_KRANKS_SCORESAMPLEGETINTERVALID_HPP

void GetItvID(const DistancePair *distance_ptr, const int &userID,
              std::vector<unsigned char> &itvID_l) const {
    if (n_interval_ > 256) {
        spdlog::error("the number of interval larger than 256, program exit");
        exit(-1);
    }
    assert(itvID_l.size() == n_data_item_);
    for (int candID = 0; candID < n_data_item_; candID++) {
        assert(user_ip_bound_l_[userID].first <= user_ip_bound_l_[userID].second);
        const double IP_lb = user_ip_bound_l_[userID].first;
        const double IP_ub = user_ip_bound_l_[userID].second;
        const double itv_dist = interval_dist_l_[userID];
        const int itemID = distance_ptr[candID].ID_;
        const double ip = distance_ptr[candID].dist_;
        assert(IP_lb <= ip && ip <= IP_ub);
        const unsigned char itvID = std::floor((IP_ub - ip) / itv_dist);
        itvID_l[itemID] = itvID;
        assert(0 <= itvID && itvID < n_interval_);
    }

}

#endif //REVERSE_KRANKS_SCORESAMPLEGETINTERVALID_HPP
