//
// Created by bianzheng on 2023/5/5.
//

#ifndef REVERSE_KRANKS_TREENODEINFO_HPP
#define REVERSE_KRANKS_TREENODEINFO_HPP
namespace ReverseMIPS {
    class TreeNodeInfo {
    public:
        int height, n_element;
        double radius;

        TreeNodeInfo(int height, int n_element, double radius) : height(height), n_element(n_element),
                                                                 radius(radius) {}

        TreeNodeInfo() : height(-1), n_element(-1),
                         radius(-1) {}

    };

}
#endif //REVERSE_KRANKS_TREENODEINFO_HPP
