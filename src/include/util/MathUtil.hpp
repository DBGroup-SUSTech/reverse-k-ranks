//
// Created by BianZheng on 2022/10/29.
//

#ifndef REVERSE_K_RANKS_MATHUTIL_HPP
#define REVERSE_K_RANKS_MATHUTIL_HPP

#include <limits>

namespace ReverseMIPS {
    float constexpr sqrtNewtonRaphson(float x, float curr, float prev) {
        return curr == prev
               ? curr
               : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
    }

    /*
    * Constexpr version of the square root
    * Return value:
    *	- For a finite and non-negative value of "x", returns an approximation for the square root of "x"
    *   - Otherwise, returns NaN
    */
    float constexpr sqrt(float x) {
        return x >= 0 && x < std::numeric_limits<float>::infinity()
               ? sqrtNewtonRaphson(x, x, 0)
               : std::numeric_limits<float>::quiet_NaN();
    }
}
#endif //REVERSE_K_RANKS_MATHUTIL_HPP
