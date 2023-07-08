#pragma once

#include <algorithm>

namespace ReverseMIPS {
//    double InnerProduct(const double *pVect1, const double *pVect2, const int dim) {
//        double res = 0;
//        for (unsigned i = 0; i < dim; i++) {
//            res += pVect1[i] * pVect2[i];
//        }
//        return res;
//    }

    float InnerProduct(const float *pVect1, const float *pVect2, const int& dim) {
        float res = 0;
        for (unsigned i = 0; i < dim; i++) {
            res += pVect1[i] * pVect2[i];
        }
        return res;
    }

    double InnerProduct(const double *pVect1, const double *pVect2, const int& dim) {
        double res = 0;
        for (unsigned i = 0; i < dim; i++) {
            res += pVect1[i] * pVect2[i];
        }
        return res;
    }

    int InnerProduct(const int *vecs1, const int *vecs2, const int& dim) {
        int res = 0;
        for (unsigned i = 0; i < dim; i++) {
            res += vecs1[i] * vecs2[i];
        }
        return res;
    }

}