#ifndef CALCULATOR_H
#define CALCULATOR_H

#include "Base.h"
#include "../structs/Matrix.h"
#include "../structs/VectorElement.h"

namespace Calculator{

    inline float innerProduct(const float *qRow, const float *pRow, const int dim) {

        float value = 0;

        for (int colIndex = 0; colIndex < dim; colIndex++) {
            value += qRow[colIndex] * pRow[colIndex];
        }
        return value;
    }

    inline float l2Distance(const float *qRow, const float *pRow, const int dim) {

        float value = 0;

        for (int colIndex = 0; colIndex < dim; colIndex++) {
            value += (qRow[colIndex] - pRow[colIndex]) * (qRow[colIndex] - pRow[colIndex]);
        }

        return sqrt(value);
    }

    inline void calSingleNorm(const float *ptr, const int dim, float &norm) {

        norm = 0;
        for (int i = 0; i < dim; i++) {
            norm += ptr[i] * ptr[i];
        }
        norm = sqrt(norm);
    }

    inline void calNorms(const Matrix &m, vector<VectorElement> &norms, float &maxNorm) {
        norms.resize(m.rowNum);
        maxNorm = -1;
        for (int rowID = 0; rowID < m.rowNum; rowID++) {
            float norm = 0;
            const float *row = m.getRowPtr(rowID);
            for (int colIndex = 0; colIndex < m.colNum; colIndex++) {
                norm += row[colIndex] * row[colIndex];
            }
            norm = sqrt(norm);
            norms[rowID] = VectorElement(rowID, norm);
            if (norm > maxNorm) {
                maxNorm = norm;
            }
        }

    }

    inline void calNorms(const Matrix &m, vector<VectorElement> &norms, float &maxNorm, float &minValue) {
        norms.resize(m.rowNum);
        minValue = FLT_MAX;
        maxNorm = -1;
        for (int rowID = 0; rowID < m.rowNum; rowID++) {
            float norm = 0;
            const float *row = m.getRowPtr(rowID);
            for (int colIndex = 0; colIndex < m.colNum; colIndex++) {
                norm += row[colIndex] * row[colIndex];
                if(row[colIndex] < minValue) {
                    minValue = row[colIndex];
                }
            }
            norm = sqrt(norm);
            norms[rowID] = VectorElement(rowID, norm);
            if (norm > maxNorm) {
                maxNorm = norm;
            }
        }


        if(minValue < 0){
            minValue = abs(minValue);
        } else {
            minValue = 0;
        }

    }


    inline void calNorms(const Matrix &m, vector<VectorElement> &norms) {
        norms.resize(m.rowNum);

        for (int rowID = 0; rowID < m.rowNum; rowID++) {
            float norm = 0;
            const float *row = m.getRowPtr(rowID);
            for (int colIndex = 0; colIndex < m.colNum; colIndex++) {
                norm += row[colIndex] * row[colIndex];
            }
            norm = sqrt(norm);
            norms[rowID] = VectorElement(rowID, norm);
        }

    }

}
#endif //CALCULATOR_H
