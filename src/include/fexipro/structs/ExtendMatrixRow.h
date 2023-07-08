#ifndef EXTENDMATRIXROW_H
#define EXTENDMATRIXROW_H

#include "../util/Base.h"

class ExtendMatrixRow {

public:
    int gRowID; // original ID
    float norm;
    float subNorm;
    float *rawData;

    inline ExtendMatrixRow() {
        this->gRowID = -1;
        this->norm = 0;
        this->rawData = NULL;
    }

    inline ExtendMatrixRow(int gRowID, int colNum) : gRowID(gRowID) {
        this->rawData = new float[colNum];
    }

    inline ExtendMatrixRow(int gRowID, float norm, int colNum) : gRowID(gRowID), norm(norm) {
        this->rawData = new float[colNum];
    }

    inline ~ExtendMatrixRow() {
        if (rawData) {
            delete[] rawData;
        }
    }

};

#endif //EXTENDMATRIXROW_H
