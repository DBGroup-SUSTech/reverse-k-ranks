#ifndef SVDINTMATRIXROW_H
#define SVDINTMATRIXROW_H

#include "../util/Base.h"
#include "ExtendMatrixRow.h"

class SVDIntMatrixRow : public ExtendMatrixRow {

public:
    int *iRawData;
    float subNorm;
    int sumOfCoordinate1; //sumOfCoordinate + dimension * 1
    int sumOfCoordinate2;

    inline SVDIntMatrixRow(){
        this->iRawData = NULL;
    }

    inline SVDIntMatrixRow(int gRowID, int colNum, float norm):ExtendMatrixRow(gRowID, norm, colNum) {
        this->iRawData = new int[colNum];
    }

    inline ~SVDIntMatrixRow(){

        if(iRawData) {
            delete[] iRawData;
        }
    }
};
#endif //SVDINTMATRIXROW_H