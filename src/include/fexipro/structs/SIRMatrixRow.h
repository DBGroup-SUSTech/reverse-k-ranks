#ifndef SIRMATRIXROW_H
#define SIRMATRIXROW_H

#include "../util/Base.h"
#include "ExtendMatrixRow.h"

class SIRMatrixRow : public ExtendMatrixRow {

public:
    int *iRawData;
    float subNorm;
    int sumOfCoordinate1; //sumOfCoordinate + dimension * 1
    int sumOfCoordinate2;
    float subVNorm;
    float subTransformedSubVNorm;
    float partialSumOfCoordinate;
    float leftPartialSumOfCoordinate;
    float sumOfCoordinate;

    inline SIRMatrixRow(){
        this->iRawData = NULL;
    }

    inline SIRMatrixRow(int gRowID, int colNum, float norm):ExtendMatrixRow(gRowID, norm, colNum) {
        this->iRawData = new int[colNum];
    }

    inline ~SIRMatrixRow(){

        if(iRawData) {
            delete[] iRawData;
        }
    }
};
#endif //SIRMATRIXROW_H