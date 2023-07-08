#ifndef EXTENDMATRIX_H
#define EXTENDMATRIX_H

#include <bitset>

#include "fexipro/util/Calculator.h"
#include "fexipro/util/Base.h"
#include "fexipro/util/TransformUtil.h"

#include "fexipro/structs/VectorElement.h"

// sorted matrix by norm
template <class T>
class ExtendMatrix {

private:
    T *rows;

public:
    int rowNum;
    int colNum;
    // for int bound
    float minValue;
    float maxValue;
    float ratio;
    // for FEIPR-SI
    float ratio1;
    float ratio2;

    inline ExtendMatrix(){
        this->rows = NULL;
    }

    // for Sequential Scan
    inline void initExtendMatrix(const Matrix &rawData, const vector<VectorElement> &sortedNorm) {
        this->rowNum = rawData.rowNum;
        this->colNum = rawData.colNum;
        this->rows = new T[rowNum];

        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {
            const float *rawRowPtr = rawData.getRowPtr(sortedNorm[rowIndex].id);
            this->rows[rowIndex] = *(new T(sortedNorm[rowIndex].id, sortedNorm[rowIndex].data, colNum));
            for (int colIndex = 0; colIndex < colNum; colIndex++) {
                this->rows[rowIndex].rawData[colIndex] = rawRowPtr[colIndex];
            }
        }
    }

    // for transformed matrix
    inline void initExtendMatrix(const Matrix &rawData, const int checkDim, const vector<VectorElement> &sortedNorm, const vector<float> &unsortedSubNorm) {
        this->rowNum = rawData.rowNum;
        this->colNum = rawData.colNum;
        this->rows = new T[rowNum];

        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {
            const float *rawRowPtr = rawData.getRowPtr(sortedNorm[rowIndex].id);
            this->rows[rowIndex] = *(new T(sortedNorm[rowIndex].id, sortedNorm[rowIndex].data, colNum));
            this->rows[rowIndex].subNorm = unsortedSubNorm[sortedNorm[rowIndex].id];
            for (int colIndex = 0; colIndex < colNum; colIndex++) {
                this->rows[rowIndex].rawData[colIndex] = rawRowPtr[colIndex];
            }
        }
    }

    // for SVD + transformation
//    inline void initSVDTransformExtendMatrix(const Matrix &rawData, const Matrix &vRawData, const int checkDim, vector<float> &addend) {
//        this->rowNum = rawData.rowNum;
//        this->colNum = rawData.colNum + 2;
//        this->rows = new T[rowNum];
//
//        vector<VectorElement> pNorms(rawData.rowNum);
//        vector<VectorElement> vNorms(vRawData.rowNum);
//        float maxVNorm;
//        Calculator::calNorms(rawData, pNorms);
//        Calculator::calNorms(vRawData, vNorms, maxVNorm);
//        sort(pNorms.begin(), pNorms.end(), greater<VectorElement>());
//
//        Matrix transformedP;
//        vector<float> sortedSubNorm;
//
//        TransformUtil::makePPositiveSVDIncr(vRawData, transformedP, maxVNorm, checkDim, pNorms, vNorms, sortedSubNorm, addend);
//
//        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {
//            int gRowID = pNorms[rowIndex].id;
//            const float *rawRowPtr = rawData.getRowPtr(gRowID);
//            const float *transformedPtr = transformedP.getRowPtr(rowIndex);
//            this->rows[rowIndex] = *(new T(gRowID, pNorms[rowIndex].data, rawData.colNum, colNum));
//            this->rows[rowIndex].norm = pNorms[rowIndex].data;
//            this->rows[rowIndex].subVNorm = sortedSubNorm[rowIndex];
//            for (int colIndex = 0; colIndex < rawData.colNum; colIndex++) {
//                this->rows[rowIndex].rawData[colIndex] = rawRowPtr[colIndex];
//                this->rows[rowIndex].vRawData[colIndex] = transformedPtr[colIndex];
//            }
//
//            this->rows[rowIndex].vRawData[rawData.colNum] = transformedPtr[rawData.colNum];
//            this->rows[rowIndex].vRawData[rawData.colNum+1] = transformedPtr[rawData.colNum+1];
//
//        }
//    }

    // for SVD + transformation (incr on transformed vector)
    inline void initSVDTransformExtendMatrix(const Matrix &rawData, const Matrix &vRawData, const int checkDim, vector<float> &addend, float &minValue) {
        this->rowNum = rawData.rowNum;
        this->colNum = rawData.colNum + 2;
        this->rows = new T[rowNum];

        vector<VectorElement> pNorms(rawData.rowNum);
        vector<VectorElement> vNorms(vRawData.rowNum);
        float maxVNorm;

        Calculator::calNorms(rawData, pNorms);
        Calculator::calNorms(vRawData, vNorms, maxVNorm, minValue);
        minValue = std::max(minValue, 1.0f);
        sort(pNorms.begin(), pNorms.end(), greater<VectorElement>());

        Matrix transformedP;
        vector<float> sortedSubNorm;

        for(int i=0;i<addend.size();i++){
            addend[i] += minValue;
        }

        TransformUtil::makePPositiveSVDIncr(vRawData, transformedP, maxVNorm, checkDim, pNorms, vNorms, sortedSubNorm, addend);

        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {
            int gRowID = pNorms[rowIndex].id;
            const float *vRawPtr = vRawData.getRowPtr(gRowID);
            const float *transformedPtr = transformedP.getRowPtr(rowIndex);
            this->rows[rowIndex] = *(new T(gRowID, pNorms[rowIndex].data, this->colNum));

            float subVNorm = 0;
            this->rows[rowIndex].partialSumOfCoordinate = - transformedPtr[0];
            this->rows[rowIndex].rawData[0] = transformedPtr[0];
            this->rows[rowIndex].rawData[1] = transformedPtr[1];

            float sumOfCoordinate = 0;

            for (int colIndex = 2; colIndex < checkDim; colIndex++) {
                this->rows[rowIndex].rawData[colIndex] = transformedPtr[colIndex];
                sumOfCoordinate += addend[colIndex-2] * addend[colIndex-2] + addend[colIndex-2] * vRawPtr[colIndex-2];
            }

            for (int colIndex = checkDim; colIndex < this->colNum; colIndex++) {
                this->rows[rowIndex].rawData[colIndex] = transformedPtr[colIndex];
                subVNorm += transformedPtr[colIndex] * transformedPtr[colIndex];
                sumOfCoordinate += addend[colIndex-2] * addend[colIndex-2] + addend[colIndex-2] * vRawPtr[colIndex-2];
            }
            this->rows[rowIndex].subVNorm = sqrt(subVNorm);
            this->rows[rowIndex].sumOfCoordinate = sumOfCoordinate;
        }
    }

    // for SVD + transformation (incr on original svd vector, then transformed vector)
    inline void initSVDTransformExtendMatrix2(const Matrix &rawData, const Matrix &vRawData, const int checkDim2, vector<float> &addend, float &globalMinValue) {
        this->rowNum = rawData.rowNum;
        this->colNum = rawData.colNum;
        this->rows = new T[rowNum];

        vector<VectorElement> pNorms(rawData.rowNum);
        vector<VectorElement> vNorms(vRawData.rowNum);
        float maxVNorm;

        Calculator::calNorms(rawData, pNorms);
        Calculator::calNorms(vRawData, vNorms, maxVNorm, globalMinValue);
        globalMinValue = std::max(globalMinValue, 1.0f);
        sort(pNorms.begin(), pNorms.end(), greater<VectorElement>());

        Matrix transformedP;
        vector<float> sortedSubNorm;

        for (int i = 0; i < addend.size(); i++) {
            addend[i] += globalMinValue;
        }

        TransformUtil::makePPositiveSVDIncr(vRawData, transformedP, maxVNorm, checkDim2, pNorms, vNorms, sortedSubNorm, addend);

        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {
            int gRowID = pNorms[rowIndex].id;
            const float *vRawPtr = vRawData.getRowPtr(gRowID);
            const float *transformedPtr = transformedP.getRowPtr(rowIndex);
            this->rows[rowIndex] = *(new T(gRowID, pNorms[rowIndex].data, this->colNum));

            float subVNorm = 0;

            this->rows[rowIndex].partialSumOfCoordinate = - transformedPtr[0];

            float sumOfCoordinate = 0;

            for (int colIndex = 2; colIndex < checkDim2; colIndex++) {
                this->rows[rowIndex].rawData[colIndex - 2] = vRawPtr[colIndex - 2];
                sumOfCoordinate += addend[colIndex-2] * addend[colIndex-2] + addend[colIndex-2] * vRawPtr[colIndex-2];
            }

            this->rows[rowIndex].leftPartialSumOfCoordinate = sumOfCoordinate;

            for (int colIndex = checkDim2; colIndex < this->colNum + 2; colIndex++) {
                this->rows[rowIndex].rawData[colIndex - 2] = vRawPtr[colIndex - 2];
                subVNorm += vRawPtr[colIndex-2] * vRawPtr[colIndex-2];
                sumOfCoordinate += addend[colIndex-2] * addend[colIndex-2] + addend[colIndex-2] * vRawPtr[colIndex-2];
            }
            this->rows[rowIndex].subTransformedSubVNorm = sortedSubNorm[rowIndex];
            this->rows[rowIndex].subVNorm = sqrt(subVNorm);
            this->rows[rowIndex].sumOfCoordinate = sumOfCoordinate;
        }
    }

    // for FEIPR-S
    inline void initSVDExtendMatrix(const Matrix &rawData, const Matrix &vRawData, const vector<VectorElement> &sortedNorm,
                        const vector<float> &vNorms, const vector<float> &subVNorms) {
        this->rowNum = rawData.rowNum;
        this->colNum = rawData.colNum;
        this->rows = new T[rowNum];

        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {
            const float *rawRowPtr = rawData.getRowPtr(sortedNorm[rowIndex].id);
            const float *vRawRowPtr = vRawData.getRowPtr(sortedNorm[rowIndex].id);
            this->rows[rowIndex] = *(new T(sortedNorm[rowIndex].id, sortedNorm[rowIndex].data, colNum));
            this->rows[rowIndex].vNorm = vNorms[sortedNorm[rowIndex].id];
            this->rows[rowIndex].subVNorm = subVNorms[sortedNorm[rowIndex].id];
            for (int colIndex = 0; colIndex < colNum; colIndex++) {
//                this->rows[rowIndex].rawData[colIndex] = rawRowPtr[colIndex];
                this->rows[rowIndex].vRawData[colIndex] = vRawRowPtr[colIndex];
            }

        }
    }

    // for FEIPR-S with individual reorder
    inline void initSVDExtendMatrix(const Matrix &rawData, const Matrix &vRawData, const vector<VectorElement> &sortedNorm, const float *vSubNorms) {
        this->rowNum = rawData.rowNum;
        this->colNum = rawData.colNum;
        this->rows = new T[rowNum];

        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {
            const float *vRawRowPtr = vRawData.getRowPtr(sortedNorm[rowIndex].id);
            const float *vSubNorm = &vSubNorms[sortedNorm[rowIndex].id * colNum];
            this->rows[rowIndex] = *(new T(sortedNorm[rowIndex].id, sortedNorm[rowIndex].data, colNum));

            for (int colIndex = 0; colIndex < colNum; colIndex++) {
                this->rows[rowIndex].vRawData[colIndex] = vRawRowPtr[colIndex];
                this->rows[rowIndex].subVNorm[colIndex] = vSubNorm[colIndex];
            }

        }
    }

    // for FEIPR-I
    inline void initIntExtendMatrix(const Matrix &rawData, const vector<VectorElement> &sortedNorm, const int mapMaxValue) {
        this->rowNum = rawData.rowNum;
        this->colNum = rawData.colNum;
        this->rows = new T[rowNum];

        minValue = FLT_MAX;
        maxValue = -FLT_MAX;

        for (int rowID = 0; rowID < rawData.rowNum; rowID++) {

            const float *row = rawData.getRowPtr(rowID);

            for (int colIndex = 0; colIndex < rawData.colNum; colIndex++) {

                if(row[colIndex] < minValue) {
                    minValue = row[colIndex];
                }

                if(row[colIndex] > maxValue) {
                    maxValue = row[colIndex];
                }
            }
        }

        float absMin = fabs(minValue);
        float absMax = fabs(maxValue); // maxValue can be negative
        float denominator = absMin > absMax ? absMin : absMax;
        if(denominator==0){
            ratio = 0;
        } else {
            ratio = mapMaxValue / denominator;
        }

        int sumOfCoordinate; //sumOfCoordinate + dimension * 1
        int temp;
        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {
            const float *rawRowPtr = rawData.getRowPtr(sortedNorm[rowIndex].id);
            sumOfCoordinate = 0;
            this->rows[rowIndex] = *(new T(sortedNorm[rowIndex].id, sortedNorm[rowIndex].data, colNum));

            for (int colIndex = 0; colIndex < colNum; colIndex++) {
                this->rows[rowIndex].rawData[colIndex] = rawRowPtr[colIndex];
                temp = floor(rawRowPtr[colIndex] * ratio);
                this->rows[rowIndex].iRawData[colIndex] = temp;
                sumOfCoordinate += fabs(temp);
            }

            this->rows[rowIndex].sumOfCoordinate = sumOfCoordinate + colNum;
        }
    }

    // for FEIPR-I-2
    inline void initIntExtendMatrix2(const Matrix &rawData, const vector<VectorElement> &sortedNorm, const int mapMaxValue, const int checkDim) {
        this->rowNum = rawData.rowNum;
        this->colNum = rawData.colNum;
        this->rows = new T[rowNum];

        minValue = FLT_MAX;
        maxValue = -FLT_MAX;

        for (int rowID = 0; rowID < rawData.rowNum; rowID++) {

            const float *row = rawData.getRowPtr(rowID);

            for (int colIndex = 0; colIndex < rawData.colNum; colIndex++) {

                if(row[colIndex] < minValue) {
                    minValue = row[colIndex];
                }

                if(row[colIndex] > maxValue) {
                    maxValue = row[colIndex];
                }
            }
        }

        float absMin = fabs(minValue);
        float absMax = fabs(maxValue); // maxValue can be negative
        float denominator = absMin > absMax ? absMin : absMax;
        if(denominator==0){
            ratio = 0;
        } else {
            ratio = mapMaxValue / denominator;
        }

        int sumOfCoordinate; //sumOfCoordinate + dimension * 1
        int temp;
        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {
            const float *rawRowPtr = rawData.getRowPtr(sortedNorm[rowIndex].id);
            sumOfCoordinate = 0;
            this->rows[rowIndex] = *(new T(sortedNorm[rowIndex].id, colNum));

            float norm = 0;

            for (int colIndex = colNum-1; colIndex >= checkDim; colIndex--) {
                temp = rawRowPtr[colIndex] * ratio;
                this->rows[rowIndex].rawData[colIndex] = temp;
                norm += temp * temp;

                temp = floor(temp);
                this->rows[rowIndex].iRawData[colIndex] = temp;
                sumOfCoordinate += fabs(temp);
            }

            this->rows[rowIndex].sumOfCoordinateRight = sumOfCoordinate + colNum - checkDim + 1;
            this->rows[rowIndex].subNorm = sqrt(norm);
            sumOfCoordinate = 0;

            for (int colIndex = checkDim - 1; colIndex >=0; colIndex--) {
                temp = rawRowPtr[colIndex] * ratio;
                this->rows[rowIndex].rawData[colIndex] = temp;
                norm += temp * temp;

                temp = floor(temp);
                this->rows[rowIndex].iRawData[colIndex] = temp;
                sumOfCoordinate += fabs(temp);
            }
            this->rows[rowIndex].norm = sqrt(norm);
            this->rows[rowIndex].sumOfCoordinateLeft = sumOfCoordinate + checkDim - 1;
        }
    }

    // for FEIPR-SI
    inline void initSVDIntExtendMatrix(const Matrix &vRawData,
                                       const vector<VectorElement> &sortedNorm,
                                       const int checkDim, const int maxMapValue) {
        this->rowNum = vRawData.rowNum;
        this->colNum = vRawData.colNum;
        this->rows = new T[rowNum];

        int currentGID;
        float minValue = FLT_MAX;
        float maxValue = -FLT_MAX;

        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {

            currentGID = sortedNorm[rowIndex].id;
            const float *vRawRowPtr = vRawData.getRowPtr(currentGID);

            this->rows[rowIndex] = *(new T(sortedNorm[rowIndex].id, colNum, sortedNorm[rowIndex].data));

            for (int colIndex = 0; colIndex < checkDim; colIndex++) {

                this->rows[rowIndex].rawData[colIndex] = vRawRowPtr[colIndex];

                if (vRawRowPtr[colIndex] < minValue) {
                    minValue = vRawRowPtr[colIndex];
                }

                if (vRawRowPtr[colIndex] > maxValue) {
                    maxValue = vRawRowPtr[colIndex];
                }

            }
        }

        float absMin = fabs(minValue);
        float absMax = fabs(maxValue); // maxValue can be negative
        float denominator = absMin > absMax ? absMin : absMax;
        if(denominator==0){
            ratio1 = 0;
        } else {
            ratio1 = maxMapValue / denominator;
        }

        int sumOfCoordinate1; //sumOfCoordinate + dimension * 1
        int temp;

        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {

            currentGID = sortedNorm[rowIndex].id;
            const float *vRawRowPtr = vRawData.getRowPtr(currentGID);

            sumOfCoordinate1 = checkDim;

            for (int colIndex = 0; colIndex < checkDim; colIndex++) {

                temp = floor(vRawRowPtr[colIndex] * ratio1);

                this->rows[rowIndex].iRawData[colIndex] = temp;

                sumOfCoordinate1 += fabs(temp);

            }

            this->rows[rowIndex].sumOfCoordinate1 = sumOfCoordinate1;
        }

        minValue = FLT_MAX;
        maxValue = -FLT_MAX;
        float subNorm = 0;

        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {

            currentGID = sortedNorm[rowIndex].id;
            const float *vRawRowPtr = vRawData.getRowPtr(currentGID);
            subNorm = 0;
            for (int colIndex = checkDim; colIndex < vRawData.colNum; colIndex++) {

                this->rows[rowIndex].rawData[colIndex] = vRawRowPtr[colIndex];
                subNorm += vRawRowPtr[colIndex] * vRawRowPtr[colIndex];

                if (vRawRowPtr[colIndex] < minValue) {
                    minValue = vRawRowPtr[colIndex];
                }

                if (vRawRowPtr[colIndex] > maxValue) {
                    maxValue = vRawRowPtr[colIndex];
                }

            }
            this->rows[rowIndex].subNorm = sqrt(subNorm);

        }

        absMin = fabs(minValue);
        absMax = fabs(maxValue); // maxValue can be negative
        denominator = absMin > absMax ? absMin : absMax;
        if(denominator==0){
            ratio2 = 0;
        } else {
            ratio2 = maxMapValue / denominator;
        }

        int sumOfCoordinate2; //sumOfCoordinate + dimension * 1
        int dimOfSecondPart = vRawData.colNum - checkDim;
        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {

            currentGID = sortedNorm[rowIndex].id;
            const float *vRawRowPtr = vRawData.getRowPtr(currentGID);

            sumOfCoordinate2 = dimOfSecondPart;

            for (int colIndex = checkDim; colIndex < vRawData.colNum; colIndex++) {

                temp = floor(vRawRowPtr[colIndex] * ratio2);

                this->rows[rowIndex].iRawData[colIndex] = temp;

                sumOfCoordinate2 += fabs(temp);

            }

            this->rows[rowIndex].sumOfCoordinate2 = sumOfCoordinate2;
        }

        ratio1 = 1 / ratio1;
        ratio2 = 1 / ratio2;
    }


    // for FEIPR-SIR
    inline void initSIRMatrix(const Matrix &rawData, const Matrix &vRawData,
                                       const int checkDim, const int checkDim2, const int maxMapValue, vector<float> &addend, float &globalMinValue) {
        this->rowNum = vRawData.rowNum;
        this->colNum = vRawData.colNum;
        this->rows = new T[rowNum];

        vector<VectorElement> pNorms(rawData.rowNum);
        vector<VectorElement> vNorms(vRawData.rowNum);
        float maxVNorm;

        Calculator::calNorms(rawData, pNorms);
        Calculator::calNorms(vRawData, vNorms, maxVNorm, globalMinValue);
        globalMinValue= std::max(globalMinValue, 1.0f);

        sort(pNorms.begin(), pNorms.end(), greater<VectorElement>());

        Matrix transformedP;
        vector<float> sortedSubNorm;

        for (int i = 0; i < addend.size(); i++) {
            addend[i] += globalMinValue;
        }

        TransformUtil::makePPositiveSVDIncr(vRawData, transformedP, maxVNorm, checkDim2, pNorms, vNorms, sortedSubNorm, addend);

        int currentGID;
        float minValue = FLT_MAX;
        float maxValue = -FLT_MAX;

        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {

            currentGID = pNorms[rowIndex].id;
            const float *vRawRowPtr = vRawData.getRowPtr(currentGID);
            const float *transformedPtr = transformedP.getRowPtr(rowIndex);

            this->rows[rowIndex] = *(new T(pNorms[rowIndex].id, colNum, pNorms[rowIndex].data));

            float subVNorm = 0;

            this->rows[rowIndex].partialSumOfCoordinate = - transformedPtr[0];

            for (int colIndex = 0; colIndex < 2; colIndex++) {
                this->rows[rowIndex].rawData[colIndex] = vRawRowPtr[colIndex];

                if (vRawRowPtr[colIndex] < minValue) {
                    minValue = vRawRowPtr[colIndex];
                }

                if (vRawRowPtr[colIndex] > maxValue) {
                    maxValue = vRawRowPtr[colIndex];
                }
            }

            float sumOfCoordinate = 0;

            for (int colIndex = 2; colIndex < checkDim; colIndex++) {

                this->rows[rowIndex].rawData[colIndex] = vRawRowPtr[colIndex];

                if (vRawRowPtr[colIndex] < minValue) {
                    minValue = vRawRowPtr[colIndex];
                }

                if (vRawRowPtr[colIndex] > maxValue) {
                    maxValue = vRawRowPtr[colIndex];
                }

                sumOfCoordinate += addend[colIndex-2] * addend[colIndex-2] + addend[colIndex-2] * vRawRowPtr[colIndex-2];

            }

            for (int colIndex = checkDim; colIndex < checkDim2; colIndex++) {

                this->rows[rowIndex].rawData[colIndex] = vRawRowPtr[colIndex];

                if (vRawRowPtr[colIndex] < minValue) {
                    minValue = vRawRowPtr[colIndex];
                }

                if (vRawRowPtr[colIndex] > maxValue) {
                    maxValue = vRawRowPtr[colIndex];
                }

                sumOfCoordinate += addend[colIndex-2] * addend[colIndex-2] + addend[colIndex-2] * vRawRowPtr[colIndex-2];

            }

            this->rows[rowIndex].leftPartialSumOfCoordinate = sumOfCoordinate;

            for (int colIndex = checkDim2; colIndex < this->colNum; colIndex++) {
                this->rows[rowIndex].rawData[colIndex] = vRawRowPtr[colIndex];
                subVNorm += vRawRowPtr[colIndex-2] * vRawRowPtr[colIndex-2];
                sumOfCoordinate += addend[colIndex-2] * addend[colIndex-2] + addend[colIndex-2] * vRawRowPtr[colIndex-2];
            }

            for (int colIndex = this->colNum; colIndex < this->colNum + 2; colIndex++) {
                subVNorm += vRawRowPtr[colIndex-2] * vRawRowPtr[colIndex-2];
                sumOfCoordinate += addend[colIndex-2] * addend[colIndex-2] + addend[colIndex-2] * vRawRowPtr[colIndex-2];
            }

            this->rows[rowIndex].subTransformedSubVNorm = sortedSubNorm[rowIndex];
            this->rows[rowIndex].subVNorm = sqrt(subVNorm);
            this->rows[rowIndex].sumOfCoordinate = sumOfCoordinate;
        }

        float absMin = fabs(minValue);
        float absMax = fabs(maxValue); // maxValue can be negative
        float denominator = absMin > absMax ? absMin : absMax;
        if(denominator==0){
            ratio1 = 0;
        } else {
            ratio1 = maxMapValue / denominator;
        }

        int sumOfCoordinate1; //sumOfCoordinate + dimension * 1
        int temp;

        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {

            currentGID = pNorms[rowIndex].id;
            const float *vRawRowPtr = vRawData.getRowPtr(currentGID);

            sumOfCoordinate1 = checkDim;

            for (int colIndex = 0; colIndex < checkDim; colIndex++) {

                temp = floor(vRawRowPtr[colIndex] * ratio1);

                this->rows[rowIndex].iRawData[colIndex] = temp;

                sumOfCoordinate1 += fabs(temp);

            }

            this->rows[rowIndex].sumOfCoordinate1 = sumOfCoordinate1;
        }

        minValue = FLT_MAX;
        maxValue = -FLT_MAX;
        float subNorm = 0;

        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {

            currentGID = pNorms[rowIndex].id;
            const float *vRawRowPtr = vRawData.getRowPtr(currentGID);
            subNorm = 0;
            for (int colIndex = checkDim; colIndex < vRawData.colNum; colIndex++) {

                this->rows[rowIndex].rawData[colIndex] = vRawRowPtr[colIndex];
                subNorm += vRawRowPtr[colIndex] * vRawRowPtr[colIndex];

                if (vRawRowPtr[colIndex] < minValue) {
                    minValue = vRawRowPtr[colIndex];
                }

                if (vRawRowPtr[colIndex] > maxValue) {
                    maxValue = vRawRowPtr[colIndex];
                }

            }
            this->rows[rowIndex].subNorm = sqrt(subNorm);

        }

        absMin = fabs(minValue);
        absMax = fabs(maxValue); // maxValue can be negative
        denominator = absMin > absMax ? absMin : absMax;
        if(denominator==0){
            ratio2 = 0;
        } else {
            ratio2 = maxMapValue / denominator;
        }

        int sumOfCoordinate2; //sumOfCoordinate + dimension * 1
        int dimOfSecondPart = vRawData.colNum - checkDim;
        for (int rowIndex = 0; rowIndex < rowNum; rowIndex++) {

            currentGID = pNorms[rowIndex].id;
            const float *vRawRowPtr = vRawData.getRowPtr(currentGID);

            sumOfCoordinate2 = dimOfSecondPart;

            for (int colIndex = checkDim; colIndex < vRawData.colNum; colIndex++) {

                temp = floor(vRawRowPtr[colIndex] * ratio2);

                this->rows[rowIndex].iRawData[colIndex] = temp;

                sumOfCoordinate2 += fabs(temp);

            }

            this->rows[rowIndex].sumOfCoordinate2 = sumOfCoordinate2;
        }

        ratio1 = 1 / ratio1;
        ratio2 = 1 / ratio2;
    }

    inline ~ExtendMatrix(){
        if(rows) {
            delete[] rows;
        }
    }

    inline T *getRowPtr(const int rowIndex) const {
        return &rows[rowIndex];
    }


};

#endif //EXTENDMATRIX_H
