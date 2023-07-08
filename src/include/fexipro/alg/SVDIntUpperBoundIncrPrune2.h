#ifndef SIRPRUNE_H
#define SIRPRUNE_H

#include "fexipro/structs/Matrix.h"
#include "fexipro/structs/SVDIntMatrixRow.h"
#include "fexipro/structs/ExtendMatrix.h"
#include "fexipro/structs/FastHeap.h"
#include "fexipro/structs/SIRMatrixRow.h"
#include "fexipro/structs/VectorElement.h"

#include "fexipro/util/Base.h"
#include "fexipro/util/Conf.h"
#include "fexipro/util/SVDUtil.h"
#include "fexipro/util/Monitor.h"
#include "fexipro/util/Calculator.h"


// FEIPR-SIR
class SIRPrune {

private:

    ExtendMatrix<SIRMatrixRow> *preprocessedP;
    Matrix *u;
    int checkDim;
    int checkDim2;
    int numOfDim;
    int scalingValue;
    vector<float> addend;
    float minValue;

    void transferQ(const Matrix &q, const int &k, float &qNorm, float &newSVDQNorm, float &subQNorm,
                   float &subTransformedQNorm,
                   float &sumOfCoordinate, float &leftPartialSumOfCoordinate, const float *qPtr,
                   float *newQ,
                   int *qIntPtr, int &qSumOfCoordinate1, int &qSumOfCoordinate2, float &qRatio1,
                   float &qRatio2, size_t &ip_count);

    void refine(const Matrix &q, const int &k, VectorElement *heap, const float qNorm, float *newQ,
                const float subQNorm, const int *newQIntPtr, int qSumOfCoordinate1, int qSumOfCoordinate2,
                float ratio1, float ratio2, const float newSVDQNorm, const float subTransformQNorm,
                const float sumOfQCoordinate, const float leftPartialQSumOfCoordinate, size_t &ip_count);

public:
    SIRPrune(const int scalingValue, const float SIGMA, const Matrix *p);

    SIRPrune(const SIRPrune &) = delete;

    SIRPrune &operator=(const SIRPrune &) = delete;

    SIRPrune(SIRPrune &&x)
            : preprocessedP(x.preprocessedP), u(x.u), checkDim(x.checkDim), checkDim2(x.checkDim2),
              numOfDim(x.numOfDim), scalingValue(x.scalingValue), addend(std::move(x.addend)), minValue(x.minValue) {
        x.preprocessedP = nullptr;
        x.u = nullptr;
    }

    SIRPrune &operator=(SIRPrune rhs) noexcept {
        using std::swap;
        swap(this->preprocessedP, rhs.preprocessedP);
        swap(this->u, rhs.u);
        swap(this->checkDim, rhs.checkDim);
        swap(this->checkDim2, rhs.checkDim2);
        swap(this->numOfDim, rhs.numOfDim);
        swap(this->scalingValue, rhs.scalingValue);
        swap(this->addend, rhs.addend);
        swap(this->minValue, rhs.minValue);
        return *this;
    }

    ~SIRPrune();

    void topK(const Matrix &q, const int &k, std::vector<VectorElement> &vector_element_l, size_t &ip_count);

};

inline void SIRPrune::transferQ(const Matrix &q, const int &k, float &qNorm, float &newSVDQNorm, float &subQNorm,
                                float &subTransformedQNorm,
                                float &sumOfCoordinate, float &leftPartialSumOfCoordinate, const float *qPtr,
                                float *newQ,
                                int *qIntPtr, int &qSumOfCoordinate1, int &qSumOfCoordinate2, float &qRatio1,
                                float &qRatio2, size_t &ip_count) {

    float minValue = FLT_MAX;
    float maxValue = -FLT_MAX;
    qNorm = 0;

    for (int colIndex = 0; colIndex < u->colNum; colIndex++) {
        qNorm += qPtr[colIndex] * qPtr[colIndex];
    }
    qNorm = sqrt(qNorm);

    newSVDQNorm = 0;
    for (int rowIndex = u->rowNum - 1; rowIndex >= checkDim; rowIndex--) {

        newQ[rowIndex] = 0;
        const float *uPtr = u->getRowPtr(rowIndex);

        for (int colIndex = 0; colIndex < u->colNum; colIndex++) {
            newQ[rowIndex] += uPtr[colIndex] * qPtr[colIndex];
        }
        ip_count++;

        if (newQ[rowIndex] < minValue) {
            minValue = newQ[rowIndex];
        }

        if (newQ[rowIndex] > maxValue) {
            maxValue = newQ[rowIndex];
        }

        newSVDQNorm += newQ[rowIndex] * newQ[rowIndex];

    }

    subQNorm = sqrt(newSVDQNorm);

    float absMin = fabs(minValue);
    float absMax = fabs(maxValue); // maxValue can be negative
    float denominator = absMin > absMax ? absMin : absMax;

    if (denominator == 0) {
        qRatio2 = 0;
    } else {
        qRatio2 = scalingValue / denominator;
    }

    qSumOfCoordinate2 = 0; //sumOfCoordinate

    for (int colIndex = q.colNum - 1; colIndex >= checkDim; colIndex--) {
        qIntPtr[colIndex] = floor(newQ[colIndex] * qRatio2);
        qSumOfCoordinate2 += fabs(qIntPtr[colIndex]);
    }

    minValue = FLT_MAX;
    maxValue = -FLT_MAX;

    for (int rowIndex = checkDim - 1; rowIndex >= 0; rowIndex--) {

        newQ[rowIndex] = 0;
        const float *uPtr = u->getRowPtr(rowIndex);

        for (int colIndex = 0; colIndex < u->colNum; colIndex++) {
            newQ[rowIndex] += uPtr[colIndex] * qPtr[colIndex];
        }
        ip_count++;

        newSVDQNorm += newQ[rowIndex] * newQ[rowIndex];

        if (newQ[rowIndex] < minValue) {
            minValue = newQ[rowIndex];
        }

        if (newQ[rowIndex] > maxValue) {
            maxValue = newQ[rowIndex];
        }

    }

    newSVDQNorm = sqrt(newSVDQNorm);

    subTransformedQNorm = 0;
    sumOfCoordinate = 0;

    float tmp;
    for (int colIndex = numOfDim; colIndex >= checkDim; colIndex--) {
        sumOfCoordinate += addend[colIndex] * newQ[colIndex] / newSVDQNorm;
        tmp = (newQ[colIndex] / newSVDQNorm + addend[colIndex]) * 2;
        subTransformedQNorm += tmp * tmp;
    }
    subTransformedQNorm = sqrt(subTransformedQNorm);
    leftPartialSumOfCoordinate = 0;

    absMin = fabs(minValue);
    absMax = fabs(maxValue); // maxValue can be negative
    denominator = absMin > absMax ? absMin : absMax;

    if (denominator == 0) {
        qRatio1 = 0;
    } else {
        qRatio1 = scalingValue / denominator;
    }

    qSumOfCoordinate1 = 0; //sumOfCoordinate

    for (int colIndex = checkDim - 1; colIndex >= 0; colIndex--) {
        qIntPtr[colIndex] = floor(newQ[colIndex] * qRatio1);
        qSumOfCoordinate1 += fabs(qIntPtr[colIndex]);

        tmp = addend[colIndex] * newQ[colIndex] / newSVDQNorm;
        sumOfCoordinate += tmp;
        leftPartialSumOfCoordinate += tmp;
    }

    qRatio1 = 1 / qRatio1;
    qRatio2 = 1 / qRatio2;

}

inline void SIRPrune::refine(const Matrix &q, const int &k, VectorElement *heap, const float qNorm, float *newQ,
                             const float subQNorm, const int *newQIntPtr, int qSumOfCoordinate1, int qSumOfCoordinate2,
                             float ratio1, float ratio2, const float newSVDQNorm, const float subTransformQNorm,
                             const float sumOfQCoordinate, const float leftPartialQSumOfCoordinate,
                             size_t &ip_count) {

    int heapCount = 0;
    for (int rowIndex = 0; rowIndex < k; rowIndex++) {
        const SIRMatrixRow *pRowPtr = preprocessedP->getRowPtr(rowIndex);
        heap_enqueue(Calculator::innerProduct(newQ, pRowPtr->rawData, q.colNum), pRowPtr->gRowID, heap, &heapCount);
        ip_count++;
    }

    float originalLowerBound = heap[0].data;

    int gRowIDForBoundItem = heap[0].id;
    const SIRMatrixRow *pRowPtr = preprocessedP->getRowPtr(gRowIDForBoundItem);

    float lowerBoundInNewSpace = (originalLowerBound / newSVDQNorm + sumOfQCoordinate + pRowPtr->sumOfCoordinate) * 2 +
                                  pRowPtr->partialSumOfCoordinate;

    for (int rowIndex = k; rowIndex < preprocessedP->rowNum; rowIndex++) {

        const SIRMatrixRow *pRowPtr = preprocessedP->getRowPtr(rowIndex);

        if (pRowPtr->norm * qNorm <= originalLowerBound) {
            break;
        } else {

            int bound = pRowPtr->sumOfCoordinate1 + qSumOfCoordinate1;
            const int *pIntPtr = pRowPtr->iRawData;

            for (int dim = 0; dim < checkDim; dim++) {
                bound += pIntPtr[dim] * newQIntPtr[dim];
            }

            float subNormBound = subQNorm * pRowPtr->subNorm;
            float leftInt = bound * ratio1;
            if (leftInt + subNormBound <= originalLowerBound) {
                continue;
            }

            bound = pRowPtr->sumOfCoordinate2 + qSumOfCoordinate2;
            for (int dim = checkDim; dim < q.colNum; dim++) {
                bound += pIntPtr[dim] * newQIntPtr[dim];
            }

            if (leftInt + bound * ratio2 <= originalLowerBound) {
                continue;
            }

            float innerProduct = 0;
            const float *pPtr = pRowPtr->rawData;
            for (int dim = 0; dim < checkDim; dim++) {
                innerProduct += pPtr[dim] * newQ[dim];
            }

            if (innerProduct + subNormBound <= originalLowerBound) {
                continue;
            }

            float sublLowerBoundInNewSpace =
                    (innerProduct / newSVDQNorm + leftPartialQSumOfCoordinate + pRowPtr->leftPartialSumOfCoordinate) *
                    2 + pRowPtr->partialSumOfCoordinate;

            if (sublLowerBoundInNewSpace + pRowPtr->subTransformedSubVNorm * subTransformQNorm <=
                lowerBoundInNewSpace) {
                continue;
            }

            for (int dim = checkDim; dim < q.colNum; dim++) {
                innerProduct += pPtr[dim] * newQ[dim];
            }

            if (innerProduct > originalLowerBound) {

                heap_dequeue(heap, &heapCount);
                heap_enqueue(innerProduct, pRowPtr->gRowID, heap, &heapCount);
                originalLowerBound = heap[0].data;

                int gRowIDForBoundItem = heap[0].id;
                const SIRMatrixRow *pRowPtr = preprocessedP->getRowPtr(gRowIDForBoundItem);
                lowerBoundInNewSpace =
                        (originalLowerBound / newSVDQNorm + sumOfQCoordinate + pRowPtr->sumOfCoordinate) * 2 +
                        pRowPtr->partialSumOfCoordinate;

            }
        }
    }
}

inline SIRPrune::SIRPrune(const int scalingValue, const float SIGMA, const Matrix *p) {

    fmat P_t(p->rawData, p->colNum, p->rowNum, false, true);
//    P_t = P_t.t();

    this->u = new Matrix();
    this->scalingValue = scalingValue;
    this->numOfDim = p->colNum;

    Matrix *v = new Matrix();
//    this->checkDim = SVD(P_t, p->colNum, p->rowNum, *u, *v, SIGMA);

    // add the two extended dimensions
    this->checkDim = SVD(P_t, p->colNum, p->rowNum, *u, *v, addend, SIGMA);
    this->checkDim2 = checkDim + 2;

    preprocessedP = new ExtendMatrix<SIRMatrixRow>();
    preprocessedP->initSIRMatrix(*p, *v, checkDim, checkDim2, scalingValue, addend, minValue);

    delete v;

}

inline SIRPrune::~SIRPrune() {
    if (u) {
        delete u;
    }

    if (preprocessedP) {
        delete preprocessedP;
    }
}

inline void
SIRPrune::topK(const Matrix &q, const int &k, std::vector<VectorElement> &vector_element_l, size_t &ip_count) {

    assert(vector_element_l.size() == q.rowNum);

    const float pRatio1 = preprocessedP->ratio1;
    const float pRatio2 = preprocessedP->ratio2;

//#pragma omp parallel for default(none) private(newQ, newQIntPtr, qSumOfCoordinate1, qSumOfCoordinate2, subQNorm, transformSubQNorm, newSVDQNorm, qNorm, sumOfQCoordinate, leftPartialQSumOfCoordinate, qRatio1, qRatio2, ratio1, ratio2) shared(q, heap_ele_l, k, ip_count, vector_element_l, pRatio1, pRatio2)
#pragma omp parallel for default(none) reduction(+:ip_count) shared(q, k, vector_element_l, pRatio1, pRatio2)
    for (int qID = 0; qID < q.rowNum; qID++) {

        std::vector<VectorElement> heap_ele_l(k);

        std::vector<float> newQ(q.colNum);
        std::vector<int> newQIntPtr(q.colNum);
        int qSumOfCoordinate1 = 0;
        int qSumOfCoordinate2 = 0;

        float subQNorm = 0;
        float transformSubQNorm = 0;
        float newSVDQNorm = 0;
        float qNorm = 0;
        float sumOfQCoordinate = 0;
        float leftPartialQSumOfCoordinate = 0;

        float qRatio1;
        float qRatio2;
        float ratio1;
        float ratio2;

        VectorElement *heap = heap_ele_l.data();

        const float *qPtr = q.getRowPtr(qID);

        transferQ(q, k, qNorm, newSVDQNorm, subQNorm, transformSubQNorm, sumOfQCoordinate, leftPartialQSumOfCoordinate,
                  qPtr,
                  newQ.data(), newQIntPtr.data(), qSumOfCoordinate1, qSumOfCoordinate2, qRatio1, qRatio2, ip_count);

        ratio1 = qRatio1 * pRatio1;
        ratio2 = qRatio2 * pRatio2;

        refine(q, k, heap, qNorm, newQ.data(), subQNorm, newQIntPtr.data(), qSumOfCoordinate1, qSumOfCoordinate2,
               ratio1, ratio2,
               newSVDQNorm, transformSubQNorm, sumOfQCoordinate, leftPartialQSumOfCoordinate, ip_count);

        int topk_min_id = 0;
        float topk_min_ip = heap_ele_l[0].data;
        for (int topk_idx = 1; topk_idx < k; topk_idx++) {
            if (topk_min_ip > heap_ele_l[topk_idx].data) {
                topk_min_ip = heap_ele_l[topk_idx].data;
                topk_min_id = topk_idx;
            }
        }
        vector_element_l[qID] = heap_ele_l[topk_min_id];
    }

//    Logger::Log("online time: " + to_string(tt.getElapsedTime()) + " secs");
//
//    if (Conf::outputResult) {
//        string resultFileName = Conf::resultPathPrefix + "-" + Conf::dataset + "-" + Conf::algName + "-" + to_string
//                (Conf::k) + "-" + to_string(Conf::scalingValue) + "-" +
//                                to_string(Conf::SIGMA) + ".txt";
//        FileUtil::outputResult(q->rowNum, k, results, resultFileName);
//    }

}

#endif //SIRPRUNE_H
