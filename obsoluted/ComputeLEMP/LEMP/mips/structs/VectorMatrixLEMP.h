//    Copyright 2015 Christina Teflioudi
// 
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
// 
//        http://www.apache.org/licenses/LICENSE-2.0
// 
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

/*
 * VectorMatrix.h
 *
 *  Created on: Oct 10, 2013
 *      Author: chteflio
 */

#ifndef VECTORMATRIX_H_
#define VECTORMATRIX_H_

#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
//#include <util/exception.h>
//#include <util/io.h>

#include <string>
#include <ostream>
#include <iomanip>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include "BasicStructs.h"
//#include "util/random/RandomWrapper.h"

#ifdef  WITH_SIMD
#include <pmmintrin.h> //SSE3
#endif


using boost::unordered_map;

namespace mips {

    inline void skipLineFromFile(std::ifstream &file) {
        char c = 0;
        while (c != '\n' && !file.eof() && file.good()) {
            file >> std::noskipws >> c;
        }
        file >> std::skipws;
    }

    inline void computeDefaultBlockOffsets(row_type size, row_type blocks, std::vector<row_type> &blockOffsets,
                                           row_type start = 0) {

        blockOffsets.resize(blocks);
        row_type minSize = size / blocks;
        row_type remainder = size % blocks;
        for (row_type i = 0; i < blocks; ++i) {
            if (i == 0) {
                blockOffsets[i] = start;
            } else {
                blockOffsets[i] = minSize + blockOffsets[i - 1];
                if (remainder > 0) {
                    ++blockOffsets[i];
                    --remainder;
                }
            }
        }
    };

    inline void scaleAndCopy(float *v1, const float *v2, float scale, col_type colNum) {
        for (int j = 0; j < colNum; ++j) {
            v1[j] = v2[j] * scale;
        }
    }

    inline void copy(float *v1, const float *v2, col_type colNum) {
        for (int j = 0; j < colNum; ++j) {
            v1[j] = v2[j];
        }
        //         std::memcpy((void*) v1, (void*) v2, sizeof (float)*colNum);

    }

    inline float calculateLength(const float *vec, col_type colNum) {
        float len = 0;
        for (int j = 0; j < colNum; ++j) {
            len += vec[j] * vec[j];
        }
        return sqrt(len);
    }

    class VectorMatrixLEMP {
        float *data;
        bool shuffled, normalized, extraMult;
        row_type offset;
        col_type lengthOffset;
        int sizeDiv2; // for simd instruction  

        inline void zeroOutLastPadding() {
            for (row_type i = 0; i < rowNum; ++i) {
                data[(i + 1) * offset - 1] = 0; // zero-out the last padding
            }
        }

        // stuff that needs to be done in both read methods
        // rowNum and colNum need to be initialized before calling this method

        inline void readFromFileCommon() {
            if (pow(2, sizeof(col_type) * 8) - 1 < colNum) {
                std::cerr << "Your vectors have dimensionality " << colNum
                          << " which is more than what lemp is compiled to store. Change the col_type in BasicStructs.h and recompile!"
                          << std::endl;
                exit(1);
            }
            if (pow(2, sizeof(row_type) * 8) - 1 < rowNum) {
                std::cerr << "Your dataset has " << rowNum
                          << " vectors which is more than what lemp is compiled to store. Change the row_type in BasicStructs.h and recompile!"
                          << std::endl;
                exit(1);
            }

            initializeBasics(colNum, rowNum, false);

            if (colNum < NUM_LISTS) {
                std::cout << "[WARNING] Your vectors have dimensionality" << colNum
                          << " and the tuner will try to search among " << NUM_LISTS <<
                          ". Perhaps you want to change the parameter NUM_LISTS in Definitions.h and recompile!"
                          << std::endl;
            }
            if (LOWER_LIMIT_PER_BUCKET >= rowNum) {
                std::cout << "[WARNING] You have " << rowNum
                          << " vectors and the tuner will try to take a sample of at least  " << LOWER_LIMIT_PER_BUCKET
                          <<
                          " vectors per probe bucket. Perhaps you want to change the parameter LOWER_LIMIT_PER_BUCKET in Definitions.h and recompile!"
                          << std::endl;
            }

            for (int i = 0; i < rowNum; i++) {
                setLengthInData(i, 1);
            }
        }

        inline void readFromFileCSV(const std::string &fileName, ta_size_type col, ta_size_type row) {
            std::ifstream file(fileName.c_str(), std::ios_base::in);

            if (!file.is_open()) {
                std::cout << "[ERROR] Fail to open file: " << fileName << std::endl;
                exit(1);
            }

            rowNum = row;
            colNum = col;

            std::cout << "[INFO] VectorMatrix will be read from " << fileName << " (" << rowNum
                      << " vectors with dimensionality " << (0 + colNum) << ")" << std::endl;

            VectorMatrixLEMP::readFromFileCommon();

            std::string buffer;
            if (file) {
                for (ta_size_type i = 0; i < row; i++) {
                    float *d = getMatrixRowPtr(i);

                    for (ta_size_type j = 0; j < col; j++) {
                        float f;
                        file >> f;
                        if (j != col - 1) {
                            std::getline(file, buffer, ',');
                        }
                        d[j] = f;
                    }
                    std::getline(file, buffer);
                }
            }

            file.close();
        }

        inline void readFromFileMMA(const std::string &fileName, bool left = true) {
            std::ifstream file(fileName.c_str(), std::ios_base::in);

            if (!file.is_open()) {
                std::cout << "[ERROR] Fail to open file: " << fileName << std::endl;
                exit(1);
            }

            while (file.peek() == '%') {
                skipLineFromFile(file);
            }

            ta_size_type col; // columns
            ta_size_type row; // rows
            file >> row >> col;

            rowNum = (left ? row : col);
            colNum = (left ? col : row);

            std::cout << "[INFO] VectorMatrix will be read from " << fileName << " (" << rowNum
                      << " vectors with dimensionality " << (0 + colNum) << ")" << std::endl;

            VectorMatrixLEMP::readFromFileCommon();

            if (left) {
                if (file) {
                    for (ta_size_type i = 0; i < col; i++) {// read one column
                        for (ta_size_type j = 0; j < row; j++) {
                            float f;
                            file >> f;

                            float *d = getMatrixRowPtr(j);
                            d[i] = f;
                        }
                    }
                }
                file.close();
            } else {
                if (file) {
                    for (ta_size_type i = 0; i < col; i++) {// read one column
                        for (ta_size_type j = 0; j < row; j++) {
                            float f;
                            file >> f;

                            float *d = getMatrixRowPtr(i);
                            d[j] = f;
                        }
                    }
                }
                file.close();
            }
        }

        //        const VectorMatrix& operator =(const VectorMatrix& m);

    public:

        std::vector<float> cweights; // forAP
        std::vector<float> maxVectorCoord; // forAP
        std::vector<row_type> vectorNNZ; // forAP
        std::vector<QueueElement> lengthInfo; // data: length id: vectorId
        std::vector<float> epsilonEquivalents;
        col_type colNum;
        row_type rowNum;

        friend void splitMatrices(const VectorMatrixLEMP &originalMatrix, std::vector<VectorMatrixLEMP> &matrices);

        friend void
        initializeMatrices(const VectorMatrixLEMP &originalMatrix, std::vector<VectorMatrixLEMP> &matrices, bool sort,
                           bool ignoreLengths, float epsilon);

        inline VectorMatrixLEMP() : data(nullptr), shuffled(false), normalized(false),
                                    lengthOffset(1) {////////////////////// 1 is for padding
        }

        inline VectorMatrixLEMP(const std::vector<std::vector<float> > m) : data(nullptr), shuffled(false),
                                                                             normalized(false), lengthOffset(1) {

            initializeBasics(m[0].size(), m.size(), false);

//#pragma omp parallel for default(none) shared(m) schedule(static, 1000)
            for (int i = 0; i < rowNum; ++i) {
                float *v1 = getMatrixRowPtr(i);
                const float *v2 = &m[i][0];
//                std::memcpy((void*) v1, (void*) v2, sizeof (float)*colNum);

                copy(v1, v2, colNum);

//                for (int j = 0; j < colNum; ++j) {
////                    v1[j] = v2[j];
//                    std::cout<<v1[j]<<" ";
//                }
//                std::cout<<std::endl;
            }

//            std::cout<<"offset: "<<(int)offset<<" "<<(int)lengthOffset<<std::endl;

        }

        inline VectorMatrixLEMP(const float *raw_data_ptr, const int &n_vecs, const int &vec_dim) : data(nullptr),
                                                                                                    shuffled(false),
                                                                                                    normalized(false),
                                                                                                    lengthOffset(1) {

            initializeBasics(vec_dim, n_vecs, false);

//#pragma omp parallel for default(none) shared(raw_data_ptr) schedule(static, 1000)
            for (int vecsID = 0; vecsID < rowNum; ++vecsID) {
                float *v1 = getMatrixRowPtr(vecsID);
                const float *v2 = raw_data_ptr + (size_t) vecsID * colNum;
                for (int j = 0; j < colNum; ++j) {
                    v1[j] = v2[j];
                }
            }

        }

        VectorMatrixLEMP &operator=(const VectorMatrixLEMP &r) {
            colNum = r.colNum;
            rowNum = r.rowNum;
            shuffled = r.shuffled;
            normalized = r.normalized;
            extraMult = r.extraMult;
            offset = r.offset;
            lengthOffset = r.lengthOffset;
            sizeDiv2 = r.sizeDiv2;


            lengthInfo.clear();
            lengthInfo.reserve(r.lengthInfo.size());
            std::copy(r.lengthInfo.begin(), r.lengthInfo.end(), back_inserter(lengthInfo));

            cweights.clear();
            cweights.reserve(r.cweights.size());
            std::copy(r.cweights.begin(), r.cweights.end(), back_inserter(cweights));

            maxVectorCoord.clear();
            maxVectorCoord.reserve(r.maxVectorCoord.size());
            std::copy(r.maxVectorCoord.begin(), r.maxVectorCoord.end(), back_inserter(maxVectorCoord));

            vectorNNZ.clear();
            vectorNNZ.reserve(r.vectorNNZ.size());
            std::copy(r.vectorNNZ.begin(), r.vectorNNZ.end(), back_inserter(vectorNNZ));

            epsilonEquivalents.clear();
            epsilonEquivalents.reserve(r.epsilonEquivalents.size());
            std::copy(r.epsilonEquivalents.begin(), r.epsilonEquivalents.end(), back_inserter(epsilonEquivalents));

            int res = posix_memalign((void **) &(data), 16, sizeof(float) * offset * rowNum);

            if (res != 0) {
                std::cout << "[ERROR] Problem with allocating memory for VectorMatrix!" << std::endl;
                exit(1);
            }

            std::memcpy((void *) data, (void *) r.data, sizeof(float) * offset * rowNum);
            return *this;
        }

        inline ~VectorMatrixLEMP() {
            if (data != nullptr) {
                free(data);
                data = nullptr;
            }

        }

        inline void initializeBasics(col_type numOfColumns, row_type numOfRows, bool norm) {
            colNum = numOfColumns;
            offset = colNum + 2;
            sizeDiv2 = colNum & (-2);
            extraMult = (sizeDiv2 < colNum);
            if (extraMult)
                offset++;


            rowNum = numOfRows;

            normalized = norm;
            lengthInfo.resize(rowNum);
            int res = posix_memalign((void **) &(data), 16, sizeof(float) * offset * rowNum);

            if (res != 0) {
                std::cout << "[ERROR] Problem with allocating memory for VectorMatrix!" << std::endl;
                exit(1);
            }

            if (extraMult) {
                zeroOutLastPadding();
            }
        }

        inline void readFromFile(const std::string &fileName, int numCoordinates, int numVectors, bool left = true) {
            if (boost::algorithm::ends_with(fileName, ".csv")) {
                if (numCoordinates == 0 || numVectors == 0) {
                    std::cerr
                            << "When using csv files, you should provide the number of coordinates (--r) and the number of vectors (--m or --n)!"
                            << std::endl;
                    exit(1);
                }
                readFromFileCSV(fileName, numCoordinates, numVectors);
            } else if (boost::algorithm::ends_with(fileName, ".mma")) {
                readFromFileMMA(fileName, left);
            } else {
                std::cerr << "No valid input file format to read a VectorMatrix from!" << std::endl;
                exit(1);
            }
        }

        inline void init(const VectorMatrixLEMP &matrix, bool sort, bool ignoreLength) {
            initializeBasics(matrix.colNum, matrix.rowNum, true);


            if (ignoreLength) {
//#pragma omp parallel for  schedule(static, 1000)
                // get lengths
                for (int i = 0; i < rowNum; ++i) {
                    const float *vec = matrix.getMatrixRowPtr(i);
                    float len = calculateLength(vec, colNum);
                    lengthInfo[i] = QueueElement(1, i);
                    setLengthInData(i, 1);
                    float x = 1 / len;
                    float *d1 = getMatrixRowPtr(i);
                    scaleAndCopy(d1, vec, x, colNum);
                }


            } else {
//#pragma omp parallel for schedule(static, 1000)
                for (int i = 0; i < rowNum; ++i) {
                    const float *vec = matrix.getMatrixRowPtr(i);
                    float len = calculateLength(vec, colNum);
                    lengthInfo[i] = QueueElement(len, i);
                }

                if (sort) {
                    shuffled = true;
                    std::sort(lengthInfo.begin(), lengthInfo.end(), std::greater<QueueElement>());
                }


//#pragma omp parallel for schedule(static, 1000)
                for (int i = 0; i < rowNum; ++i) {
                    setLengthInData(i, lengthInfo[i].data);
                    float x = 1 / lengthInfo[i].data;
                    float *d1 = getMatrixRowPtr(i);
                    float *d2 = matrix.getMatrixRowPtr(lengthInfo[i].id);
                    scaleAndCopy(d1, d2, x, colNum);
                }

            }
        }

        inline void addVectors(const VectorMatrixLEMP &matrix, const std::vector<row_type> &dataIds) {
            initializeBasics(matrix.colNum, dataIds.size(), false);
            for (int i = 0; i < rowNum; ++i) {
                const float *vec = matrix.getMatrixRowPtr(dataIds[i]);
                lengthInfo[i] = QueueElement(1, dataIds[i]);
                float *d1 = getMatrixRowPtr(i);
                scaleAndCopy(d1, vec, 1, colNum);
            }
        }

        inline float *getMatrixRowPtr(row_type row) const {// the row starts from pos 1. Do ptr[-1] to get the length
            return &data[row * offset + 1 + lengthOffset];
        }

        inline void print(row_type row) const {

            const float *vec = getMatrixRowPtr(row);


            for (int i = 0; i < colNum; ++i) {

                std::cout << i << ":" << vec[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "Length: " << vec[-1] << " or " << lengthInfo[row].data << std::endl;
            std::cout << "hasId: " << lengthInfo[row].id << std::endl;

        }

        inline float getVectorLength(row_type row) const {
            return data[row * offset + lengthOffset];
        }

        inline float setLengthInData(row_type row, float len) {
            return data[row * offset + lengthOffset] = len;
        }

        inline row_type getId(row_type row) const {
            return (normalized ? lengthInfo[row].id : row);
        }

        inline float cosine(row_type row, const float *query) const {

            const float *d_ptr = getMatrixRowPtr(row);
            float cosine = 0;

#ifdef  WITH_SIMD
            __m128d sum = _mm_set1_pd(0.0);
            int size = colNum + extraMult;
            for (int i = 0; i < size; i += 2) {
                sum = _mm_add_pd(sum, _mm_mul_pd(_mm_load_pd(d_ptr + i), _mm_load_pd(query + i)));
            }
            cosine = _mm_cvtsd_f64(_mm_hadd_pd(sum, sum));
            return cosine;
#else
            for (int i = 0; i < colNum; ++i) {
                cosine += query[i] * d_ptr[i];
            }
            return cosine;
#endif
        }

        inline float L2Distance(row_type row, const float *query) const {

            const float *d_ptr = getMatrixRowPtr(row);
            float dist = 0;

            if (normalized) {
                for (int i = 0; i < colNum; ++i) {
                    float value = query[i] * query[-1] - d_ptr[i] * d_ptr[-1]; // unnormalize
                    dist += value * value;
                }
            } else {
                for (int i = 0; i < colNum; ++i) {
                    dist += (query[i] - d_ptr[i]) * (query[i] - d_ptr[i]);
                }
            }
            return sqrt(dist);
        }

        inline float L2Distance2(row_type row, const float *query) const {
            // I assume non normalized case as needed in PCA trees
            const float *d_ptr = getMatrixRowPtr(row);
            float dist = 0;

            for (int i = 0; i < colNum; ++i) {
                dist += (query[i] - d_ptr[i]) * (query[i] - d_ptr[i]);
            }
            return dist;
        }

        inline float innerProduct(row_type row, const float *query) const {
            const float ip = query[-1] * getVectorLength(row) * cosine(row, query);
            return ip;
        }

        inline std::pair<bool, float> passesThreshold(row_type row, const float *query, float theta) const {

            std::pair<bool, float> p;
            float ip = 1;

            if (normalized) {
                ip = query[-1] * getVectorLength(row);

                if (ip < theta) {
                    p.first = false;
                    return p;
                }
            }

            ip *= cosine(row, query);
            p.second = ip;

            if (ip < theta) {
                p.first = false;
                return p;
            } else {
                p.first = true;
                return p;
            }
        }


    };

    // ignores the lengths

    inline void splitMatrices(const VectorMatrixLEMP &originalMatrix, std::vector<VectorMatrixLEMP> &matrices) {
        row_type threads = matrices.size();

        if (threads == 1) {
            matrices[0].initializeBasics(originalMatrix.colNum, originalMatrix.rowNum, false);

            for (int i = 0; i < matrices[0].rowNum; ++i) {
                const float *vec = originalMatrix.getMatrixRowPtr(i);
                matrices[0].lengthInfo[i] = QueueElement(1, i);
                matrices[0].setLengthInData(i, 1);
                float *d1 = matrices[0].getMatrixRowPtr(i);
                scaleAndCopy(d1, vec, 1, originalMatrix.colNum);
            }
        } else {
//            omp_set_num_threads(threads);

            std::vector<row_type> permuteVector(originalMatrix.rowNum);
            std::iota(permuteVector.begin(), permuteVector.end(), 0);

//            rg::Random32 random(123);
//            rg::shuffle(permuteVector.begin(), permuteVector.end(), random);

            std::mt19937 g(123);
            std::shuffle(permuteVector.begin(), permuteVector.end(), g);

            std::vector<row_type> blockOffsets;
            computeDefaultBlockOffsets(permuteVector.size(), threads, blockOffsets);

//#pragma omp parallel
            {
//                row_type tid = omp_get_thread_num();
                row_type tid = 0;

                row_type start = blockOffsets[tid];
                row_type end = (tid == blockOffsets.size() - 1 ? originalMatrix.rowNum : blockOffsets[tid + 1]);

                matrices[tid].initializeBasics(originalMatrix.colNum, end - start, true);

                for (int i = start; i < end; ++i) {
                    row_type ind = permuteVector[i];
                    const float *vec = originalMatrix.getMatrixRowPtr(ind);
                    matrices[tid].lengthInfo[i - start] = QueueElement(1, i - start);
                    matrices[tid].setLengthInData(i - start, 1);
                    float *d1 = matrices[tid].getMatrixRowPtr(i - start);
                    scaleAndCopy(d1, vec, 1, originalMatrix.colNum);
                    matrices[tid].lengthInfo[i - start].id = ind; // the original id
                }
            }
        }
    }

    /*  map: id: original matrix id, first: thread second: posInMatrix
     */
    inline void
    initializeMatrices(const VectorMatrixLEMP &originalMatrix, std::vector<VectorMatrixLEMP> &matrices, bool sort,
                       bool ignoreLengths, float epsilon = 0) {

        row_type threads = matrices.size();


        if (threads == 1) {

            matrices[0].initializeBasics(originalMatrix.colNum, originalMatrix.rowNum, true);


            if (ignoreLengths) {

#if defined(ABS_APPROX) || defined(HYBRID_APPROX)
                matrices[0].epsilonEquivalents.resize(matrices[0].rowNum, epsilon);
#endif


                for (int i = 0; i < matrices[0].rowNum; ++i) {
                    const float *vec = originalMatrix.getMatrixRowPtr(i);
                    float len = calculateLength(vec, matrices[0].colNum);
                    matrices[0].lengthInfo[i] = QueueElement(1, i);
                    matrices[0].setLengthInData(i, 1);
                    float x = 1 / len;
                    float *d1 = matrices[0].getMatrixRowPtr(i);
                    scaleAndCopy(d1, vec, x, originalMatrix.colNum);
#if defined(ABS_APPROX) || defined(HYBRID_APPROX)
                    matrices[0].epsilonEquivalents[i] *= x;
#endif
                }

            } else {
                for (int i = 0; i < matrices[0].rowNum; ++i) {
                    const float *vec = originalMatrix.getMatrixRowPtr(i);
                    float len = calculateLength(vec, matrices[0].colNum);
                    matrices[0].lengthInfo[i] = QueueElement(len, i);
                }

                if (sort) {
                    matrices[0].shuffled = true;
                    std::sort(matrices[0].lengthInfo.begin(), matrices[0].lengthInfo.end(),
                              std::greater<QueueElement>());
                }

                for (int i = 0; i < matrices[0].rowNum; ++i) {
                    matrices[0].setLengthInData(i, matrices[0].lengthInfo[i].data);
                    float x = 1 / matrices[0].lengthInfo[i].data;
                    float *d1 = matrices[0].getMatrixRowPtr(i);
                    float *d2 = originalMatrix.getMatrixRowPtr(matrices[0].lengthInfo[i].id);
                    scaleAndCopy(d1, d2, x, originalMatrix.colNum);
                }

            }

        } else { // multiple threads

//            omp_set_num_threads(threads);

            std::vector<row_type> permuteVector(originalMatrix.rowNum);
            std::iota(permuteVector.begin(), permuteVector.end(), 0);


//            rg::Random32 random(123);
//            rg::shuffle(permuteVector.begin(), permuteVector.end(), random);

            std::mt19937 g(123);
            std::shuffle(permuteVector.begin(), permuteVector.end(), g);

            std::vector<row_type> blockOffsets;
            computeDefaultBlockOffsets(permuteVector.size(), threads, blockOffsets);


//#pragma omp parallel
            {
//                row_type tid = omp_get_thread_num();
                row_type tid = 0;

                row_type start = blockOffsets[tid];
                row_type end = (tid == blockOffsets.size() - 1 ? originalMatrix.rowNum : blockOffsets[tid + 1]);

                matrices[tid].initializeBasics(originalMatrix.colNum, end - start, true);

                if (ignoreLengths) {

#if defined(ABS_APPROX) || defined(HYBRID_APPROX)
                    matrices[tid].epsilonEquivalents.resize(matrices[tid].rowNum, epsilon);
#endif

                    for (int i = start; i < end; ++i) {
                        row_type ind = permuteVector[i];
                        const float *vec = originalMatrix.getMatrixRowPtr(ind);
                        float len = calculateLength(vec, matrices[tid].colNum);
                        matrices[tid].lengthInfo[i - start] = QueueElement(1, i - start);
                        matrices[tid].setLengthInData(i - start, 1);
                        float x = 1 / len;
                        float *d1 = matrices[tid].getMatrixRowPtr(i - start);
                        scaleAndCopy(d1, vec, x, originalMatrix.colNum);
                        matrices[tid].lengthInfo[i - start].id = ind; // the original id
#if defined(ABS_APPROX) || defined(HYBRID_APPROX)
                        matrices[tid].epsilonEquivalents[i] *= x;
#endif
                    }
                } else {
                    for (int i = start; i < end; ++i) {
                        row_type ind = permuteVector[i];
                        const float *vec = originalMatrix.getMatrixRowPtr(ind);
                        float len = calculateLength(vec, matrices[tid].colNum);
                        matrices[tid].lengthInfo[i - start] = QueueElement(len, i - start);
                    }

                    if (sort) {
                        matrices[tid].shuffled = true;
                        std::sort(matrices[tid].lengthInfo.begin(), matrices[tid].lengthInfo.end(),
                                  std::greater<QueueElement>());
                    }


                    for (int i = 0; i < matrices[tid].rowNum; ++i) {
                        matrices[tid].setLengthInData(i, matrices[tid].lengthInfo[i].data);
                        float x = 1 / matrices[tid].lengthInfo[i].data;
                        row_type ind = permuteVector[matrices[tid].lengthInfo[i].id + start];
                        float *d1 = matrices[tid].getMatrixRowPtr(i);
                        float *d2 = originalMatrix.getMatrixRowPtr(ind);
                        scaleAndCopy(d1, d2, x, originalMatrix.colNum);
                        matrices[tid].lengthInfo[i].id = ind; // the original id
                    }
                }
            }

        }
    }


}
#endif /* VECTORMATRIX_H_ */
