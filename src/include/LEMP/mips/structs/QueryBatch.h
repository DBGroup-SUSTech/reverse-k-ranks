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
 * QueryBucket_withTuning2.h
 *
 *  Created on: Mar 25, 2014
 *      Author: chteflio
 */

#ifndef QUERYBATCH_H_
#define QUERYBATCH_H_

namespace mips {

    float matrixToMatrixTime2 = 0;

    class QueryBatch {
        col_type* queues;
        bool initializedQueues;
        row_type rowNum;
        std::vector<bool> inactiveQueries;
        row_type inactiveCounter;
        std::pair<float, float> normL2; // first: min second: max
        
    public:

        row_type startPos, endPos;
        

        inline bool hasInitializedQueues() const {
            return initializedQueues;
        }

        inline bool isWorkDone() const {
            return (inactiveCounter == rowNum);
        }

        inline void inactivateQuery(row_type queryPosInWholeMatrix) {
            inactiveQueries[queryPosInWholeMatrix - startPos] = true;
            inactiveCounter++;
        }

        inline bool isQueryInactive(row_type queryPosInWholeMatrix) const {
            return inactiveQueries[queryPosInWholeMatrix - startPos];
        }

        inline float maxLength() const{
            return normL2.second;
        }

        inline float minLength() const{
            return normL2.first;
        }


        inline void preprocess(const VectorMatrixLEMP& userMatrix, col_type maxLists);

        inline col_type* getQueue(row_type user, col_type maxLists) const {
            return &queues[user * maxLists];
        }


        inline QueryBatch() : initializedQueues(false), queues(nullptr), inactiveCounter(0) {
        };

        inline ~QueryBatch() {
            if (queues != nullptr)
                delete[] queues;

        }

        inline void init(const VectorMatrixLEMP& matrix, row_type startInd, row_type endInd, const LempArguments& args) {
            startPos = startInd;
            endPos = endInd;
            rowNum = endPos - startPos;
            normL2.second = matrix.getVectorLength(startPos);
            normL2.first = matrix.getVectorLength(endPos - 1);

            if (args.k > 0) {
                inactiveQueries.resize(rowNum);
            }
        }

    };

    inline void QueryBatch::preprocess(const VectorMatrixLEMP& userMatrix, col_type maxLists) {

        queues = new col_type[rowNum * maxLists];

        std::vector<QueueElement> tmp;
        tmp.resize(maxLists);

        for (row_type j = 0; j < rowNum; ++j) {
            const float* query = userMatrix.getMatrixRowPtr(j + startPos);

            for (col_type i = 0; i < maxLists; ++i) {
                float value = fabs(query[i]);

                tmp[i] = QueueElement(value, i);
            }
            std::make_heap(tmp.begin(), tmp.end(), std::greater<QueueElement>());

            for (col_type i = maxLists; i < userMatrix.colNum; ++i) {
                float value = fabs(query[i]);


                if (value > tmp.front().data) {
                    std::pop_heap(tmp.begin(), tmp.end(), std::greater<QueueElement>());
                    tmp.pop_back();
                    tmp.emplace_back(value, i);
                    std::push_heap(tmp.begin(), tmp.end(), std::greater<QueueElement>());
                }
            }

            std::sort(tmp.begin(), tmp.end(), std::greater<QueueElement>());
            for (col_type i = 0; i < maxLists; ++i) {
                queues[j * maxLists + i] = tmp[i].id;
            }
        }

        initializedQueues = true;

    }

}

#endif /* QUERYBATCH_H_ */
