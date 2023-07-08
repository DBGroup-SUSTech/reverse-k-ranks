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
 * CandidateVerification.h
 *
 *  Created on: May 12, 2014
 *      Author: chteflio
 */

#ifndef CANDIDATEVERIFICATION_H_
#define CANDIDATEVERIFICATION_H_


#include <iterator>

#include "RetrievalArguments.h"
#include "VectorMatrixLEMP.h"


namespace mips {

    float countABOVE = 0;


    inline float approximateHybridError(RetrievalArguments &arg, float lengthNextBucket, float invLengthNextBucket) {
        float error = 0;
        float countAbove = 0;


        for (row_type i = 0; i < arg.k; i++) {
            if (arg.heap[i].data >= lengthNextBucket) {
                countAbove++;
            } else {
                error += 1 - arg.heap[i].data * invLengthNextBucket;
            }
        }
        error += countAbove * (1 - arg.R);

        countABOVE += countAbove;
        return error / arg.k;
    }


    inline float approximateRelError(RetrievalArguments &arg) {
        float error = 0;
        float margin = arg.heap.front().data * (1 + arg.epsilon);
        float invMargin = 1 / margin;

        for (row_type i = 0; i < arg.k; i++) {
            if (arg.heap[i].data < margin) {
                error += 1 - arg.heap[i].data * invMargin;
            }
        }

        return error / arg.k;
    }

    inline void
    verifyCandidates_lengthTest(const float *query, row_type numCandidatesToVerify, RetrievalArguments *arg) {
        std::pair<bool, float> p;

        for (row_type i = 0; i < numCandidatesToVerify; ++i) {
            row_type row = arg->candidatesToVerify[i];
            p = arg->probeMatrix->passesThreshold(row, query, arg->theta);

            if (p.first) {
                arg->n_results++;
//                arg->results.emplace_back(p.second, arg->queryId, arg->probeMatrix->getId(row));
            }
        }
        arg->comparisons += numCandidatesToVerify;

    }

    inline void
    verifyCandidates_noLengthTest(const float *query, row_type numCandidatesToVerify, RetrievalArguments *arg,
                                  const int &max_rank) {

        for (row_type i = 0; i < numCandidatesToVerify; ++i) {
            row_type row = arg->candidatesToVerify[i];
            float ip = arg->probeMatrix->innerProduct(row, query);
            arg->comparisons++;
            if (ip >= arg->theta) {
                arg->n_results++;
//                 arg->results.emplace_back(ip, arg->queryId, arg->probeMatrix->getId(row));
//                 std::cout<<"row: "<<row<<" id: "<<arg->probeMatrix->getId(row)<<" ip: "<<ip<<std::endl;
            }
            if (arg->n_results > max_rank) {
                break;
            }
        }
//        std::cout<<"RS: "<<arg->results.size()<<std::endl;

    }

    inline void
    verifyCandidatesTopK_noLengthTest(const float *query, row_type numCandidatesToVerify, RetrievalArguments *arg) {

        float minScore = arg->heap.front().data;

        for (row_type i = 0; i < numCandidatesToVerify; ++i) {
            row_type row = arg->candidatesToVerify[i];
            float ip = arg->probeMatrix->innerProduct(row, query);

            if (ip > minScore) {
                std::pop_heap(arg->heap.begin(), arg->heap.end(), std::greater<QueueElement>());
                arg->heap.pop_back();
                arg->heap.emplace_back(ip, arg->probeMatrix->getId(row));
                std::push_heap(arg->heap.begin(), arg->heap.end(), std::greater<QueueElement>());
                minScore = arg->heap.front().data;
            }
        }
        arg->comparisons += numCandidatesToVerify;

    }

    inline void
    verifyCandidatesTopK_lengthTest(const float *query, row_type numCandidatesToVerify, RetrievalArguments *arg) {


        float minScore = arg->heap.front().data;

        for (row_type i = 0; i < numCandidatesToVerify; ++i) {
            row_type row = arg->candidatesToVerify[i];

            if (arg->probeMatrix->getVectorLength(row) <= minScore)
                continue;

            float ip = arg->probeMatrix->innerProduct(row, query);
            arg->comparisons++;

            if (ip > minScore) {
                std::pop_heap(arg->heap.begin(), arg->heap.end(), std::greater<QueueElement>());
                arg->heap.pop_back();
                arg->heap.emplace_back(ip, arg->probeMatrix->getId(row));
                std::push_heap(arg->heap.begin(), arg->heap.end(), std::greater<QueueElement>());
                minScore = arg->heap.front().data;
            }
        }
    }


    // These are used by TA
    // examines if the item should be included in the result and does the corresponding housekeeping

    inline void verifyCandidate(row_type posMatrix, const float *query, RetrievalArguments *arg) {
        arg->comparisons++;

        std::pair<bool, float> p;
        p = arg->probeMatrix->passesThreshold(posMatrix, query, arg->theta);

        if (p.first) {
            arg->n_results++;
//             arg->results.emplace_back(p.second, arg->queryId, arg->probeMatrix->getId(posMatrix));
        }
    }

    inline void verifyCandidateTopk(row_type posMatrix, const float *query, RetrievalArguments *arg) {
        std::pair<bool, float> p;
        arg->comparisons++;
        p = arg->probeMatrix->passesThreshold(posMatrix, query, arg->heap.front().data);

        if (p.first) {
            // remove min element from the heap
            pop_heap(arg->heap.begin(), arg->heap.end(),
                     std::greater<QueueElement>()); // Yes! I need to use greater to get a min heap!
            arg->heap.pop_back();
            // and push new element inside the heap
            arg->heap.emplace_back(p.second, arg->probeMatrix->getId(posMatrix));
            push_heap(arg->heap.begin(), arg->heap.end(), std::greater<QueueElement>());

        }

    }


}


#endif /* CANDIDATEVERIFICATION_H_ */
