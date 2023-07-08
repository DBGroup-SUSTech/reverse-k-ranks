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
 * File:   icoord.h
 * Author: chteflio
 *
 * Created on December 14, 2014, 11:59 AM
 */

#ifndef ICOORD_H
#define    ICOORD_H

#include "ListsTuneData.h"


namespace mips {

    class IncrRetriever : public Retriever {
    public:


        IncrRetriever() = default;

        ~IncrRetriever() = default;

        inline void run(const float *query, ProbeBucket &probeBucket, RetrievalArguments *arg, const int& max_rank) const {

            QueueElementLists *invLists = static_cast<QueueElementLists *> (probeBucket.getIndex(SL));

            float qi;
            row_type numCandidatesToVerify = 0;


            float seenQi2 = 1;

            float localTheta = probeBucket.bucketScanThreshold / query[-1];

#ifdef TIME_IT
            arg->t.start();
#endif
            bool shouldScan = invLists->calculateIntervals(query, arg->listsQueue, arg->intervals, localTheta, arg->numLists);
#ifdef TIME_IT
            arg->t.stop();
            arg->boundsTime += arg->t.elapsedTime().nanos();
#endif

            if (shouldScan) {

                if (arg->numLists == 1) {
                    scanOnly1List(query, probeBucket, arg, invLists, false);

                } else {

#ifdef TIME_IT
                    arg->t.start();
#endif
                    //initialize
                    qi = query[arg->intervals[0].col];
                    seenQi2 -= qi * qi;

                    QueueElement *entry = invLists->getElement(arg->intervals[0].start);
                    row_type length = arg->intervals[0].end - arg->intervals[0].start;

                    for (row_type i = 0; i < length; ++i) {
                        arg->ext_cp_array[entry[i].id].addFirst(qi, entry[i].data);
                    }

                    // add Candidates
                    for (col_type j = 1; j < arg->numLists; ++j) {
                        qi = query[arg->intervals[j].col];

                        if (qi == 0)
                            continue;

                        seenQi2 -= qi * qi;

                        entry = invLists->getElement(arg->intervals[j].start);
                        length = arg->intervals[j].end - arg->intervals[j].start;

                        for (row_type i = 0; i < length; ++i) {
                            arg->ext_cp_array[entry[i].id].add(qi, entry[i].data);
                        }

                    }


#ifdef TIME_IT
                    arg->t.stop();
                    arg->scanTime += arg->t.elapsedTime().nanos();
                    arg->t.start();
#endif
                    // run first scan again to find items to verify
                    entry = invLists->getElement(arg->intervals[0].start);
                    length = arg->intervals[0].end - arg->intervals[0].start;

                    for (row_type i = 0; i < length; ++i) {
                        row_type row = entry[i].id;
                        float len = query[-1] * arg->probeMatrix->lengthInfo[row + probeBucket.startPos].data;

                        if (!arg->ext_cp_array[row].prune(len, arg->theta, seenQi2)) {
                            arg->candidatesToVerify[numCandidatesToVerify] = row + probeBucket.startPos;
                            numCandidatesToVerify++;
                        }


                    }

#ifdef TIME_IT
                    arg->t.stop();
                    arg->filterTime += arg->t.elapsedTime().nanos();
                    arg->t.start();
#endif
                    verifyCandidates_noLengthTest(query, numCandidatesToVerify, arg, max_rank);
#ifdef TIME_IT
                    arg->t.stop();
                    arg->ipTime += arg->t.elapsedTime().nanos();
#endif
                }
            }
        }

        inline void run(QueryBatch &queryBatch, ProbeBucket &probeBucket, RetrievalArguments *arg, const int& max_rank) const {
#ifdef TIME_IT
            arg->t.start();
#endif
            if (!queryBatch.hasInitializedQueues()) { //preprocess
                queryBatch.preprocess(*(arg->queryMatrix), arg->maxLists);
            }
#ifdef TIME_IT
            arg->t.stop();
            arg->preprocessTime += arg->t.elapsedTime().nanos();
#endif

            arg->numLists = probeBucket.numLists;


            for (row_type i = queryBatch.startPos; i < queryBatch.endPos; ++i) {
                const float *query = arg->queryMatrix->getMatrixRowPtr(i);

                if (query[-1] < probeBucket.bucketScanThreshold)// skip all users from this point on for this bucket
                    break;

                col_type *localQueue = queryBatch.getQueue(i - queryBatch.startPos, arg->maxLists);
                arg->setQueues(localQueue);
                arg->queryId = arg->queryMatrix->getId(i);

                run(query, probeBucket, arg, max_rank);

            }
        }

        // this runs practically COORD on 1 list.
        // Use it when the shortest necessary interval is REALLY short

        inline void scanOnly1List(const float *query, ProbeBucket &probeBucket, RetrievalArguments *arg,
                                  QueueElementLists *invLists, bool topk) const {
#ifdef TIME_IT
            arg->t.start();
#endif

            row_type numCandidatesToVerify = 0;

            QueueElement *entry = invLists->getElement(arg->intervals[0].start);
            row_type length = arg->intervals[0].end - arg->intervals[0].start;

            for (row_type i = 0; i < length; ++i) {
                arg->candidatesToVerify[numCandidatesToVerify] = entry[i].id + probeBucket.startPos;
                numCandidatesToVerify++;
            }


#ifdef TIME_IT
            arg->t.stop();
            arg->scanTime += arg->t.elapsedTime().nanos();
            arg->t.start();
#endif
            verifyCandidates_lengthTest(query, numCandidatesToVerify, arg);
#ifdef TIME_IT
            arg->t.stop();
            arg->ipTime += arg->t.elapsedTime().nanos();
#endif

        }


        inline virtual void
        tune(ProbeBucket &probeBucket, const ProbeBucket &prevBucket, std::vector<RetrievalArguments> &retrArg, const int& max_rank) {

            if (probeBucket.xValues->size() > 0) {
                ListTuneData dataForTuning;
                dataForTuning.tune<IncrRetriever>(probeBucket, prevBucket, retrArg, this, max_rank);
            } else {
                probeBucket.setAfterTuning(prevBucket.numLists, prevBucket.t_b);
            }
        }

        inline virtual void run(ProbeBucket &probeBucket, RetrievalArguments *arg, const int& max_rank) const {

            arg->numLists = probeBucket.numLists;

            for (auto &queryBatch: arg->queryBatches) {

                if (queryBatch.maxLength() < probeBucket.bucketScanThreshold) {
                    break;
                }

#ifdef TIME_IT
                arg->t.start();
#endif
                if (!queryBatch.hasInitializedQueues()) { //preprocess
                    queryBatch.preprocess(*(arg->queryMatrix), arg->maxLists);
                }
#ifdef TIME_IT
                arg->t.stop();
                arg->preprocessTime += arg->t.elapsedTime().nanos();
#endif

                for (row_type i = queryBatch.startPos; i < queryBatch.endPos; ++i) {
                    const float *query = arg->queryMatrix->getMatrixRowPtr(i);

                    if (query[-1] < probeBucket.bucketScanThreshold)// skip all users from this point on for this bucket
                        break;

                    col_type *localQueue = queryBatch.getQueue(i - queryBatch.startPos, arg->maxLists);
                    arg->setQueues(localQueue);
                    arg->queryId = arg->queryMatrix->getId(i);

                    run(query, probeBucket, arg, max_rank);
                }


            }

        }

    };
}

#endif    /* ICOORD_H */

