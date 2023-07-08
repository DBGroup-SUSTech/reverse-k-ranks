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
 * mixed.h
 *
 *  Created on: Jun 27, 2014
 *      Author: chteflio
 */

#ifndef MIXED_H_
#define MIXED_H_


namespace mips {

    template<class X> // where X can be C or I
    class LX_Retriever : public Retriever {
    public:
        LengthRetriever plainRetriever;
        X otherRetriever;

        LX_Retriever() = default;

        ~LX_Retriever() = default;

        inline LengthRetriever *getLengthRetriever() {
            return &plainRetriever;
        }

        inline virtual void
        tune(ProbeBucket &probeBucket, const ProbeBucket &prevBucket, std::vector<RetrievalArguments> &retrArg, const int& max_rank) {

            if (probeBucket.xValues->size() > 0) {

                plainRetriever.tune(probeBucket, prevBucket, retrArg, max_rank);
                retrArg[0].competitorMethod = &plainRetriever.sampleTimes;


                otherRetriever.tune(probeBucket, prevBucket, retrArg, max_rank);

                if (plainRetriever.sampleTotalTime < otherRetriever.sampleTotalTime) {
                    probeBucket.setAfterTuning(1, 1);
                }
            } else {
                probeBucket.setAfterTuning(prevBucket.numLists, prevBucket.t_b);
            }
        }

//        inline virtual void run(ProbeBucket &probeBucket, RetrievalArguments *arg) const {
//            arg->numLists = probeBucket.numLists;
//            QueryBatch &queryBatch = arg->queryBatches[0];
//
//            if (queryBatch.maxLength() < probeBucket.bucketScanThreshold) {
//                return;
//            }
//
//            plainRetriever.run(queryBatch, probeBucket, arg);
//
//        }

        inline virtual void run(ProbeBucket &probeBucket, RetrievalArguments *arg, const int& max_rank) const {
            arg->numLists = probeBucket.numLists;
            for (auto &queryBatch: arg->queryBatches) {

                if (queryBatch.maxLength() < probeBucket.bucketScanThreshold) {
                    break;
                }

                if (probeBucket.t_b == 1 ||
                    (probeBucket.t_b * queryBatch.minLength() > probeBucket.bucketScanThreshold)) {
                    plainRetriever.run(queryBatch, probeBucket, arg, max_rank);
                } else if (probeBucket.t_b * queryBatch.maxLength() <= probeBucket.bucketScanThreshold) {
                    otherRetriever.run(queryBatch, probeBucket, arg, max_rank);
                } else { // do it per query
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

                        if (query[-1] <
                            probeBucket.bucketScanThreshold)// skip all users from this point on for this bucket
                            break;

                        arg->queryId = arg->queryMatrix->getId(i);
                        if (probeBucket.t_b * query[-1] > probeBucket.bucketScanThreshold) {// do length-based
                            plainRetriever.run(query, probeBucket, arg, max_rank);
                        } else {

                            col_type *localQueue = queryBatch.getQueue(i - queryBatch.startPos, arg->maxLists);
                            arg->setQueues(localQueue);
                            otherRetriever.run(query, probeBucket, arg, max_rank);
                        }

                    }
                }
            }

        }

    };


}


#endif /* MIXED_H_ */
