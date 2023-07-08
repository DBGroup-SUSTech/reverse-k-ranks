/*
 * Bucket.h
 *
 *  Created on: Nov 19, 2013
 *      Author: chteflio
 */

#ifndef PROBEBUCKET_H_
#define PROBEBUCKET_H_


#include <boost/shared_ptr.hpp>

namespace mips {

    enum Index_Type {
        PLAIN = 0,
        SL = 1,
        INT_SL = 2,
        TREE = 3,
        AP = 4,
        LSH = 5,
        BLSH = 6

    };

    struct GlobalTopkTuneData { // stores sample results for a given bucket
        float lengthTime;
        std::vector<QueueElement> results;

        GlobalTopkTuneData() : lengthTime(0) {
        }
    };

    class Retriever;


    typedef boost::shared_ptr<Retriever> retriever_ptr;
    typedef std::vector< std::unordered_map< row_type, GlobalTopkTuneData > > Thread2Sample2Result;

    class ProbeBucket {
    public:
        void* ptrIndexes[NUM_INDEXES];
        std::pair<float, float> normL2, invNormL2; // min and max length information
        float bucketScanThreshold, runtime, t_b; // all theta_b < t_b do LENGTH

        col_type colNum, numLists;
        row_type startPos, endPos, rowNum;
        retriever_ptr ptrRetriever;

        row_type activeQueries; // active queries for this bucket. If multiple threads this will just be an estimation 
        xValues_ptr xValues; // data: theta_b(q) id: sampleId
        Thread2Sample2Result sampleThetas; // 1: thread 2: valid sample points for bucket -->result

        inline ProbeBucket() : numLists(1), t_b(1), runtime(0), activeQueries(0) {
            for (int i = 0; i < NUM_INDEXES; ++i) {
                ptrIndexes[i] = nullptr;
            }
        }

        inline ~ProbeBucket() {
            if (ptrIndexes[SL] != nullptr) {
                delete static_cast<QueueElementLists*> (ptrIndexes[SL]);
            }
            if (ptrIndexes[INT_SL] != nullptr) {
                delete static_cast<IntLists*> (ptrIndexes[INT_SL]);
            }
        }

        inline void init(const VectorMatrixLEMP& matrix, row_type startInd, row_type endInd, const LempArguments& args) {
            startPos = startInd;
            endPos = endInd;
            rowNum = endPos - startPos;
            normL2.second = matrix.getVectorLength(startPos);
            normL2.first = matrix.getVectorLength(endPos - 1);

            invNormL2.first = 1 / normL2.first;
            invNormL2.second = 1 / normL2.second;

            colNum = matrix.colNum;
        }

        inline void setAfterTuning(col_type lists, float thres) {          
            numLists = lists;
            t_b = thres;
        }

        inline bool hasIndex(Index_Type type) const {
            if (ptrIndexes[type] == nullptr)
                return false;
            else
                return true;
        }

        inline void* getIndex(Index_Type type) {
            return ptrIndexes[type];
        }

        bool isTunable(row_type availableQueries) {
            if (availableQueries < LOWER_LIMIT_PER_BUCKET * 3) {
                return false;
            } else {
                return true;
            }
        }

        inline void sampling(std::vector<RetrievalArguments>& retrArg) {
            xValues_ptr ptr(new std::vector<MatItem> ());
            xValues = ptr;
            row_type sampleSize;
            // find the queries

            // first find how many queries in total will be fired in this Above-theta problem
            std::vector<row_type> activeQueriesInPartition(retrArg.size());

            for (int t = 0; t < retrArg.size(); ++t) {
                std::vector<QueueElement>::const_iterator up = std::lower_bound(retrArg[t].queryMatrix->lengthInfo.begin(),
                        retrArg[t].queryMatrix->lengthInfo.end(), QueueElement(bucketScanThreshold, 0), std::greater<QueueElement>());
                activeQueriesInPartition[t] = up - retrArg[t].queryMatrix->lengthInfo.begin();

                activeQueries += activeQueriesInPartition[t];
            }

            
            // and based on the number of active queries pick up a good sample size
            if (activeQueries < LOWER_LIMIT_PER_BUCKET * 3) {
                sampleSize = 0;
            } else {
                sampleSize = 0.02 * activeQueries;

                if (sampleSize > UPPER_LIMIT_PER_BUCKET) {
                    sampleSize = UPPER_LIMIT_PER_BUCKET;
                }

                if (sampleSize < LOWER_LIMIT_PER_BUCKET) {// if very few elements qualify for this bucket, it does not pay off to tune.
                    sampleSize = LOWER_LIMIT_PER_BUCKET;
                }
            }

//            rg::Random32& random = retrArg[0].random;


            if (sampleSize > 0) {
                xValues->reserve(sampleSize);
                sampleSize /= retrArg.size();


                for (int t = 0; t < retrArg.size(); ++t) {
                    // do the actual sampling
//                    std::vector<row_type> sampleIndx = rg::sample(random, sampleSize, activeQueriesInPartition[t]);
                    std::vector<row_type> sample_vector(activeQueriesInPartition[t]);
                    std::vector<row_type> sampleIndx;
                    std::mt19937 g(123);
                    std::sample(sample_vector.begin(), sample_vector.end(), std::back_inserter(sampleIndx), sampleSize,
                                g);

                    // calculate the actual theta_b(q)) values
                    for (row_type i = 0; i < sampleIndx.size(); ++i) {
                        float theta_b_q = bucketScanThreshold / retrArg[t].queryMatrix->getVectorLength(sampleIndx[i]);
                        xValues->emplace_back(theta_b_q, t, sampleIndx[i]);
                    }
                }
                std::sort(xValues->begin(), xValues->end(), std::less<MatItem>());
            }
        }

        inline void setup_xValues_topk(const std::vector<RetrievalArguments>& retrArg, const Thread2Sample2Result& previousSampleThetas) {
            // for the Row-Top-k we already have the sample of queries, 
            // but the order of sample queries on the x-axis (theta_b(q)) is changing from bucket to bucket

            xValues_ptr ptr(new std::vector<MatItem>());
            xValues = ptr;

            for (int t = 0; t < retrArg.size(); ++t) {

                row_type lastInd = sampleThetas[t].size();
                auto it = sampleThetas[t].begin();

                while (lastInd > 0) {
                    float localTheta = previousSampleThetas[t].at(it->first).results.front().data;
                    localTheta *= (localTheta > 0 ? invNormL2.second : invNormL2.first);
                    xValues->emplace_back(localTheta, t, it->first);
                    lastInd--;
                    ++it;
                }
            }
            std::sort(xValues->begin(), xValues->end(), std::less<MatItem>());
        }





    };





}


#endif /* PROBEBUCKET_H_ */
