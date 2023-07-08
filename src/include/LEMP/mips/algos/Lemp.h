
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
 * algo_with_tuning2.h
 *
 *  Created on: Mar 25, 2014
 *      Author: chteflio
 */

#ifndef LEMP_H_
#define LEMP_H_


namespace mips {


    // I can change between runs: theta, method, queryMatrix
    // I cannot change between runs: probeMatrix, k

    class Lemp : public Mip {
        std::vector<VectorMatrixLEMP> queryMatrices;
        std::vector<float> cweights; //for L2AP

        std::vector<ProbeBucket> probeBuckets;
        std::vector<RetrievalArguments> retrArg;

        row_type maxProbeBucketSize;
        LempArguments args;
        row_type activeBuckets;

        inline row_type initProbeBuckets(VectorMatrixLEMP &rightMatrix);

        inline void initializeRetrievers();

        inline void
        initQueryBatches(VectorMatrixLEMP &leftMatrix, row_type maxBlockSize, std::vector<RetrievalArguments> &retrArg);

        inline void initListsInBuckets();

        inline void tune(std::vector<RetrievalArguments> &retrArg, row_type allQueries, const int &max_rank);

        inline void printAlgoName(const VectorMatrixLEMP &queryMatrix);

    public:

        inline void setTheta(float theta) {
            args.theta = theta;
        }

        inline void setMethod(LEMP_Method method) {
            args.method = method;
        }

        inline Lemp() = default;

        inline Lemp(InputArguments &in, int cacheSizeinKB, LEMP_Method method, bool isTARR, float R, float epsilon) :
                maxProbeBucketSize(0) {
            args.copyInputArguments(in);
            args.cacheSizeinKB = cacheSizeinKB;
            args.method = method;
            args.isTARR = isTARR;
            args.R = R;
            args.epsilon = epsilon;

//            logging.open(args.logFile.c_str(), std::ios_base::app);
//
//            if (!logging.is_open()) {
//                std::cout << "[WARNING] No log will be created!" << std::endl;
//            } else {
//                std::cout << "[INFO] Logging in " << args.logFile << std::endl;
//            }

//            omp_set_num_threads(args.threads);
//            retrArg.resize(args.threads);
            retrArg.resize(1);
            queryMatrices.resize(1);
        }

        inline ~Lemp() {
//            logging.close();
        }

        inline void initialize(VectorMatrixLEMP &rightMatrix) {
//            std::cout << "[INIT] ProbeMatrix contains " << rightMatrix.rowNum << " vectors with dimensionality "
//                      << (0 + rightMatrix.colNum) << std::endl;
//            TimeRecord record;
//            record.reset();
//            timer.start();
            maxProbeBucketSize = initProbeBuckets(rightMatrix);
//            timer.stop();
//            dataPreprocessingTimeRight += timer.elapsedTime().nanos();
//            dataPreprocessingTimeRight += record.get_elapsed_time_second() * 1e9;

            switch (args.method) {
                case LEMP_LI:
//#pragma omp parallel for default(none) shared(b0) schedule(static, 1) num_threads(args.threads)
                    for (row_type b = 0; b < probeBuckets.size(); ++b) {
                        retriever_ptr rPtr(new LX_Retriever<IncrRetriever>());
                        probeBuckets[b].ptrRetriever = rPtr;
//                        if (probeBuckets[b].ptrIndexes[SL] == 0) {
//                            probeBuckets[b].ptrIndexes[SL] = new QueueElementLists();
//                        }
                    }
                    break;

            }

        }

        inline void runAboveTheta(const int &max_rank, VectorMatrixLEMP &leftMatrix, size_t &n_compare, int &n_result,
                                  double &initQueryBatches_time, double &initializeRetrievers_time,
                                  double &initListsInBuckets_time, double &tune_time, double &run_time
        ) {
//            printAlgoName(leftMatrix);

//            TimeRecord record;
//            record.reset();
//            timer.start();
            //            std::vector<RetrievalArguments> retrArg; //one argument for each thread
            TimeRecord record;
            record.reset();
            initQueryBatches(leftMatrix, maxProbeBucketSize, retrArg);
            initQueryBatches_time += record.get_elapsed_time_second();
            record.reset();
            initializeRetrievers();
            initializeRetrievers_time += record.get_elapsed_time_second();
//            results.resultsVector.resize(args.threads);

//            for (auto &argument: retrArg)
//                argument.init(maxProbeBucketSize);
            retrArg[0].init(maxProbeBucketSize);

//            timer.stop();
//            dataPreprocessingTimeLeft += timer.elapsedTime().nanos();
//            dataPreprocessingTimeLeft += record.get_elapsed_time_second() * 1e9;

//            timer.start();
//            record.reset();
            record.reset();
//            initListsInBuckets();
            initListsInBuckets_time += record.get_elapsed_time_second();
//            timer.stop();
//            dataPreprocessingTimeRight += timer.elapsedTime().nanos();
//            dataPreprocessingTimeRight += record.get_elapsed_time_second() * 1e9;

            record.reset();
            tune(retrArg, leftMatrix.rowNum, max_rank);
            tune_time += record.get_elapsed_time_second();

//            std::cout << "[RETRIEVAL] Retrieval (theta = " << args.theta << ") starts ..." << std::endl;
//            logging << "theta(" << args.theta << ")\t";

//            timer.start();
//            record.reset();

//            col_type maxLists = 1;
//
//
//            switch (args.method) {
//                case LEMP_I:
//                case LEMP_LI:
//                case LEMP_C:
//                case LEMP_LC:
//
//                    std::for_each(probeBuckets.begin(), probeBuckets.begin() + activeBuckets,
//                                  [&maxLists](const ProbeBucket &b) {
//                                      if (maxLists < b.numLists) maxLists = b.numLists;
//                                  });
//
//                    for (auto &argument: retrArg)
//                        argument.setIntervals(maxLists);
//
//
//                    break;
//            }
//
//            for (auto &argument: retrArg)
//                argument.clear();

            retrArg[0].setIntervals(1);
            retrArg[0].clear();

            comp_type comparisons = 0;


            n_result = 0;
//#pragma omp parallel default(none) shared(results) reduction(+ : comparisons)
            {
//                row_type tid = omp_get_thread_num();
                row_type tid = 0;

                record.reset();
                for (row_type b = 0; b < activeBuckets; ++b) {
                    probeBuckets[b].ptrRetriever->run(probeBuckets[b], &retrArg[tid], max_rank);
                }
                run_time += record.get_elapsed_time_second();
                comparisons += retrArg[tid].comparisons;
//                results.moveAppend(retrArg[tid].results, tid);
                n_result += retrArg[tid].n_results;
            }


//            int totalSize = results.getResultSize();

//            timer.stop();
//            retrievalTime += timer.elapsedTime().nanos();
//            retrievalTime += record.get_elapsed_time_second() * 1e9;
//            totalComparisons += comparisons;
            n_compare = comparisons;

//            std::cout << "[RETRIEVAL] ... and is finished with " << totalSize << " results" << std::endl;
//            logging << totalSize << "\t";

//            outputStats();
        }

    };

    inline row_type Lemp::initProbeBuckets(VectorMatrixLEMP &rightMatrix) {
        std::vector<row_type> probeBucketOffsets;

        probeMatrix.init(rightMatrix, true, false); // normalize and sort

        row_type maxBlockSize = computeBlockOffsetsByFactorCacheFittingForItems(probeMatrix.lengthInfo,
                                                                                probeMatrix.rowNum, probeBucketOffsets,
                                                                                FACTOR, ITEMS_PER_BLOCK,
                                                                                args.cacheSizeinKB, probeMatrix.colNum,
                                                                                args);

        bucketize(probeBuckets, probeMatrix, probeBucketOffsets, args);

//        std::cout << "[INIT] ProbeBuckets = " << probeBucketOffsets.size() << std::endl;
        return maxBlockSize;
    }

    inline void Lemp::initializeRetrievers() {

        activeBuckets = probeBuckets.size();
        maxProbeBucketSize = 0;

        if (args.k == 0) { // Above-theta
            float maxUserLength = 0;

            std::for_each(queryMatrices.begin(), queryMatrices.end(),
                          [&maxUserLength](const VectorMatrixLEMP &m) {
                              if (maxUserLength < m.lengthInfo[0].data) maxUserLength = m.lengthInfo[0].data;
                          });


            for (row_type i = 0; i < probeBuckets.size(); ++i) {
                probeBuckets[i].bucketScanThreshold = args.theta * probeBuckets[i].invNormL2.second;

                // find maxProbeBucketLength
                if (maxUserLength > probeBuckets[i].bucketScanThreshold) {
                    if (maxProbeBucketSize < probeBuckets[i].endPos - probeBuckets[i].startPos)
                        maxProbeBucketSize = probeBuckets[i].endPos - probeBuckets[i].startPos;
                } else {
                    activeBuckets = i;
                    break;
                }
            }
        }

    }

    inline void Lemp::initQueryBatches(VectorMatrixLEMP &leftMatrix, row_type maxBlockSize,
                                       std::vector<RetrievalArguments> &retrArg) {

//        std::cout << "[RETRIEVAL] QueryMatrix contains " << leftMatrix.rowNum << " vectors with dimensionality "
//                  << (0 + leftMatrix.colNum) << std::endl;


//        row_type nCount = 0;
        row_type myNumThreads = args.threads;

        if (leftMatrix.rowNum < args.threads) {
            myNumThreads = leftMatrix.rowNum;
//            std::cout << "[WARNING] Query matrix contains too few elements. Suboptimal running with " << myNumThreads
//                      << " thread(s)" << std::endl;
        }
//        omp_set_num_threads(myNumThreads);
//        queryMatrices.resize(myNumThreads);
//        retrArg.resize(myNumThreads);

        if (args.k > 0) { // this is a top-k version
            initializeMatrices(leftMatrix, queryMatrices, false, true, args.epsilon); // normalize but don't sort
        } else {
            initializeMatrices(leftMatrix, queryMatrices, true, false); // normalize and sort
        }

//#pragma omp parallel default(none) shared(maxBlockSize, retrArg, myNumThreads) reduction(+ : nCount)
        {

//            row_type tid = omp_get_thread_num();
            row_type tid = 0;
            std::vector<row_type> blockOffsets;
            computeBlockOffsetsForUsersFixed(queryMatrices[tid].rowNum, blockOffsets, args.cacheSizeinKB,
                                             queryMatrices[tid].colNum, args, maxBlockSize);
            bucketize(retrArg[tid].queryBatches, queryMatrices[tid], blockOffsets, args);
//            nCount += retrArg[tid].queryBatches.size();
            retrArg[tid].initializeBasics(queryMatrices[tid], probeMatrix, args.method, args.theta, args.k,
                                          myNumThreads, args.R, args.epsilon, args.numTrees, args.search_k, true,
                                          args.isTARR);

        }

//        std::cout << "[RETRIEVAL] QueryBatches = " << nCount << std::endl;

    }

    inline void Lemp::initListsInBuckets() {

        float maxQueryLength = 0;
        // in the case of topk the tuning part has modified the activeBuckets
        row_type b0 = (args.k == 0 ? 0 : 1);

//        std::cout << "[RETRIEVAL] ProbeBuckets (active) = " << activeBuckets << std::endl;

        float worstCaseTheta;

        switch (args.method) {
            case LEMP_LI:
            case LEMP_I:
//#pragma omp parallel for schedule(dynamic, 1)
                for (row_type b = b0; b < activeBuckets; ++b) {
                    static_cast<QueueElementLists *> (probeBuckets[b].ptrIndexes[SL])->initializeLists(probeMatrix,
                                                                                                       probeBuckets[b].startPos,
                                                                                                       probeBuckets[b].endPos);
                }
                break;

        }
//         std::cout << "Done creating lists" << std::endl;
    }

    inline void Lemp::tune(std::vector<RetrievalArguments> &retrArg, row_type allQueries, const int &max_rank) {

        if (activeBuckets > 0) {
            switch (args.method) {
                case LEMP_LI:
                case LEMP_I:
                case LEMP_LC:
                case LEMP_C:

                    if (probeBuckets[0].isTunable(allQueries)) {
                        if (args.k == 0) {
//                            TimeRecord record;
//                            record.reset();
//                            timer.start();
                            // first set-up the xValues in each retriever
                            for (row_type b = 0; b < activeBuckets; ++b) {
                                probeBuckets[b].sampling(retrArg);
                            }


                            // then do the actual tuning
                            for (row_type b = 0; b < activeBuckets; ++b) {
                                probeBuckets[b].ptrRetriever->tune(probeBuckets[b],
                                                                   (b == 0 ? probeBuckets[b] : probeBuckets[b - 1]),
                                                                   retrArg, max_rank);
                            }

//                            timer.stop();
//                            tuningTime += timer.elapsedTime().nanos();
//                            tuningTime += record.get_elapsed_time_second() * 1e9;

                        }
                    } else {
//                        std::cout << "[WARNING] Too few queries (" << allQueries << ") for tuning (at least "
//                                  << LOWER_LIMIT_PER_BUCKET * 3 << " needed)" << std::endl;
//                        std::cout
//                                << "[WARNING] Using default (t_b=1, lists=1) or previous tuning values for all probe buckets "
//                                << std::endl;
//                        std::cout
//                                << "[WARNING] You can either reduce LOWER_LIMIT_PER_BUCKET and recompile or use larger batches of queries"
//                                << std::endl;
                    }


                    break;
            }
        }


    }

    inline void Lemp::printAlgoName(const VectorMatrixLEMP &queryMatrix) {
//        switch (args.method) {
//            case LEMP_LI:
//                logging << "LEMP_LI" << "\t" << args.threads << "\t";
//                std::cout << "[ALGORITHM] LEMP_LI with " << args.threads << " thread(s)" << std::endl;
//                break;
//            case LEMP_I:
//                logging << "LEMP_I" << "\t" << args.threads << "\t";
//                std::cout << "[ALGORITHM] LEMP_I with " << args.threads << " thread(s)" << std::endl;
//                break;
//            case LEMP_LC:
//                logging << "LEMP_LC" << "\t" << args.threads << "\t";
//                std::cout << "[ALGORITHM] LEMP_LC with " << args.threads << " thread(s)" << std::endl;
//                break;
//            case LEMP_C:
//                logging << "LEMP_C" << "\t" << args.threads << "\t";
//                std::cout << "[ALGORITHM] LEMP_C with " << args.threads << " thread(s)" << std::endl;
//                break;
//
//        }
//
//        logging << "P(" << probeMatrix.rowNum << "x" << (0 + probeMatrix.colNum) << ")\t";
//        logging << "Q^T(" << queryMatrix.rowNum << "x" << (0 + queryMatrix.colNum) << ")\t";
    }


}


#endif /* ALGO_WITH_TUNING_H_ */
