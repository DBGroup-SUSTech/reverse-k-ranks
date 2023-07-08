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
 * File:   Lists2.h
 * Author: chteflio
 *
 * Created on January 5, 2015, 3:59 PM
 */

#ifndef LISTS_H
#define	LISTS_H


namespace mips {

    class Index {
    protected:
//        omp_lock_t writelock;
        bool initialized;
    public:

        inline Index() : initialized(false) {
//            omp_init_lock(&writelock);
        }

        inline ~Index() {
//            omp_destroy_lock(&writelock);
        }

        inline bool isInitialized() const {
            return initialized;
        }

        inline void lockIndex() {
//            omp_set_lock(&writelock);
        }

        inline void unlockIndex() {
//            omp_unset_lock(&writelock);
        }

    };

    class QueueElementLists : public Index {
        std::vector<QueueElement> sortedCoord;
        col_type colNum;
        row_type size;

        /*
         * returns true if there is possibility for sufficient, otherwise false
         */
        inline bool getBounds(float qi, float theta, col_type col, std::pair<row_type, row_type>& necessaryIndices) const {

            bool suff = false;
            float base, x, root1, root2;
            std::pair<float, float> necessaryValues;
            std::vector<QueueElement>::const_iterator it;

            // get bounds in the form of values
            base = theta * qi;
            x = sqrt((theta * theta - 1) * (qi * qi - 1));

            root1 = base + x;
            root2 = base - x;

            necessaryValues.first = root2;
            necessaryValues.second = root1;

            row_type start = col * size;
            row_type end = (col + 1) * size;

            float y = theta / qi;


            if (qi > 0 && necessaryValues.second >= y) {
                necessaryValues.second = 1;
                suff = true;

                if (necessaryValues.first > y) {
                    necessaryValues.first = y;
                }

            } else if (qi < 0 && necessaryValues.first <= y) {
                necessaryValues.first = -1;
                suff = true;

                if (necessaryValues.second < y) {
                    necessaryValues.second = y;
                }
            }

            if (necessaryValues.first <= sortedCoord[col * size].data) {
                necessaryIndices.first = start;
            } else {

                it = std::lower_bound(sortedCoord.begin() + start, sortedCoord.begin() + end, QueueElement(necessaryValues.first, 0));
                necessaryIndices.first = (it - sortedCoord.begin());
            }

            if (necessaryValues.second > sortedCoord[(col + 1) * size - 1].data) {
                necessaryIndices.second = end;
            } else {
                it = std::upper_bound(sortedCoord.begin() + start, sortedCoord.begin() + end, QueueElement(necessaryValues.second, 0));
                necessaryIndices.second = (it - sortedCoord.begin());
            }

            return suff;
        }

    public:

        inline QueueElementLists() = default;
        inline ~QueueElementLists() = default;

        inline void initializeLists(const VectorMatrixLEMP& matrix, ta_size_type start = 0, ta_size_type end = 0) {

//            omp_set_lock(&writelock);

            if (!initialized) {

                colNum = matrix.colNum;

                if (start == end) {
                    start = 0;
                    end = matrix.rowNum;
                }
                size = end - start;
                sortedCoord.reserve(colNum * size);

                for (col_type j = 0; j < colNum; ++j) {
                    for (row_type i = start; i < end; ++i) { // scans the matrix as it is, i.e., perhaps in sorted order
                        sortedCoord.emplace_back(matrix.getMatrixRowPtr(i)[j], i - start);
                        // QueueElement.id is the position of the vector in the matrix, not necessarily the vectorID
                    }
                    std::sort(sortedCoord.begin()+(j * size), sortedCoord.end(), std::less<QueueElement>());
                }
                initialized = true;
            }
//            omp_unset_lock(&writelock);
        }

        inline row_type getRowPointer(row_type row, col_type col) const {
            return sortedCoord[col * size + row].id;
        }

        inline QueueElement* getElement(row_type pos) {
            return &sortedCoord[pos];
        }

        inline float getValue(row_type row, col_type col) const {
            return sortedCoord[col * size + row].data;
        }

        inline col_type getColNum() const {
            return colNum;
        }

        inline row_type getRowNum() const {
            return size;
        }

        inline bool calculateIntervals(const float* query, const col_type* listsQueue, std::vector<IntervalElement>& intervals,
                float localTheta, col_type lists) const {

            std::pair<row_type, row_type> necessaryIndices;

            for (col_type i = 0; i < lists; ++i) {

                getBounds(query[listsQueue[i]], localTheta, listsQueue[i], necessaryIndices);

                intervals[i].col = listsQueue[i];
                intervals[i].start = necessaryIndices.first;
                intervals[i].end = necessaryIndices.second;

                if (intervals[i].end <= intervals[i].start) {
                    return false;
                }
            }

//            std::sort(intervals.begin(), intervals.begin() + lists);
            return true;
        }


    };

    // contains the sorted lists
    // 1st Dimension: coordinates  2nd Dimension: rows (row pointers to the NormMatrix)

    class IntLists : public Index {
        std::vector<float> values;
        std::vector<row_type> ids;
        col_type colNum;
        row_type size;

        inline void getBounds(float qi, float theta, col_type col, std::pair<row_type, row_type>& necessaryIndices) const {

            float base, x, root1, root2;
            std::pair<float, float> necessaryValues;
            std::vector<float>::const_iterator it;

            // get bounds in the form of values
            base = theta * qi;
            x = sqrt((theta * theta - 1) * (qi * qi - 1));

            root1 = base + x;
            root2 = base - x;

            necessaryValues.first = root2;
            necessaryValues.second = root1;

            row_type start = col * size;
            row_type end = (col + 1) * size;

            float y = theta / qi;


            if (qi > 0) {
                necessaryValues.second = (necessaryValues.second >= y ? 1 : necessaryValues.second);

                if (necessaryValues.first > y) {
                    necessaryValues.first = y;
                }

            } else if (qi < 0) {
                necessaryValues.first = (necessaryValues.first <= y ? -1 : necessaryValues.first);

                if (necessaryValues.second < y) {
                    necessaryValues.second = y;
                }
            }

            if (necessaryValues.first <= values[start]) {
                necessaryIndices.first = start;
            } else {
                it = std::lower_bound(values.begin() + start, values.begin() + end, necessaryValues.first);
                necessaryIndices.first = it - values.begin();
            }

            if (necessaryValues.second > values[end - 1]) {
                necessaryIndices.second = end;
            } else {
                it = std::upper_bound(values.begin() + start, values.begin() + end, necessaryValues.second);
                necessaryIndices.second = it - values.begin();
            }


        }

    public:


        inline IntLists() = default;
        inline ~IntLists() = default;

        inline void initializeLists(const VectorMatrixLEMP& matrix, ta_size_type start = 0, ta_size_type end = 0) {
//            omp_set_lock(&writelock);
            if (!initialized) {
                std::vector<std::vector<QueueElement> > sortedCoord;

                sortedCoord.clear();
                sortedCoord.resize(matrix.colNum);

                colNum = matrix.colNum;

                if (start == end) {
                    start = 0;
                    end = matrix.rowNum;
                }
                size = end - start;

                ids.reserve(colNum * size);
                values.reserve(colNum * size);

                for (col_type i = 0; i < colNum; ++i) {

                    for (row_type j = start; j < end; ++j) { // scans the matrix as it is, i.e., perhaps in sorted order
                        //                        sortedCoord[i].push_back(QueueElement(matrix.getMatrixRowPtr(j)[i], j - start));
                        sortedCoord[i].emplace_back(matrix.getMatrixRowPtr(j)[i], j - start);
                        // i is the position of the vector in the matrix, not necessarily the vectorID
                    }

                    std::sort(sortedCoord[i].begin(), sortedCoord[i].end(), std::less<QueueElement>());

                    for (row_type j = 0; j < sortedCoord[i].size(); ++j) {
                        ids.push_back(sortedCoord[i][j].id);
                        values.push_back(sortedCoord[i][j].data);
                    }
                }
                initialized = true;
            }
//            omp_unset_lock(&writelock);
        }

        inline row_type getRowPointer(row_type row, col_type col) const {
            return ids[col * size + row];
        }

        inline row_type* getElement(row_type pos) {
            return &ids[pos];
        }

        inline float getValue(row_type row, col_type col) const {
            return values[col * size + row];
        }

        inline col_type getColNum() const {
            return colNum;
        }

        inline row_type getRowNum() const {
            return size;
        }

        inline bool calculateIntervals(const float* query, const col_type* listsQueue, std::vector<IntervalElement>& intervals,
                float localTheta, col_type lists) const {

            std::pair<row_type, row_type> necessaryIndices;

            for (col_type i = 0; i < lists; ++i) {

                getBounds(query[listsQueue[i]], localTheta, listsQueue[i], necessaryIndices);

                intervals[i].col = listsQueue[i];
                intervals[i].start = necessaryIndices.first;
                intervals[i].end = necessaryIndices.second;

                if (intervals[i].end <= intervals[i].start) {
                    return false;
                }
            }

            std::sort(intervals.begin(), intervals.begin() + lists);

            return true;
        }

    };

}
#endif	/* LISTS_H */

