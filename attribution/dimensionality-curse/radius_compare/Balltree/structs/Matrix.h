#ifndef MATRIX_H
#define MATRIX_H

#include "../util/Base.h"
#include "VectorElement.h"

class Matrix {

public:
    float *rawData;
    int rowNum;
    int colNum;

    inline Matrix() {
        this->rawData = NULL;
        this->rowNum = 0;
        this->colNum = 0;
    }

    inline ~Matrix() {
        if(rawData){
            delete[] rawData;
        }
    }

    inline float *getRowPtr(const int rowIndex) const {
        return &rawData[rowIndex * colNum];
    }

    inline void init(float *rawData, const int rowNum, const int colNum) {
        this->rowNum = rowNum;
        this->colNum = colNum;
        this->rawData = rawData;
    }

    inline void init(const int rowNum, const int colNum) {
        this->rowNum = rowNum;
        this->colNum = colNum;
        this->rawData = new float[rowNum * colNum];
    }

    void readData(string dataFilePath) {
        vector <string> lines;
        string line;

        ifstream fin(dataFilePath.c_str());

        int rowNum = 0;
        int colNum = 0;

        while (getline(fin, line)) {
            if (line.length() == 0) {
                continue;
            }
            lines.push_back(line);
        }

        if (lines.size() == 0) {
            return;
        }

        fin.close();

        stringstream test(lines[0]);
        float tempValue;
        while (test >> tempValue) {
            if (test.peek() == ',') {
                test.ignore();
                colNum++;
            }
        }
        colNum++;
        rowNum = lines.size();

        this->rowNum = rowNum;
        this->colNum = colNum;
        this->rawData = new float[rowNum * colNum];
        int colIndex = 0;

        for (int rowIndex = 0; rowIndex < lines.size(); rowIndex++) {
            stringstream ss(lines[rowIndex]);
            colIndex = 0;
            while (ss >> this->rawData[rowIndex * colNum + colIndex]) {
                if (ss.peek() == ',') {
                    ss.ignore();
                    colIndex++;
                }
            }
        }
    }
};

#endif //MATRIX_H