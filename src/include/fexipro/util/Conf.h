#ifndef CONF_H
#define CONF_H

#include "Base.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/algorithm/string.hpp>
using namespace boost::property_tree;
using namespace boost;

namespace Conf {
    //
    // Variables
    //
    bool log;
    bool outputResult;
    string qDataPath;
    string pDataPath;
    string algName;
    string dataset;
    float SIGMA;
    int k;
    string resultPathPrefix;
    string logPathPrefix;
    int dimension;
    int scalingValue;

//    template<class T>
//    static string PrintVec(vector <T> vec) {
//        string print = "";
//        for (auto val = vec.begin(); val != vec.end(); ++val) {
//            print = print + " " + to_string(*val);
//        }
//        return print;
//    }

    void Output(ostream &out = cout) {
        out << "--------- Read Config ----------" << endl;
        out << "bool log = " << log << endl;
        out << "bool outputResult = " << outputResult << endl;
        out << "string dataset = " << dataset << endl;
        out << "string qDataPath = " << qDataPath << endl;
        out << "string pDataPath = " << pDataPath << endl;
        out << "string algName = " << algName << endl;
        out << "k = " << k << endl;
        out << "scalingValue = " << scalingValue << endl;
        out << "SIGMA = " << SIGMA << endl;
        out << "string resultPath = " << resultPathPrefix << endl;
        out << "string logPath = " << logPathPrefix << endl;
        out << "--------------------------------" << endl;
    }

}

#endif //CONF_H