//
// Created by bianzheng on 2023/4/25.
//
#include "util/Base.h"
#include "util/Conf.h"
#include "structs/Matrix.h"
#include "util/Logger.h"
#include "alg/tree/BallTreeSearch.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

void basicLog(const Matrix &q, const Matrix &p, const int k) {
    Logger::Log("q path: " + to_string(Conf::qDataPath));
    Logger::Log("p path: " + to_string(Conf::pDataPath));
    Logger::Log("q: " + to_string(q.rowNum) + "," + to_string(q.colNum));
    Logger::Log("p: " + to_string(p.rowNum) + "," + to_string(p.colNum));
    Logger::Log("Algorithm: " + Conf::algName);
    Logger::Log("k: " + to_string(k));
}

int main(int argc, char **argv) {

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("alg", po::value<string>(&(Conf::algName))->default_value("naive"), "Algorithm")
            ("k", po::value<int>(&(Conf::k))->default_value(1), "K")
            ("dataset", po::value<string>(&(Conf::dataset)), "name of dataset for log output")
            ("q", po::value<string>(&(Conf::qDataPath)), "file path of q Data")
            ("p", po::value<string>(&(Conf::pDataPath)), "file path of p Data")
            ("scalingValue", po::value<int>(&(Conf::scalingValue))->default_value(127), "maximum value for scaling")
            ("log", po::value<bool>(&(Conf::log))->default_value(true), "whether it outputs log")
            ("logPathPrefix", po::value<string>(&(Conf::logPathPrefix))->default_value("./log"),
             "output path of log file (Prefix)")
            ("outputResult", po::value<bool>(&(Conf::outputResult))->default_value(true), "whether it outputs results")
            ("resultPathPrefix", po::value<string>(&(Conf::resultPathPrefix))->default_value("./result"),
             "output path of result file (Prefix)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

//    if (vm.count("help")) {
//        cout << desc << "\n";
//        return 0;
//    } else if (Conf::qDataPath == "" || Conf::pDataPath == "") {
//        cout << "Please specify path to data files" << endl << endl;
//        cout << desc << endl;
//        return 0;
//    }
    Conf::qDataPath = "/home/bianzheng/reverse-k-ranks/attribution/dimensionality-curse/data/q.txt";
    Conf::pDataPath = "/home/bianzheng/reverse-k-ranks/attribution/dimensionality-curse/data/p.txt";
    Conf::dataset = "MovieLens";
    Conf::k = 10;
//    Conf::SIGMA = 0.7;
    Conf::algName = "BallTree";

    Conf::Output();

    Matrix q;
    Matrix p;
    q.readData(Conf::qDataPath);
    p.readData(Conf::pDataPath);
    Conf::dimension = p.colNum;

    cout << "-----------------------" << endl;

    // ToDo: replace the old name (FEIPR) with FEXIPRO

    if (Conf::algName == "BallTree") {

        string logFileName =
                Conf::logPathPrefix + "-" + Conf::dataset + "-" + Conf::algName + "-" + to_string(Conf::k) + ".txt";
        Logger::open(logFileName);
        basicLog(q, p, Conf::k);

        ballTreeTopK(Conf::k, q, p);

    } else {
        cout << "unrecognized method" << endl;
    }

    return 0;
}
