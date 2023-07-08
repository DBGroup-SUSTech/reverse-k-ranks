//
// Created by BianZheng on 2022/11/7.
//

#ifndef REVERSE_K_RANKS_NAMETRANSLATION_HPP
#define REVERSE_K_RANKS_NAMETRANSLATION_HPP

#include "../../../../../../usr/include/c++/9/string"
#include "../../../../../../usr/local/include/spdlog/spdlog.h"

namespace ReverseMIPS {
    std::string SampleMethodName(const std::string &method_name) {
        if (method_name == "QS" ||
            method_name == "QSRPUniformLP" ||
            method_name == "QSRPNormalLP" ||

            method_name == "QSRPRefineComputeIPBound" ||
            method_name == "QSRPRefineComputeAll" ||
            method_name == "QSRPRefineLEMP" ||

            method_name == "QSRPNormalLPUpdate" ||
            method_name == "QSUpdate") {
            return "OptimalPart";
        } else if (method_name == "QSMinusHeuristic") {
            return "OptimalAll";
        } else if (method_name == "QSRPUniformCandidateNormalLP") {
            return "OptimalUniform";
        } else if (method_name == "US") {
            return "Uniform";
        } else {
            spdlog::error("not find method name, program exit");
            exit(-1);
        }
    }

    std::string SampleSearchIndexName(const std::string &method_name) {
        if (method_name == "QS" ||
            method_name == "QSRPNormalLP" ||
            method_name == "QSRPUniformLP" ||

            method_name == "QSRPRefineComputeIPBound" ||
            method_name == "QSRPRefineComputeAll" ||
            method_name == "QSRPRefineLEMP" ||

            method_name == "QSRPNormalLPUpdate" ||
            method_name == "QSUpdate") {

            return "QS";
        } else if (method_name == "QSMinusHeuristic") {

            return "QSMinusHeuristic";
        } else if (method_name == "QSRPUniformCandidateNormalLP") {

            return "QSUniformCandidate";
        } else if (method_name == "US") {

            return "US";
        } else {
            spdlog::error("not find method name, program exit");
            exit(-1);
        }
    }

    std::string RegressionMethodName(const std::string &method_name) {
        if (method_name == "QSRPUniformLP") {
            return "UniformLinearRegressionLP";
        } else if (method_name == "QSRPNormalLP" || method_name == "QSRPUniformCandidateNormalLP" ||

                   method_name == "QSRPRefineComputeIPBound" ||
                   method_name == "QSRPRefineComputeAll" ||
                   method_name == "QSRPRefineLEMP" ||

                   method_name == "QSRPNormalLPUpdate") {
            return "NormalLinearRegressionLP";
        } else {
            spdlog::error("no such training method, program exit");
            exit(-1);
        }
    }

    std::string RegressionIndexName(const std::string &method_name) {
        if (method_name == "QS" ||
            method_name == "QSRPNormalLP" ||
            method_name == "QSRPUniformLP" ||

            method_name == "QSRPRefineComputeIPBound" ||
            method_name == "QSRPRefineComputeAll" ||
            method_name == "QSRPRefineLEMP" ||

            method_name == "QSRPNormalLPUpdate" ||
            method_name == "QSUpdate") {

            return "QS";
        } else if (method_name == "QSRPUniformCandidateNormalLP") {

            return "QSUniformCandidate";
        } else {
            spdlog::error("not find method name, program exit");
            exit(-1);
        }
    }


}
#endif //REVERSE_K_RANKS_NAMETRANSLATION_HPP
