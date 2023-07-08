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

#ifndef TA_TALIB_H
#define TA_TALIB_H


#include <omp.h>

#include <random>

#include <mips/structs/Definitions.h>
#include <mips/structs/BasicStructs.h>
#include <mips/structs/Args.h>


#include <mips/structs/VectorMatrixLEMP.h>
#include <mips/structs/Results.h>
#include <mips/structs/Output.h>
#include <mips/structs/Lists.h>

///////////////////////////////////////////////////////////////////
/////////// l2ap stuff ///////////////
#include <mips/ap/includes.h>
/////////////////////////////////////

#include <mips/structs/TimeRecord.hpp>
#include <mips/structs/Candidates.h>
#include <mips/structs/TAState.h>
#include <mips/structs/TANRAState.h>
#include <mips/structs/RetrievalArguments.h>//////////////////////
#include <mips/structs/QueryBatch.h>
#include <mips/structs/ProbeBucket.h>
#include <mips/structs/Bucketize.h>
#include <mips/structs/CandidateVerification.h>

#include <mips/retrieval/Retriever.h>
#include <mips/retrieval/ListsTuneData.h>
#include <mips/retrieval/TuneTopk.h>
#include <mips/retrieval/coord.h>
#include <mips/retrieval/icoord.h>
#include <mips/retrieval/mixed.h>

#include <mips/algos/Mip.h>
#include <mips/algos/Lemp.h>


#endif
