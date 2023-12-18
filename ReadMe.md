# Introduction

- This repository provides the implementation of the paper QSRP:  Efficient Reverse $k$-Ranks Query Processing on High-dimensional Embeddings.
- This is a fast algorithm for the reverse $k$-ranks query in high dimensionality.

## Requirement

- Linux OS (Ubuntu).
  - The others have not been tested.
- `g++ 9.4.0` (or higher version) and `Openmp`.
- `openblas`, `eigen3`, `armadillo`, `boost-1.80` and `spdlog`
  - For Ubuntu user, please see the comments in `build-project.sh` for installation

- CUDA (optional)
  - You can set `USE_CUDA` as OFF in CMakeLists.txt to disable CUDA


## How to build and run

```
cd reverse-k-ranks/
sh build-project.sh
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j 16
# In run_rkr.py, change the `index_dir` and `dataset_dir` to your directory
python3 run_rkr.py
```

The code runs the dataset included in `reverse-k-ranks/dataset/`

## Method details

In `run_rkr.py`:

`GridIndex`: the Grid method shown in the framework 

`Rtree`: the MPA method in the paper

`QS`: QSRP without regression-based pruning

`QSRPNormalLP`: the QSRP method

`QSRPUniformLP`: the QSRP-DT method, see Figure 17 in the paper

`US`: Uniform Sample, the sampling-based baseline solution  



In `run_rkr_update.py`

`QSRPNormalLPUpdate` is the QSRP update method shown in the paper

`QSUpdate` is the update version of QS 

## Datasets

We have used LIBPMF (https://www.cs.utexas.edu/~rofuyu/libpmf/) as the matrix factorization model to generate the embedding. 

To download the dataset, you can refer to https://drive.google.com/drive/folders/1UloY14usLHJxSGVdoharyUUPc7nAcPdl?usp=sharing 

