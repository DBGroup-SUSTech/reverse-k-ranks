# Introduction

- This repository provides an implementation of QSRP.
- This is a fast algorithm for the reverse $k$-ranks query in high dimensionality.

## Requirement

- Linux OS (Ubuntu).
  - The others have not been tested.
- `g++ 9.4.0` (or higher version) and `Openmp`.
- `openblas`, `eigen3` and `armadillo`

## How to build

```
cd reverse-k-ranks/
sh build-project.sh
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j 16
```

Then change the `index_dir` and `dataset_dir` to your directory

Note that we include a testing dataset in `reverse-k-ranks/dataset/`

Use `python3 run_rkr.py` to run the method

if you have cuda, you can turn on the option `use_cuda` in `CMakeLists.txt`
