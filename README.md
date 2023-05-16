# README

This is the code of "Handling Learnwares Developed from Heterogeneous Feature Spaces without Auxiliary Data", which generates our toy problem results desribed in Section 6.1 and real-world tasks reuslts described in Section 6.2.

## Requirements

Please ensure that some packages of the environment are configured as follows:

- python=3.10.4
- numpy=1.23.5
- scikit-learn=1.2.0

## Major Components

- core: contains the implementation of subspace learning, RKME generation and RKME match
- datasets: contains the implementation of the synthetic and real-world tasks generation.
- experiments: contains the implementation of results generation.

## Instructions

- run "experiments/toy_example.py" and get output figures in "experiments/figures".
- run "experiments/benchmark_test.py" to get results on benchmarks.