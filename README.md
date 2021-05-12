# MatrixPerspective
This supplement is for reproducing the computational results in section 5 of the paper "A New Perspective on Low-Rank Optimization" by Dimitris Bertsimas, Ryan Cory-Wright and Jean Pauphilet. Note that a preprint is available [here](TBD).

# Introduction
The software in this package is designed to provide certifiably near-optimal solutions to low-rank problems using convex relaxations of low-rank constraints. 
The algorithms implemented here are described in the paper "A New Perspective on Low-Rank Optimization" by Bertsimas, Cory-Wright and Pauphilet. For the user's convenience, we have also included files from the MATLAB package CVXQuad which we used to generate the D-optimal design experimental results.

## Setup and Installation
In order to run this software, you must install a recent version of Julia from http://julialang.org/downloads/. The most recent version of Julia at the time this code was last tested was Julia 1.5.3; the code should work on any version of Julia beginning in 1.x

You must also have a valid installation of Mosek (>=9.0) for this software to run (academic licenses are freely available at https://www.mosek.com/products/academic-licenses/). This software was tested on Mosek version 9.1, but should also work on more recent versions of these solvers.

A number of packages must be installed in Julia before the code can be run. They can be added by running:

```
using Pkg; Pkg.add("JuMP, Mosek, MosekTools, Random, LinearAlgebra, DataFrames, Test, Suppressor, DelimitedFiles, CSV, StatsBase, Compat")
```

You will also need to ensure that the Julia packages are of the correct version, to guarentee that the code will run (any configuration where the JuMP version is >=0.19 "should" work, but using a different configuration is at your own risk). The versions of the packages which we benchmarked the code on are:

```
- JuMP  0.21.3
- Gurobi 0.9.3
- MathOptInterface 0.9.19
- Mosek 1.13
- Ipopt 0.6.5
- MosekTools 0.9.4
- StatsBase 0.33.2
- Suppressor 0.2.0
```

To run the D-optimal design experiments, you will need to install Matlab, CVX and CVXQuad, and run the code in Matlab. 

## Thank you

Thank you for your interest in MatrixPerspectiveSoftware. Please let us know if you encounter any issues using this code, or have comments or questions.  Feel free to email us anytime.

Dimitris Bertsimas
dbertsim@mit.edu

Ryan Cory-Wright
ryancw@mit.edu

Jean Pauphilet
jpauphilet@london.edu
