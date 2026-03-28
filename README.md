# 3D Poisson Solver

This project is prepared by Yanson Cheng.

This project implements a high performance solver for the 3D Poisson equation using the Jacobi Iterative method. The solver is developed in C++ and includes:

- Serial Implementation
- MPI Parallel Implementation
- Hybrid MPI + CUDA Implementation
- Automated Test Suite
- Doxygen Documentation

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Problem Description

We solve the Poisson equation on the unit cube:

∇²u(x, y, z) = f(x, y, z)

with Dirichlet boundary conditions:

u(x, y, z) = g(x, y, z) on ∂Ω

The domain is discretised using a structured grid and solved using:

- **Second-order central finite differences**
- **Jacobi iterative method**
- **L2 residual convergence criterion**

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Features

- Serial Solver ('poisson')
- MPI Parallel Solver with Domain Decomposition ('poisson-mpi')
- CUDA Accelerated Solver ('poisson-cuda')
- Verification Test Cases (analytical solutions)
- Additional Test Cases (Gaussian and Discontinuous Forcing)
- Strong and Weak Scaling Analysis (Report)
- Doxygen Documented Codebase

## Project Structure

├── src/ # Source code
│ ├── poisson_solver.cpp
│ ├── poisson_solver_mpi.cpp
│ ├── poisson_cuda.cu
│ ├── problem_setup.cpp
│ └── main*.cpp
│
├── tests/ # Test suite
│ ├── test_solver.cpp
│ └── test_solver_mpi.cpp
│
├── Makefile
├── Doxyfile
├── README.md

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Build Instructions

### Compile all targets

make

Available Executables:

1. poisson            -->    Serial Solver
2. poisson-mpi        -->    MPI Solver
3. poisson-cuda       -->    MPI + CUDA Solver
4. test_solver        -->    Serial Tests
5. test_solver_mpi    -->    MPI Tests

Usage:

Serial Solver:
./poisson --help

Example: 
./poisson --test 1 --Nx 32 --Ny 32 --Nz 32

MPI Solver:
mpirun -np 8 ./poisson-mpi --test 2 --Nx 64 --Ny 64 --Nz 64

You can also optionally specify decomposition 
--Px 2 --Py 2 --Pz 2 under MPI Solver.

However, Px x Py x Pz MUST EQUAL number of MPI ranks

CUDA Solver:
mpirun -np 4 ./poisson-cuda --test 1

Running Tests:

Serial: 
make tests

MPI:
make tests-mpi

Output: 
The solver produces solution.txt --> converged numerical solution
Terminal output: 
- Final Residual
- Error (for verification cases)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Performance Analysis

The project includes:

Strong scaling (fixed problem size)
Weak scaling (increasing problem size)
MPI vs CUDA performance comparison

Key observations:

MPI shows good strong scaling but reduced efficiency at high ranks
Weak scaling is limited by communication overhead
CUDA provides speed-up at low ranks but degrades at high ranks due to:
small local domains
host-device transfer overhead
MPI communication bottlenecks

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Documentation

To generate Doxygen documentation:
make doc

This opens in docs/html/index.html

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Parallelisation Approach

- 3D domain decomposition across MPI ranks
- Halo (ghost cell) exchange in x, y, z directions
- Non-blocking MPI communication
- Global residual computed using MPI reduction

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## CUDA Approach

Offloads Jacobi update and residual computation to GPU
MPI handles inter-process communication
Hybrid model: MPI (distributed) + CUDA (local acceleration)

Limitations observed:

GPU underutilisation at high MPI ranks
Increased communication overhead
Host-device transfer costs

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
