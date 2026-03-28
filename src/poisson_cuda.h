#pragma once

#include "poisson_solver_mpi.h"

/******************************************************************************************
 * @brief Solve the 3D Poisson equation using a CUDA-accelerated Jacobi method.
 *
 * This function performs an iterative solution of the Poisson equation on a
 * distributed domain. Each MPI rank operates on its local subdomain, while
 * the computationally intensive stencil updates are offloaded to the GPU.
 *
 * The solver uses Jacobi iteration and stops when either:
 * - the global residual falls below the specified tolerance (`epsilon`), or
 * - a maximum number of iterations is reached internally.
 *
 * @param data Local problem data structure containing:
 *             - grid dimensions (global and local),
 *             - solution arrays (`u`, `u_new`),
 *             - forcing term (`f`),
 *             - MPI communicator and neighbour information.
 * @param epsilon Convergence threshold for the residual norm.
 *
 * @return Final global residual norm after convergence.
 *
 * @note This function assumes that:
 * - Halo exchange between MPI ranks is handled outside or between iterations.
 * - Device memory management and kernel execution are handled internally.
 * - Residual computation uses a global reduction across MPI ranks.
 *****************************************************************************************/
double solve_poisson_cuda(LocalProblemData &data, double epsilon);