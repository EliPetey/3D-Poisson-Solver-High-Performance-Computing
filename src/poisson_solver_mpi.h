#pragma once
#include <cstddef>
#include <mpi.h>
#include "indexing.h"

/*************************************************************************************
 * @brief Data structure representing a locally owned subdomain in an MPI-based
 *        3D Poisson problem.
 *
 * This structure stores all information required by each MPI rank to:
 * - describe its portion of the global domain,
 * - perform halo exchanges with neighbouring processes,
 * - store solution and forcing data including ghost cells.
 *
 * The domain is decomposed into a 3D Cartesian grid of MPI processes.
 * Each process owns a subdomain augmented with halo (ghost) layers
 * for stencil computations.
 *
 * Memory layout:
 * - Arrays are stored in flattened 1D format using row-major ordering.
 * - Allocated sizes (`alloc_*`) include halo layers.
 *
 * Grid layout per dimension:
 * \code
 * halo | interior | halo
 *   0   1..local_N   local_N+1
 * \endcode
 *************************************************************************************/
struct LocalProblemData
{
    /****************************************************************
     * @brief Global grid dimensions.
     *
     * Total number of grid points in each spatial direction.
     ****************************************************************/

    int global_Nx, global_Ny, global_Nz;
    /****************************************************************
     * @brief Local subdomain dimensions (excluding halo cells).
     *
     * Number of grid points owned by this MPI rank.
     ****************************************************************/
    int local_Nx, local_Ny, local_Nz;

    /****************************************************************
     * @brief Allocated array dimensions (including halo layers).
     *
     * Typically:
     * - alloc_Nx = local_Nx + 2
     * - alloc_Ny = local_Ny + 2
     * - alloc_Nz = local_Nz + 2
     ****************************************************************/
    int alloc_Nx, alloc_Ny, alloc_Nz;

    /******************************************************************************
     * @brief Starting global indices of the local subdomain.
     *
     * These define the offset of this rank’s data within the global grid.
     ******************************************************************************/
    int start_x, start_y, start_z;

    /****************************************************************
     * @brief Neighbour ranks in each coordinate direction.
     *
     * Each pair represents:
     * - left / right neighbour in a given dimension.
     *
     * If a neighbour does not exist (domain boundary),
     * the value is MPI_PROC_NULL.
     ****************************************************************/
    int left_rank_x, right_rank_x;
    int left_rank_y, right_rank_y;
    int left_rank_z, right_rank_z;

    /****************************************************************
     * @brief MPI communicator for the Cartesian process topology.
     ****************************************************************/
    MPI_Comm comm = MPI_COMM_NULL;

    /****************************************************************
     * @brief Forcing term array.
     *
     * Stores the right-hand side of the Poisson equation:
     * \f[
     * \nabla^2 u = f
     * \f]
     ****************************************************************/
    double *f = nullptr;

    /****************************************************************
     * @brief Current solution array.
     *
     * Includes both interior values and halo cells.
     ****************************************************************/
    double *u = nullptr;

    /****************************************************************************
     * @brief Updated solution array for iterative methods.
     *
     * Used for schemes such as Jacobi iteration where updates are stored
     * separately before swapping.
     ****************************************************************************/
    double *u_new = nullptr;
};

/***********************************************************************************
 * @brief Solve the 3D Poisson equation using MPI-based Jacobi iteration.
 *
 * This function operates on a distributed domain described by
 * @ref LocalProblemData and iteratively updates the solution until
 * convergence is reached.
 *
 * @param data Local problem data structure.
 * @param epsilon Convergence tolerance for the residual norm.
 * @return Final global residual norm after convergence.
 *
 * @note Requires halo exchange between MPI ranks at each iteration.
 ***********************************************************************************/
double solve_poisson_mpi(LocalProblemData &data, double epsilon);