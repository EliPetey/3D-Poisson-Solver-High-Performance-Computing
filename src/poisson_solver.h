#pragma once
#include <cstddef>
#include "indexing.h"

/************************************************************************************************
 * @brief Data structure representing the full (non-distributed) 3D Poisson problem.
 *
 * This structure stores all data required to define and solve a Poisson equation
 * on a structured Cartesian grid in a serial (non-MPI) setting.
 *
 * The problem being solved is:
 * \f[
 * \nabla^2 u = f
 * \f]
 *
 * Memory layout:
 * - All arrays are stored as flattened 1D arrays using row-major ordering.
 * - Indexing is performed using the helper function @ref idx.
 *
 * Grid layout:
 * \code
 * i = 0 ........ Nx-1
 * j = 0 ........ Ny-1
 * k = 0 ........ Nz-1
 * \endcode
 *
 * Boundary points (i = 0, Nx-1, etc.) are assumed to satisfy Dirichlet
 * boundary conditions.
 ************************************************************************************************/
struct ProblemData
{
    /*****************************************************************
     * @brief Number of grid points in each spatial direction.
     *
     * Defines the resolution of the computational domain.
     *****************************************************************/
    int Nx = 0;
    int Ny = 0;
    int Nz = 0;

    /*************************************************************
     * @brief Forcing term array.
     *
     * Stores the right-hand side of the Poisson equation.
     0************************************************************/
    double *f = nullptr;

    /*************************************************************
     * @brief Current solution array.
     *
     * Contains the current approximation to the solution.
     *************************************************************/
    double *u = nullptr;

    /**************************************************************************
     * @brief Temporary solution array for iterative updates.
     *
     * Used in Jacobi iteration to store updated values before swapping.
     **************************************************************************/
    double *u_new = nullptr;

    /***************************************************************
     * @brief Exact or boundary solution values.
     *
     * Used for:
     * - applying Dirichlet boundary conditions, or
     * - computing error against a known analytical solution.
     ***************************************************************/
    double *g = nullptr;
};

/***************************************************************************************
 * @brief Solve the 3D Poisson equation using a Jacobi iterative method.
 *
 * This function performs an iterative solution of the discretised Poisson
 * equation on a structured grid using second-order central differences.
 *
 * The solver updates the solution until the residual norm falls below the
 * specified tolerance.
 *
 * @param data Problem data structure containing grid dimensions and arrays.
 * @param epsilon Convergence tolerance for the residual norm.
 * @return Final residual norm after convergence.
 *
 * @note Uses Jacobi iteration (two-array scheme with swapping).
 * @note Boundary conditions are assumed to be handled in the initialisation.
 ***************************************************************************************/
double solve_poisson(ProblemData &data, double epsilon);