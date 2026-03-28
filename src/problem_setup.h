#define _USE_MATH_DEFINES
#pragma once
#include <string>
#include "poisson_solver.h"

using namespace std;

/**********************************************************************************************
 * @brief Allocate memory and initialise ProblemData structure.
 *
 * Allocates arrays for:
 * - forcing term (`f`)
 * - solution (`u`)
 * - temporary solution (`u_new`)
 * - exact/boundary solution (`g`)
 *
 * All arrays are initialised to zero.
 *
 * @param data Problem data structure to initialise.
 * @param Nx Number of grid points in x-direction.
 * @param Ny Number of grid points in y-direction.
 * @param Nz Number of grid points in z-direction.
 **********************************************************************************************/
void allocate(ProblemData &data, int Nx, int Ny, int Nz);

/**********************************************************************************************
 * @brief Deallocate memory associated with ProblemData.
 *
 * Frees all dynamically allocated arrays and resets pointers to nullptr.
 * Grid dimensions are reset to zero.
 *
 * @param data Problem data structure to clean up.
 **********************************************************************************************/
void delete_dynamic_memory(ProblemData &data);

/**********************************************************************************************
 * @brief Evaluate analytical solution for verification cases.
 *
 * Used for:
 * - applying Dirichlet boundary conditions
 * - computing numerical error
 *
 * @param test_case Identifier of the test case.
 * @param x x-coordinate.
 * @param y y-coordinate.
 * @param z z-coordinate.
 * @return Exact solution value at (x, y, z).
 *
 * @throws std::runtime_error If test case is invalid.
 **********************************************************************************************/
double verification_solution(int test_case, double x, double y, double z);

/**********************************************************************************************
 * @brief Compute L2 error between numerical and analytical solutions.
 *
 * Computes:
 * \f[
 * \|u_{numerical} - u_{exact}\|_2
 * \f]
 *
 * @param data Problem data containing computed solution.
 * @param test Test case identifier.
 * @return L2 norm of the error.
 **********************************************************************************************/
double compute_error(const ProblemData &data, int test);

/**********************************************************************************************
 * @brief Evaluate forcing function for a given test case.
 *
 * Defines the right-hand side of the Poisson equation:
 * \f[
 * \nabla^2 u = f
 * \f]
 *
 * @param test_case Identifier of the test case.
 * @param x x-coordinate.
 * @param y y-coordinate.
 * @param z z-coordinate.
 * @return Forcing value at (x, y, z).
 *
 * @throws std::runtime_error If test case is invalid.
 **********************************************************************************************/
double forcing(int test_case, double x, double y, double z);

/**********************************************************************************************
 * @brief Initialise problem using spatial discretisation.
 *
 * This function:
 * - allocates memory,
 * - evaluates forcing at each grid point,
 * - applies Dirichlet boundary conditions using analytical solution.
 *
 * @param data Problem data structure.
 * @param test_case Selected test case.
 * @param Nx Number of grid points in x-direction.
 * @param Ny Number of grid points in y-direction.
 * @param Nz Number of grid points in z-direction.
 **********************************************************************************************/
void spatial_discretisation_and_buildup(ProblemData &data, int test_case, int Nx, int Ny, int Nz);

/**********************************************************************************************
 * @brief Read forcing data from file.
 *
 * Expected file format:
 * Nx Ny Nz
 * x y z value
 *
 * Only the forcing values are stored.
 *
 * @param data Problem data structure.
 * @param filename Path to input file.
 *
 * @throws std::runtime_error If file cannot be opened or parsed.
 **********************************************************************************************/
void read_forcing(ProblemData &data, const string &filename);

/**********************************************************************************************
 * @brief Write computed solution to file.
 *
 * Output format:
 * Nx Ny Nz
 * x y z u(x,y,z)
 *
 * @param filename Output file name.
 * @param data Problem data containing solution.
 *
 * @throws std::runtime_error If file cannot be opened.
 **********************************************************************************************/
void output_write(const string &filename, const ProblemData &data);
