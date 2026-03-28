#include "problem_setup.h"
#include <boost/program_options.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace std;
namespace po = boost::program_options;

/************************************************************************
 * @brief Macro used to map a C++ symbol name to the corresponding
 * Fortran symbol naming convention.
 *
 * This is used for linking against BLAS-style Fortran routines
 * whose symbols typically have a trailing underscore.
 *
 * @param x Base function name.
 ************************************************************************/
#define F77NAME(x) x##_
extern "C"

{
    /******************************************************************************************
     * @brief Compute the Euclidean norm of a vector using the Fortran BLAS routine `dnrm2`.
     *
     * This routine returns the 2-norm of the vector:
     * \f[
     * \|x\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
     * \f]
     *
     * @param n Number of elements in the vector.
     * @param x Pointer to the input vector.
     * @param incX Stride between successive vector elements.
     * @return Euclidean 2-norm of the vector.
     ******************************************************************************************/
    double F77NAME(dnrm2)(const int &n, const double *x, const int &incX);
}

/*********************************************************************************
 * @brief Allocate memory for problem data arrays.
 *
 * This function initialises the grid dimensions and allocates memory
 * for the following arrays:
 * - forcing term (`f`)
 * - solution (`u`)
 * - temporary solution (`u_new`)
 * - exact/boundary solution (`g`)
 *
 * All arrays are zero-initialised.
 *
 * @param data Problem data structure to initialise.
 * @param Nx Number of grid points in x-direction.
 * @param Ny Number of grid points in y-direction.
 * @param Nz Number of grid points in z-direction.
 ********************************************************************************/
void allocate(ProblemData &data, int Nx, int Ny, int Nz)
{
    data.Nx = Nx;
    data.Ny = Ny;
    data.Nz = Nz;

    const size_t total = static_cast<size_t>(Nx) * Ny * Nz;

    data.f = new double[total]();
    data.u = new double[total]();
    data.u_new = new double[total]();
    data.g = new double[total]();
}

/***************************************************************************
 * @brief Free dynamically allocated memory in ProblemData.
 *
 * Deallocates all arrays and resets pointers to nullptr.
 * Grid dimensions are reset to zero to avoid accidental reuse.
 *
 * @param data Problem data structure to clean up.
 ***************************************************************************/
void delete_dynamic_memory(ProblemData &data)
{
    delete[] data.f;
    delete[] data.u;
    delete[] data.u_new;
    delete[] data.g;

    data.f = nullptr;
    data.u = nullptr;
    data.u_new = nullptr;
    data.g = nullptr;

    data.Nx = 0;
    data.Ny = 0;
    data.Nz = 0;
}

/*********************************************************************************
 * @brief Compute analytical solution for verification test cases.
 *
 * Provides exact solutions used for:
 * - boundary conditions
 * - error analysis
 *
 * Supported cases:
 * 1. Polynomial: \f$ u = x^2 + y^2 + z^2 \f$
 * 2. Smooth trigonometric solution
 * 3. Anisotropic oscillatory solution
 *
 * @param test_case Identifier of the test case.
 * @param x x-coordinate.
 * @param y y-coordinate.
 * @param z z-coordinate.
 * @return Exact solution value at (x, y, z).
 *
 * @throws std::runtime_error If test case is invalid.
 *********************************************************************************/
double verification_solution(int test_case, double x, double y, double z)
{
    switch (test_case)
    {
    case 1:
        return x * x + y * y + z * z;
    case 2:
        return sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
    case 3:
        return sin(M_PI * x) * sin(4.0 * M_PI * y) * sin(8.0 * M_PI * z);
    case 4:
        return 0;
    case 5:
        return 0;
    default:
        throw runtime_error("Invalid test case entered");
    }
}

/********************************************************************************
 * @brief Compute L2 error between numerical and analytical solution.
 *
 * The error is defined as:
 * \f[
 * \|u_{numerical} - u_{exact}\|_2
 * \f]
 *
 * The exact solution is obtained from @ref verification_solution.
 * The norm is computed using BLAS `dnrm2`.
 *
 * @param data Problem data containing computed solution.
 * @param test Test case identifier.
 * @return L2 norm of the error.
 ********************************************************************************/
double compute_error(const ProblemData &data, int test)
{
    double hx = 1.0 / (data.Nx - 1);
    double hy = 1.0 / (data.Ny - 1);
    double hz = 1.0 / (data.Nz - 1);
    int size = data.Nx * data.Ny * data.Nz;

    double error_sum = 0.0;

    double *difference = new double[size]();

    for (int i = 0; i < data.Nx; ++i)
    {
        double x = i * hx;
        for (int j = 0; j < data.Ny; ++j)
        {
            double y = j * hy;
            for (int k = 0; k < data.Nz; ++k)
            {
                double z = k * hz;
                double u_exact = verification_solution(test, x, y, z);
                double u_poisson = data.u[idx(i, j, k, data.Ny, data.Nz)];
                const size_t p = idx(i, j, k, data.Ny, data.Nz);
                difference[p] = u_poisson - u_exact;
            }
        }
    }
    error_sum = F77NAME(dnrm2)(size, difference, 1);
    delete[] difference;
    return error_sum;
}

/*************************************************************************
 * @brief Define forcing functions for different test cases.
 *
 * These correspond to the analytical solutions used in
 * @ref verification_solution.
 *
 * Supported cases:
 * 1. Constant forcing (polynomial solution)
 * 2. Smooth sinusoidal forcing
 * 3. Anisotropic oscillatory forcing
 * 4. Gaussian forcing (localized source)
 * 5. Discontinuous forcing
 *
 * @param test_case Identifier of the test case.
 * @param x x-coordinate.
 * @param y y-coordinate.
 * @param z z-coordinate.
 * @return Forcing value at (x, y, z).
 *
 * @throws std::runtime_error If test case is invalid.
 *************************************************************************/
double forcing(int test_case, double x, double y, double z)
{
    switch (test_case)
    {
    case 1:
        return 6.0;
    case 2:
        return -3.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
    case 3:
        return -81.0 * M_PI * M_PI * sin(M_PI * x) * sin(4.0 * M_PI * y) * sin(8.0 * M_PI * z);
    case 4:
        return 100.0 * exp(-100.0 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5)));
    case 5:
        return (x < 0.5) ? 1.0 : -1.0;
    default:
        throw runtime_error("Invalid test case entered");
    }
}

/*************************************************************************************************
 * @brief Build and initialise the Poisson problem using spatial discretisation.
 *
 * This function:
 * - allocates memory,
 * - evaluates the forcing term at each grid point,
 * - applies Dirichlet boundary conditions using the analytical solution.
 *
 * Interior points are left initialised to zero (or previous values),
 * while boundary points are set using @ref verification_solution.
 *
 * @param data Problem data structure.
 * @param test_case Selected test case.
 * @param Nx Number of grid points in x-direction.
 * @param Ny Number of grid points in y-direction.
 * @param Nz Number of grid points in z-direction.
 *************************************************************************************************/
void spatial_discretisation_and_buildup(ProblemData &data, int test_case, int Nx, int Ny, int Nz)
{
    allocate(data, Nx, Ny, Nz);

    const double hx = 1.0 / (Nx - 1);
    const double hy = 1.0 / (Ny - 1);
    const double hz = 1.0 / (Nz - 1);

    for (int i = 0; i < Nx; ++i)
    {
        const double x = i * hx;
        for (int j = 0; j < Ny; ++j)
        {
            const double y = j * hy;
            for (int k = 0; k < Nz; ++k)
            {
                const double z = k * hz;
                const size_t p = idx(i, j, k, Ny, Nz);

                data.f[p] = forcing(test_case, x, y, z);

                const bool is_boundary = (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1 || k == 0 || k == Nz - 1);

                if (is_boundary)
                {
                    data.g[p] = verification_solution(test_case, x, y, z);
                    data.u[p] = data.g[p];
                    data.u_new[p] = data.g[p];
                }
            }
        }
    }
}

/*****************************************************************************************
 * @brief Read forcing function values from a file.
 *
 * The file format is expected to be:
 * Nx Ny Nz
 * x y z value
 * ...
 *
 * Only the forcing values are stored; coordinates are read but not used.
 *
 * @param data Problem data structure.
 * @param filename Path to input file.
 *
 * @throws std::runtime_error If file cannot be opened or parsed.
 *****************************************************************************************/
void read_forcing(ProblemData &data, const string &filename)
{
    ifstream fin(filename);
    if (!fin)
    {
        throw runtime_error("Error! Forcing file cannot be opened!");
    }

    int Nx, Ny, Nz;
    fin >> Nx >> Ny >> Nz;
    if (!fin)
    {
        throw runtime_error("Error! Nx Ny Nz failed to read from forcing file!");
    }

    allocate(data, Nx, Ny, Nz);

    double x, y, z, value;
    const size_t total = static_cast<size_t>(Nx) * Ny * Nz;

    for (size_t n = 0; n < total; ++n)
    {
        fin >> x >> y >> z >> value;
        if (!fin)
        {
            throw runtime_error("Error! Failed to read from forcing file!");
        }
        data.f[n] = value;
    }
}

/*********************************************************************
 * @brief Write computed solution to file.
 *
 * Output format:
 * Nx Ny Nz
 * x y z u(x,y,z)
 *
 * The solution is written for every grid point.
 *
 * @param filename Output file name.
 * @param data Problem data containing the solution.
 *
 * @throws std::runtime_error If file cannot be opened.
 *********************************************************************/
void output_write(const string &filename, const ProblemData &data)
{
    ofstream fout(filename);
    if (!fout)
    {
        throw runtime_error("Error! File cannot be opened!");
    }

    fout << data.Nx << " " << data.Ny << " " << data.Nz << endl;

    const double hx = 1.0 / (data.Nx - 1);
    const double hy = 1.0 / (data.Ny - 1);
    const double hz = 1.0 / (data.Nz - 1);

    for (int i = 0; i < data.Nx; ++i)
    {
        const double x = i * hx;
        for (int j = 0; j < data.Ny; ++j)
        {
            const double y = j * hy;
            for (int k = 0; k < data.Nz; ++k)
            {
                const double z = k * hz;
                fout << x << " " << y << " " << z << " " << data.u[idx(i, j, k, data.Ny, data.Nz)] << endl;
            }
        }
    }
}