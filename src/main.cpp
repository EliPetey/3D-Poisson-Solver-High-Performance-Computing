#define _USE_MATH_DEFINES
#include <boost/program_options.hpp>
#include "poisson_solver.h"
#include "problem_setup.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <fstream>

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

/*********************************************************************************************
 * @brief Entry point for the serial Poisson solver application.
 *
 * This program solves a discretised Poisson problem on a 3D structured grid.
 * The problem may be initialised either from:
 * - a built-in analytic test case, or
 * - an external forcing file.
 *
 * The program performs the following steps:
 * - parse command-line arguments,
 * - validate the selected input mode,
 * - build the problem data structures,
 * - solve the Poisson system,
 * - compute the final error,
 * - write the solution to file,
 * - release dynamically allocated memory.
 *
 * Supported command-line options:
 * - `--help`               Print available options
 * - `--forcing <file>`     Read forcing data from file
 * - `--test <1-5>`         Select built-in test case
 * - `--Nx <int>`           Number of grid points in x-direction
 * - `--Ny <int>`           Number of grid points in y-direction
 * - `--Nz <int>`           Number of grid points in z-direction
 * - `--epsilon <double>`   Residual convergence threshold
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * @return `0` on successful completion.
 *
 * @throws std::runtime_error If the command-line arguments are invalid.
 *********************************************************************************************/
int main(int argc, char *argv[])
{
    /************************************************************************
     * @brief Default solver parameters and storage.
     *
     * These values are used unless overridden by command-line arguments.
     * The `problem` structure stores all grid, forcing, solution, and
     * discretisation data required by the solver.
     ************************************************************************/
    int test = 1;
    int Nx = 32;
    int Ny = 32;
    int Nz = 32;
    double epsilon = 1e-8;
    ProblemData problem;

    /// Name of the input forcing file when file-based initialisation is used.
    string forcing_filename;

    /// Program options defined
    po::options_description opts("Allowed options");
    opts.add_options()("help", "Print available options.")("forcing", po::value<string>(&forcing_filename), "input forcing file")("test", po::value<int>(&test)->default_value(1), "Test case to use (1-5)")("Nx", po::value<int>(&Nx)->default_value(32), "Number of grid points (x)")("Ny", po::value<int>(&Ny)->default_value(32), "Number of grid points (y)")("Nz", po::value<int>(&Nz)->default_value(32), "Number of grid points (z)")("epsilon", po::value<double>(&epsilon)->default_value(1e-8), "Residual threshold");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        cout << opts << endl;
        return 0;
    };

    /********************************************************************************************************
     * @brief Command Line Argument input criteria
     *
     * Defines the 5 boolean types for:
     *
     * 1. --test
     * 2. --forcing
     * 3. --User entered Nx
     * 4. --User entered Ny
     * 5. --User entered Nz
     *
     * Checks the following:
     *
     * 1. Either --forcing or --test should be provided, not both
     * 2. Only if --test provided, then --Nx, --Ny, --Nz can be specified
     * 3. If --test is given, only options 1-5 is allowed
     * 4. if --forcing specified, reads the forcing function using values of Nx Ny Nz from the file
     ********************************************************************************************************/

    /****************************************************************************
     * @brief Determine which command-line options were explicitly provided.
     *
     * These flags are used to validate the user input and ensure that
     * only one problem initialisation mode is selected.
     ****************************************************************************/
    const bool test_present = vm.count("test") > 0;
    const bool forcing_present = vm.count("forcing") > 0;
    const bool Nx_user = vm.count("Nx") > 0;
    const bool Ny_user = vm.count("Ny") > 0;
    const bool Nz_user = vm.count("Nz") > 0;

    if (test_present == forcing_present)
    {
        throw runtime_error("Either --forcing or --test must be provided, not BOTH!");
    }

    if (forcing_present && (Nx_user || Ny_user || Nz_user))
    {
        throw runtime_error("You can only use --Nx --Ny --Nz only if --test is specified!");
    }

    if (test_present && (test < 1 || test > 5))
    {
        throw runtime_error("Only use 1-5 for test case!");
    }

    /*************************************************************************
     * @brief Initialise the problem data.
     *
     * In forcing-file mode, the forcing function and grid dimensions are
     * read from file. Otherwise, the selected analytic test case is used
     * to construct the discretised problem on the specified grid.
     *************************************************************************/
    if (forcing_present)
    {
        read_forcing(problem, forcing_filename);
    }
    else
    {
        spatial_discretisation_and_buildup(problem, test, Nx, Ny, Nz);
    }

    /**************************************************************************
     * @brief Solve the Poisson system and evaluate solution accuracy.
     *
     * The solver iterates until the residual drops below the user-defined
     * convergence threshold. After convergence, the error against the
     * verification solution is computed.
     **************************************************************************/
    double final_residual = solve_poisson(problem, epsilon);
    double error = compute_error(problem, test);

    /// Write the computed solution to an output file 'solution.txt in the same folder'
    output_write("solution.txt", problem);

    /// Release dynamically allocated memory associated with the problem
    delete_dynamic_memory(problem);

    cout << "Final Residual: " << final_residual << endl;
    cout << "Final Error: " << error << endl;

    return 0;
}