#include <iostream>
#include "../src/poisson_solver.h"
#include "../src/problem_setup.h"

using namespace std;

/**************************************************************************************
 * @brief Execute a single serial Poisson solver test case.
 *
 * This function:
 * - builds the problem using spatial discretisation,
 * - solves the Poisson equation,
 * - computes the error against the analytical solution,
 * - checks convergence based on the residual.
 *
 * @param test Test case identifier.
 * @param Nx Number of grid points in x-direction.
 * @param Ny Number of grid points in y-direction.
 * @param Nz Number of grid points in z-direction.
 * @param epsilon Convergence tolerance for residual.
 *
 * @return true if the test passes (residual < epsilon), false otherwise.
 **************************************************************************************/
bool execute_case(int test, int Nx, int Ny, int Nz, double epsilon)
{
    /// Problem data container
    ProblemData data;

    /// Initialise problem (forcing + boundary conditions)
    spatial_discretisation_and_buildup(data, test, Nx, Ny, Nz);

    /// Solve Poisson equation
    double residual = solve_poisson(data, epsilon);

    /// Compute error against analytical solution
    double error = compute_error(data, test);

    cout << "Test: " << test << " Residual: " << residual << " Error: " << error << endl;

    /// Check convergence criterion
    bool passed = (residual < epsilon);

    /// Clean up allocated memory
    delete_dynamic_memory(data);

    return passed;
}

/****************************************************************************
 * @brief Entry point for serial Poisson solver test suite.
 *
 * This program runs multiple predefined test cases to verify:
 * - solver convergence,
 * - correctness of implementation.
 *
 * Test cases:
 * - Case 1: Small grid, strict tolerance
 * - Case 2: Larger grid, relaxed tolerance
 * - Case 3: Anisotropic grid
 *
 * The program reports the number of failed tests and returns:
 * - 0 if all tests pass,
 * - non-zero otherwise.
 *
 * @return Exit status code.
 ****************************************************************************/
int main()
{
    int failed = 0;

    /// Run test cases with different grid sizes and tolerances
    if (!execute_case(1, 32, 32, 32, 1e-8))
    {
        failed++;
    }
    if (!execute_case(2, 64, 64, 64, 1e-1))
    {
        failed++;
    }
    if (!execute_case(3, 32, 64, 128, 2e-1))
    {
        failed++;
    }

    /// Report overall test results
    if (failed == 0)
    {
        cout << "All tests passed!" << endl;
        return 0;
    }
    else
    {
        cout << failed << "tests failed!" << endl;
        return 1;
    }
}