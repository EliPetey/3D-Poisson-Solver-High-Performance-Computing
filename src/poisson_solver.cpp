#include "poisson_solver.h"
#include <iostream>

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

/******************************************************************************************************************************
 * @brief Solve the 3D Poisson equation using Jacobi iteration.
 *
 * This function solves:
 * \f[
 * \nabla^2 u = f
 * \f]
 * on a structured 3D grid using second-order central finite differences.
 *
 * The Jacobi method is used as the iterative solver:
 * - Each iteration updates all interior points using values from the previous iteration.
 * - Boundary points are assumed fixed (Dirichlet conditions).
 *
 * Algorithm per iteration:
 * 1. Update solution using a 7-point stencil
 * 2. Swap solution buffers (`u` and `u_new`)
 * 3. Compute residual vector
 * 4. Compute residual norm using BLAS
 *
 * Convergence criterion:
 * \f[
 * \|r\|_2 < \epsilon
 * \f]
 *
 * @param data Problem data structure containing:
 *             - grid dimensions (`Nx`, `Ny`, `Nz`)
 *             - solution arrays (`u`, `u_new`)
 *             - forcing term (`f`)
 * @param epsilon Convergence tolerance for the residual norm.
 *
 * @return Final residual norm after convergence.
 *
 * @note Uses a maximum iteration cap of 10,000.
 * @note Residual is computed over the full grid (including boundaries, though only interior contributes).
 *****************************************************************************************************************************/
double solve_poisson(ProblemData &data, double epsilon)
{
    /// Extract grid dimensions and initialise solver variables
    int Nx = data.Nx;
    int Ny = data.Ny;
    int Nz = data.Nz;
    int size = Nx * Ny * Nz;
    double residual = 1.0;
    int count = 0;

    /// Residual vector (flattened 3D array)
    double *r = new double[size]();

    double *f = data.f;
    double *u = data.u;
    double *u_new = data.u_new;

    /// Compute grid spacing and stencil coefficients
    double hx = 1.0 / (Nx - 1);
    double hy = 1.0 / (Ny - 1);
    double hz = 1.0 / (Nz - 1);

    double inv_hx2 = 1.0 / (hx * hx);
    double inv_hy2 = 1.0 / (hy * hy);
    double inv_hz2 = 1.0 / (hz * hz);

    double denom = 2.0 * (inv_hx2 + inv_hy2 + inv_hz2);

    /****************************************************************
     * @brief Main Jacobi iteration loop.
     *
     * Iterates until convergence or maximum iterations reached.
     ****************************************************************/
    while (residual > epsilon)
    {
        /// Update interior grid points using 7 point stencil
        for (int i = 1; i < Nx - 1; ++i)
        {
            for (int j = 1; j < Ny - 1; ++j)
            {
                for (int k = 1; k < Nz - 1; ++k)
                {
                    int p = idx(i, j, k, Ny, Nz);
                    u_new[p] = ((u[idx(i + 1, j, k, Ny, Nz)] + u[idx(i - 1, j, k, Ny, Nz)]) * inv_hx2 +
                                (u[idx(i, j + 1, k, Ny, Nz)] + u[idx(i, j - 1, k, Ny, Nz)]) * inv_hy2 +
                                (u[idx(i, j, k + 1, Ny, Nz)] + u[idx(i, j, k - 1, Ny, Nz)]) * inv_hz2 -
                                f[p]) /
                               denom;
                }
            }
        }

        /// Swap solution buffers for next iteration
        double *temp = u;
        u = u_new;
        u_new = temp;

        /// Reset residual vector
        for (int n = 0; n < size; ++n)
        {
            r[n] = 0.0;
        }

        /// Compute residual for interior points
        for (int i = 1; i < Nx - 1; ++i)
        {
            for (int j = 1; j < Ny - 1; ++j)
            {
                for (int k = 1; k < Nz - 1; ++k)
                {
                    int p = idx(i, j, k, Ny, Nz);

                    double laplace =
                        (u[idx(i + 1, j, k, Ny, Nz)] - 2.0 * u[p] + u[idx(i - 1, j, k, Ny, Nz)]) * inv_hx2 +
                        (u[idx(i, j + 1, k, Ny, Nz)] - 2.0 * u[p] + u[idx(i, j - 1, k, Ny, Nz)]) * inv_hy2 +
                        (u[idx(i, j, k + 1, Ny, Nz)] - 2.0 * u[p] + u[idx(i, j, k - 1, Ny, Nz)]) * inv_hz2;

                    r[p] = f[p] - laplace;
                }
            }
        }

        /// Compute global residual norm using BLAS
        residual = F77NAME(dnrm2)(size, r, 1);
        ++count;
        std::cout << "Count: " << count << " Residual: " << residual << std::endl;
    }

    /// Store final solution back into data structure
    data.u = u;
    data.u_new = u_new;

    delete[] r;
    return residual;
}