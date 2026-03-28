#include "poisson_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>

using namespace std;

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

/**************************************************************
 * @brief Compute flattened 3D array index for CUDA kernels.
 *
 * Converts 3D indices (i, j, k) into a 1D linear index for
 * row-major storage layout.
 *
 * @param i Index in x-direction.
 * @param j Index in y-direction.
 * @param k Index in z-direction.
 * @param Ny Size in y-direction.
 * @param Nz Size in z-direction.
 * @return Flattened array index.
 **************************************************************/
__host__ __device__
    size_t
    idx_cuda(int i, int j, int k, int Ny, int Nz)
{
    return static_cast<size_t>(i) * Ny * Nz + static_cast<size_t>(j) * Nz + static_cast<size_t>(k);
}

/*************************************************************************
 * @brief Check CUDA error status and abort on failure.
 *
 * If a CUDA API call fails, this function prints an error message
 * and terminates the program.
 *
 * @param err CUDA error code.
 * @param msg Description of the operation being checked.
 *************************************************************************/
static void check_cuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        cerr << "CUDA error: " << msg << " : "
             << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

/********************************************************************************
 * @brief CUDA kernel performing one Jacobi iteration update.
 *
 * Each thread updates one interior grid point using a 7-point stencil.
 * Boundary points are preserved (Dirichlet conditions).
 *
 * Update formula:
 * \f[
 * u^{new}_{i,j,k} =
 * \frac{
 * (u_{i+1,j,k} + u_{i-1,j,k})/h_x^2 +
 * (u_{i,j+1,k} + u_{i,j-1,k})/h_y^2 +
 * (u_{i,j,k+1} + u_{i,j,k-1})/h_z^2 - f
 * }{
 * 2(h_x^{-2} + h_y^{-2} + h_z^{-2})
 * }
 * \f]
 *
 * @param u Current solution field.
 * @param u_new Updated solution field.
 * @param f Forcing term.
 * @param local_Nx Local grid size (x).
 * @param local_Ny Local grid size (y).
 * @param local_Nz Local grid size (z).
 * @param alloc_Ny Allocated size including halos (y).
 * @param alloc_Nz Allocated size including halos (z).
 * @param global_Nx Global grid size (x).
 * @param global_Ny Global grid size (y).
 * @param global_Nz Global grid size (z).
 * @param start_x Global start index (x).
 * @param start_y Global start index (y).
 * @param start_z Global start index (z).
 * @param inv_hx2 Inverse squared grid spacing in x.
 * @param inv_hy2 Inverse squared grid spacing in y.
 * @param inv_hz2 Inverse squared grid spacing in z.
 * @param denom Denominator constant for Jacobi update.
 *******************************************************************************/
__global__ void jacobi_kernel(
    const double *u,
    double *u_new,
    const double *f,
    int local_Nx, int local_Ny, int local_Nz,
    int alloc_Ny, int alloc_Nz,
    int global_Nx, int global_Ny, int global_Nz,
    int start_x, int start_y, int start_z,
    double inv_hx2, double inv_hy2, double inv_hz2,
    double denom)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = local_Nx * local_Ny * local_Nz;

    if (tid >= total)
        return;

    int i0 = tid / (local_Ny * local_Nz);
    int rem = tid % (local_Ny * local_Nz);
    int j0 = rem / local_Nz;
    int k0 = rem % local_Nz;

    int i = i0 + 1;
    int j = j0 + 1;
    int k = k0 + 1;

    int gi = start_x + i0;
    int gj = start_y + j0;
    int gk = start_z + k0;

    bool global_boundary =
        (gi == 0 || gi == global_Nx - 1 ||
         gj == 0 || gj == global_Ny - 1 ||
         gk == 0 || gk == global_Nz - 1);

    size_t p = idx_cuda(i, j, k, alloc_Ny, alloc_Nz);
    size_t pxp = idx_cuda(i + 1, j, k, alloc_Ny, alloc_Nz);
    size_t pxm = idx_cuda(i - 1, j, k, alloc_Ny, alloc_Nz);
    size_t pyp = idx_cuda(i, j + 1, k, alloc_Ny, alloc_Nz);
    size_t pym = idx_cuda(i, j - 1, k, alloc_Ny, alloc_Nz);
    size_t pzp = idx_cuda(i, j, k + 1, alloc_Ny, alloc_Nz);
    size_t pzm = idx_cuda(i, j, k - 1, alloc_Ny, alloc_Nz);

    if (global_boundary)
    {
        u_new[p] = u[p];
    }
    else
    {
        u_new[p] = ((u[pxp] + u[pxm]) * inv_hx2 +
                    (u[pyp] + u[pym]) * inv_hy2 +
                    (u[pzp] + u[pzm]) * inv_hz2 -
                    f[p]) /
                   denom;
    }
}

/**************************************************************************
 * @brief CUDA kernel to compute the residual of the Poisson equation.
 *
 * Computes:
 * \f[
 * r = f - \nabla^2 u
 * \f]
 *
 * Each thread evaluates the residual at one grid point.
 * Boundary points are assigned zero residual.
 *
 * @param u Current solution field.
 * @param f Forcing term.
 * @param r Output residual array (flattened).
 * @param local_Nx Local grid size (x).
 * @param local_Ny Local grid size (y).
 * @param local_Nz Local grid size (z).
 * @param alloc_Ny Allocated size including halos (y).
 * @param alloc_Nz Allocated size including halos (z).
 * @param global_Nx Global grid size (x).
 * @param global_Ny Global grid size (y).
 * @param global_Nz Global grid size (z).
 * @param start_x Global start index (x).
 * @param start_y Global start index (y).
 * @param start_z Global start index (z).
 * @param inv_hx2 Inverse squared grid spacing (x).
 * @param inv_hy2 Inverse squared grid spacing (y).
 * @param inv_hz2 Inverse squared grid spacing (z).
 *************************************************************************/
__global__ void residual_kernel(
    const double *u,
    const double *f,
    double *r,
    int local_Nx, int local_Ny, int local_Nz,
    int alloc_Ny, int alloc_Nz,
    int global_Nx, int global_Ny, int global_Nz,
    int start_x, int start_y, int start_z,
    double inv_hx2, double inv_hy2, double inv_hz2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = local_Nx * local_Ny * local_Nz;

    if (tid >= total)
        return;

    int i0 = tid / (local_Ny * local_Nz);
    int rem = tid % (local_Ny * local_Nz);
    int j0 = rem / local_Nz;
    int k0 = rem % local_Nz;

    int i = i0 + 1;
    int j = j0 + 1;
    int k = k0 + 1;

    int gi = start_x + i0;
    int gj = start_y + j0;
    int gk = start_z + k0;

    bool global_boundary =
        (gi == 0 || gi == global_Nx - 1 ||
         gj == 0 || gj == global_Ny - 1 ||
         gk == 0 || gk == global_Nz - 1);

    if (global_boundary)
    {
        r[tid] = 0.0;
        return;
    }

    size_t p = idx_cuda(i, j, k, alloc_Ny, alloc_Nz);
    size_t pxp = idx_cuda(i + 1, j, k, alloc_Ny, alloc_Nz);
    size_t pxm = idx_cuda(i - 1, j, k, alloc_Ny, alloc_Nz);
    size_t pyp = idx_cuda(i, j + 1, k, alloc_Ny, alloc_Nz);
    size_t pym = idx_cuda(i, j - 1, k, alloc_Ny, alloc_Nz);
    size_t pzp = idx_cuda(i, j, k + 1, alloc_Ny, alloc_Nz);
    size_t pzm = idx_cuda(i, j, k - 1, alloc_Ny, alloc_Nz);

    double laplace =
        (u[pxp] - 2.0 * u[p] + u[pxm]) * inv_hx2 +
        (u[pyp] - 2.0 * u[p] + u[pym]) * inv_hy2 +
        (u[pzp] - 2.0 * u[p] + u[pzm]) * inv_hz2;

    r[tid] = f[p] - laplace;
}

/**************************************************************************************
 * @brief Solve the Poisson equation using CUDA-accelerated Jacobi iteration.
 *
 * This function performs an iterative Jacobi solve on a distributed domain.
 * Each MPI rank operates on its local subdomain while CUDA accelerates
 * the stencil computation.
 *
 * Workflow per iteration:
 * 1. Copy updated solution (with halo values) to device
 * 2. Launch Jacobi kernel
 * 3. Copy updated solution back to host
 * 4. Compute residual using CUDA kernel
 * 5. Compute global residual using MPI_Allreduce
 *
 * The iteration stops when:
 * - residual < epsilon, OR
 * - maximum iterations reached
 *
 * @param data Local problem data (grid, solution, forcing, communicator).
 * @param epsilon Convergence tolerance for residual.
 * @return Final global residual norm.
 *
 * @note Uses BLAS `dnrm2` for local residual norm computation.
 * @note Requires halo exchange to be performed on the host side.
 *************************************************************************************/
double solve_poisson_cuda(LocalProblemData &data, double epsilon)
{
    int global_Nx = data.global_Nx;
    int global_Ny = data.global_Ny;
    int global_Nz = data.global_Nz;

    int local_Nx = data.local_Nx;
    int local_Ny = data.local_Ny;
    int local_Nz = data.local_Nz;

    int alloc_Nx = data.alloc_Nx;
    int alloc_Ny = data.alloc_Ny;
    int alloc_Nz = data.alloc_Nz;

    int start_x = data.start_x;
    int start_y = data.start_y;
    int start_z = data.start_z;

    double *u = data.u;
    double *u_new = data.u_new;
    double *f = data.f;

    const int padded_size = alloc_Nx * alloc_Ny * alloc_Nz;
    const int owned_size = local_Nx * local_Ny * local_Nz;

    double *d_u = nullptr;
    double *d_u_new = nullptr;
    double *d_f = nullptr;
    double *d_r = nullptr;

    check_cuda(cudaMalloc(&d_u, padded_size * sizeof(double)), "cudaMalloc d_u");
    check_cuda(cudaMalloc(&d_u_new, padded_size * sizeof(double)), "cudaMalloc d_u_new");
    check_cuda(cudaMalloc(&d_f, padded_size * sizeof(double)), "cudaMalloc d_f");
    check_cuda(cudaMalloc(&d_r, owned_size * sizeof(double)), "cudaMalloc d_r");

    check_cuda(cudaMemcpy(d_u, u, padded_size * sizeof(double), cudaMemcpyHostToDevice), "copy u H2D");
    check_cuda(cudaMemcpy(d_u_new, u_new, padded_size * sizeof(double), cudaMemcpyHostToDevice), "copy u_new H2D");
    check_cuda(cudaMemcpy(d_f, f, padded_size * sizeof(double), cudaMemcpyHostToDevice), "copy f H2D");

    const double hx = 1.0 / (global_Nx - 1);
    const double hy = 1.0 / (global_Ny - 1);
    const double hz = 1.0 / (global_Nz - 1);

    const double inv_hx2 = 1.0 / (hx * hx);
    const double inv_hy2 = 1.0 / (hy * hy);
    const double inv_hz2 = 1.0 / (hz * hz);

    const double denom = 2.0 * (inv_hx2 + inv_hy2 + inv_hz2);

    const int threads = 256;
    const int blocks = (owned_size + threads - 1) / threads;

    double *r_host = new double[owned_size];

    double residual = 1.0;
    int iter = 0;

    while (residual > epsilon)
    {
        /// Keep the simple version first:
        /// assume host arrays already contain latest halo data from MPI
        check_cuda(cudaMemcpy(d_u, u, padded_size * sizeof(double), cudaMemcpyHostToDevice),
                   "copy halo-updated u H2D");

        jacobi_kernel<<<blocks, threads>>>(
            d_u, d_u_new, d_f,
            local_Nx, local_Ny, local_Nz,
            alloc_Ny, alloc_Nz,
            global_Nx, global_Ny, global_Nz,
            start_x, start_y, start_z,
            inv_hx2, inv_hy2, inv_hz2,
            denom);

        check_cuda(cudaGetLastError(), "launch jacobi_kernel");
        check_cuda(cudaDeviceSynchronize(), "sync jacobi_kernel");

        /// Copy updated field back so host-side MPI code can access it
        check_cuda(cudaMemcpy(u_new, d_u_new, padded_size * sizeof(double), cudaMemcpyDeviceToHost),
                   "copy u_new D2H");

        /// swap host pointers
        double *tmp = u;
        u = u_new;
        u_new = tmp;

        /// keep LocalProblemData in sync
        data.u = u;
        data.u_new = u_new;

        /// copy swapped u back to device for residual
        check_cuda(cudaMemcpy(d_u, u, padded_size * sizeof(double), cudaMemcpyHostToDevice),
                   "copy swapped u H2D");

        residual_kernel<<<blocks, threads>>>(
            d_u, d_f, d_r,
            local_Nx, local_Ny, local_Nz,
            alloc_Ny, alloc_Nz,
            global_Nx, global_Ny, global_Nz,
            start_x, start_y, start_z,
            inv_hx2, inv_hy2, inv_hz2);

        check_cuda(cudaGetLastError(), "launch residual_kernel");
        check_cuda(cudaDeviceSynchronize(), "sync residual_kernel");

        check_cuda(cudaMemcpy(r_host, d_r, owned_size * sizeof(double), cudaMemcpyDeviceToHost),
                   "copy residual D2H");

        int inc = 1;
        double local_norm = F77NAME(dnrm2)(owned_size, r_host, inc);
        double local_sq = local_norm * local_norm;
        double global_sq = 0.0;
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, data.comm);
        residual = sqrt(global_sq);

        ++iter;

        int rank;
        MPI_Comm_rank(data.comm, &rank);
        if (rank == 0 && (iter == 1 || iter % 50 == 0))
        {
            cout << "Iteration: " << iter << " Residual: " << residual << endl;
        }
    }

    /// final host sync
    check_cuda(cudaMemcpy(data.u, d_u, padded_size * sizeof(double), cudaMemcpyDeviceToHost),
               "final copy u D2H");

    delete[] r_host;

    cudaFree(d_u);
    cudaFree(d_u_new);
    cudaFree(d_f);
    cudaFree(d_r);

    return residual;
}