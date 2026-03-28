#include "poisson_solver_mpi.h"

#include <cmath>
#include <iostream>
#include <mpi.h>
#include <cstdlib>

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

/*****************************************************************************
 * @brief Pack halo data in the x-direction for non-blocking communication.
 *
 * Extracts boundary planes:
 * - left face (i = 1)
 * - right face (i = local_Nx)
 *
 * @param u Local solution array including halo cells.
 * @param local_Nx Local grid size in x-direction.
 * @param local_Ny Local grid size in y-direction.
 * @param local_Nz Local grid size in z-direction.
 * @param alloc_Ny Allocated size in y (including halos).
 * @param alloc_Nz Allocated size in z (including halos).
 * @param send_left_x Buffer for left face.
 * @param send_right_x Buffer for right face.
 *****************************************************************************/
void pack_halo_x(const double *u, int local_Nx, int local_Ny, int local_Nz,
                 int alloc_Ny, int alloc_Nz,
                 double *send_left_x, double *send_right_x)
{
    int n = 0;
    for (int j = 1; j <= local_Ny; ++j)
    {
        for (int k = 1; k <= local_Nz; ++k)
        {
            send_left_x[n] = u[idx(1, j, k, alloc_Ny, alloc_Nz)];
            send_right_x[n] = u[idx(local_Nx, j, k, alloc_Ny, alloc_Nz)];
            ++n;
        }
    }
}

/*****************************************************
 * @brief Pack halo data in the y-direction.
 *
 * Packs:
 * - front face (j = 1)
 * - back face (j = local_Ny)
 *****************************************************/
void pack_halo_y(const double *u, int local_Nx, int local_Ny, int local_Nz,
                 int alloc_Ny, int alloc_Nz,
                 double *send_left_y, double *send_right_y)
{
    int n = 0;
    for (int i = 1; i <= local_Nx; ++i)
    {
        for (int k = 1; k <= local_Nz; ++k)
        {
            send_left_y[n] = u[idx(i, 1, k, alloc_Ny, alloc_Nz)];
            send_right_y[n] = u[idx(i, local_Ny, k, alloc_Ny, alloc_Nz)];
            ++n;
        }
    }
}

/******************************************************
 * @brief Pack halo data in the z-direction.
 *
 * Packs:
 * - bottom face (k = 1)
 * - top face (k = local_Nz)
 ******************************************************/
void pack_halo_z(const double *u, int local_Nx, int local_Ny, int local_Nz,
                 int alloc_Ny, int alloc_Nz,
                 double *send_left_z, double *send_right_z)
{
    int n = 0;
    for (int i = 1; i <= local_Nx; ++i)
    {
        for (int j = 1; j <= local_Ny; ++j)
        {
            send_left_z[n] = u[idx(i, j, 1, alloc_Ny, alloc_Nz)];
            send_right_z[n] = u[idx(i, j, local_Nz, alloc_Ny, alloc_Nz)];
            ++n;
        }
    }
}

/***********************************************************************
 * @brief Unpack received halo data into ghost cells (x-direction).
 *
 * Received data is written into:
 * - left halo (i = 0)
 * - right halo (i = local_Nx + 1)
 *
 * @param u Solution array including halos.
 * @param left_rank_x Rank of left neighbour.
 * @param right_rank_x Rank of right neighbour.
 ***********************************************************************/
void unpack_halo_x(double *u, int local_Nx, int local_Ny, int local_Nz,
                   int alloc_Ny, int alloc_Nz,
                   int left_rank_x, int right_rank_x,
                   const double *recv_left_x, const double *recv_right_x)
{
    int n = 0;
    if (left_rank_x != MPI_PROC_NULL)
    {
        for (int j = 1; j <= local_Ny; ++j)
        {
            for (int k = 1; k <= local_Nz; ++k)
            {
                u[idx(0, j, k, alloc_Ny, alloc_Nz)] = recv_left_x[n];
                ++n;
            }
        }
    }

    n = 0;
    if (right_rank_x != MPI_PROC_NULL)
    {
        for (int j = 1; j <= local_Ny; ++j)
        {
            for (int k = 1; k <= local_Nz; ++k)
            {
                u[idx(local_Nx + 1, j, k, alloc_Ny, alloc_Nz)] = recv_right_x[n];
                ++n;
            }
        }
    }
}

/**************************************************************
 * @brief Unpack halo data in y-direction into ghost layers.
 **************************************************************/
void unpack_halo_y(double *u, int local_Nx, int local_Ny, int local_Nz,
                   int alloc_Ny, int alloc_Nz,
                   int left_rank_y, int right_rank_y,
                   const double *recv_left_y, const double *recv_right_y)
{
    int n = 0;
    if (left_rank_y != MPI_PROC_NULL)
    {
        for (int i = 1; i <= local_Nx; ++i)
        {
            for (int k = 1; k <= local_Nz; ++k)
            {
                u[idx(i, 0, k, alloc_Ny, alloc_Nz)] = recv_left_y[n];
                ++n;
            }
        }
    }

    n = 0;
    if (right_rank_y != MPI_PROC_NULL)
    {
        for (int i = 1; i <= local_Nx; ++i)
        {
            for (int k = 1; k <= local_Nz; ++k)
            {
                u[idx(i, local_Ny + 1, k, alloc_Ny, alloc_Nz)] = recv_right_y[n];
                ++n;
            }
        }
    }
}

/**************************************************************
 * @brief Unpack halo data in z-direction into ghost layers.
 **************************************************************/
void unpack_halo_z(double *u, int local_Nx, int local_Ny, int local_Nz,
                   int alloc_Ny, int alloc_Nz,
                   int left_rank_z, int right_rank_z,
                   const double *recv_left_z, const double *recv_right_z)
{
    int n = 0;
    if (left_rank_z != MPI_PROC_NULL)
    {
        for (int i = 1; i <= local_Nx; ++i)
        {
            for (int j = 1; j <= local_Ny; ++j)
            {
                u[idx(i, j, 0, alloc_Ny, alloc_Nz)] = recv_left_z[n];
                ++n;
            }
        }
    }

    n = 0;
    if (right_rank_z != MPI_PROC_NULL)
    {
        for (int i = 1; i <= local_Nx; ++i)
        {
            for (int j = 1; j <= local_Ny; ++j)
            {
                u[idx(i, j, local_Nz + 1, alloc_Ny, alloc_Nz)] = recv_right_z[n];
                ++n;
            }
        }
    }
}

/******************************************************************************
 * @brief Initiate non-blocking halo exchange in all three directions.
 *
 * This function:
 * 1. Packs boundary data into send buffers
 * 2. Posts non-blocking receives (MPI_Irecv)
 * 3. Posts non-blocking sends (MPI_Isend)
 *
 * Communication pattern:
 * - x-direction uses tags 100/101
 * - y-direction uses tags 200/201
 * - z-direction uses tags 300/301
 *
 * @param u Local solution array.
 * @param comm MPI communicator.
 * @param reqs Array of MPI requests (size 12).
 ******************************************************************************/
void start_halo_exchange(
    const double *u,
    int local_Nx, int local_Ny, int local_Nz,
    int alloc_Ny, int alloc_Nz,
    int left_rank_x, int right_rank_x,
    int left_rank_y, int right_rank_y,
    int left_rank_z, int right_rank_z,
    MPI_Comm comm,
    double *send_left_x, double *send_right_x, double *recv_left_x, double *recv_right_x,
    double *send_left_y, double *send_right_y, double *recv_left_y, double *recv_right_y,
    double *send_left_z, double *send_right_z, double *recv_left_z, double *recv_right_z,
    MPI_Request reqs[12])
{
    pack_halo_x(u, local_Nx, local_Ny, local_Nz, alloc_Ny, alloc_Nz, send_left_x, send_right_x);
    pack_halo_y(u, local_Nx, local_Ny, local_Nz, alloc_Ny, alloc_Nz, send_left_y, send_right_y);
    pack_halo_z(u, local_Nx, local_Ny, local_Nz, alloc_Ny, alloc_Nz, send_left_z, send_right_z);

    const int face_x = local_Ny * local_Nz;
    const int face_y = local_Nx * local_Nz;
    const int face_z = local_Nx * local_Ny;

    /// Post receives first
    MPI_Irecv(recv_left_x, face_x, MPI_DOUBLE, left_rank_x, 100, comm, &reqs[0]);
    MPI_Irecv(recv_right_x, face_x, MPI_DOUBLE, right_rank_x, 101, comm, &reqs[1]);

    MPI_Irecv(recv_left_y, face_y, MPI_DOUBLE, left_rank_y, 200, comm, &reqs[2]);
    MPI_Irecv(recv_right_y, face_y, MPI_DOUBLE, right_rank_y, 201, comm, &reqs[3]);

    MPI_Irecv(recv_left_z, face_z, MPI_DOUBLE, left_rank_z, 300, comm, &reqs[4]);
    MPI_Irecv(recv_right_z, face_z, MPI_DOUBLE, right_rank_z, 301, comm, &reqs[5]);

    /// Then sends
    MPI_Isend(send_left_x, face_x, MPI_DOUBLE, left_rank_x, 101, comm, &reqs[6]);
    MPI_Isend(send_right_x, face_x, MPI_DOUBLE, right_rank_x, 100, comm, &reqs[7]);

    MPI_Isend(send_left_y, face_y, MPI_DOUBLE, left_rank_y, 201, comm, &reqs[8]);
    MPI_Isend(send_right_y, face_y, MPI_DOUBLE, right_rank_y, 200, comm, &reqs[9]);

    MPI_Isend(send_left_z, face_z, MPI_DOUBLE, left_rank_z, 301, comm, &reqs[10]);
    MPI_Isend(send_right_z, face_z, MPI_DOUBLE, right_rank_z, 300, comm, &reqs[11]);
}

/*************************************************************
 * @brief Complete halo exchange and unpack received data.
 *
 * This function:
 * 1. Waits for all communication to complete
 * 2. Unpacks received data into ghost cells
 *
 * @param u Local solution array.
 * @param reqs Array of MPI requests.
 *************************************************************/
void finish_halo_exchange(
    double *u,
    int local_Nx, int local_Ny, int local_Nz,
    int alloc_Ny, int alloc_Nz,
    int left_rank_x, int right_rank_x,
    int left_rank_y, int right_rank_y,
    int left_rank_z, int right_rank_z,
    const double *recv_left_x, const double *recv_right_x,
    const double *recv_left_y, const double *recv_right_y,
    const double *recv_left_z, const double *recv_right_z,
    MPI_Request reqs[12])
{
    MPI_Waitall(12, reqs, MPI_STATUSES_IGNORE);

    unpack_halo_x(u, local_Nx, local_Ny, local_Nz, alloc_Ny, alloc_Nz,
                  left_rank_x, right_rank_x, recv_left_x, recv_right_x);

    unpack_halo_y(u, local_Nx, local_Ny, local_Nz, alloc_Ny, alloc_Nz,
                  left_rank_y, right_rank_y, recv_left_y, recv_right_y);

    unpack_halo_z(u, local_Nx, local_Ny, local_Nz, alloc_Ny, alloc_Nz,
                  left_rank_z, right_rank_z, recv_left_z, recv_right_z);
}

/******************************************************************************
 * @brief Perform blocking halo exchange in x-direction using MPI_Sendrecv.
 *
 * Packs, exchanges, and unpacks halo data in a single routine.
 *
 * @note Simpler but less efficient than non-blocking version.
 ******************************************************************************/
void exchange_halo_x(double *u, int local_Nx, int local_Ny, int local_Nz,
                     int left_rank_x, int right_rank_x, MPI_Comm comm,
                     double *send_left_x, double *send_right_x,
                     double *recv_left_x, double *recv_right_x)
{
    const int alloc_Ny = local_Ny + 2;
    const int alloc_Nz = local_Nz + 2;
    int face_size = local_Ny * local_Nz;

    int n = 0;
    /*********************************
     * Packing left face i = 1;
     *********************************/
    for (int j = 1; j <= local_Ny; ++j)
    {
        for (int k = 1; k <= local_Nz; ++k)
        {
            send_left_x[n] = u[idx(1, j, k, alloc_Ny, alloc_Nz)];
            ++n;
        }
    }

    /**************************************
     * Packing right face i = local_Nx
     **************************************/
    n = 0;
    for (int j = 1; j <= local_Ny; ++j)
    {
        for (int k = 1; k <= local_Nz; ++k)
        {
            send_right_x[n] = u[idx(local_Nx, j, k, alloc_Ny, alloc_Nz)];
            ++n;
        }
    }

    /*********************************
     * Exchange with left neighbour
     *********************************/
    MPI_Sendrecv(send_left_x, face_size, MPI_DOUBLE, left_rank_x, 100,
                 recv_right_x, face_size, MPI_DOUBLE, right_rank_x, 100,
                 comm, MPI_STATUS_IGNORE);

    /***********************************
     * Exchange with right neighbour
     ***********************************/
    MPI_Sendrecv(send_right_x, face_size, MPI_DOUBLE, right_rank_x, 101,
                 recv_left_x, face_size, MPI_DOUBLE, left_rank_x, 101,
                 comm, MPI_STATUS_IGNORE);

    /**********************************************
     * Unpack into layers
     *
     * This places data into the ghost halo cells
     *
     * Here, left halo goes into i = 0;
     **********************************************/
    if (left_rank_x != MPI_PROC_NULL)
    {
        n = 0;
        for (int j = 1; j <= local_Ny; ++j)
        {
            for (int k = 1; k <= local_Nz; ++k)
            {
                u[idx(0, j, k, alloc_Ny, alloc_Nz)] = recv_left_x[n];
                ++n;
            }
        }
    }

    /****************************************************
     * Unpack into layers part 2
     *
     * This places data into the ghost halo cells
     *
     * Here, right halo goes into i = local_Nx + 1;
     ****************************************************/
    if (right_rank_x != MPI_PROC_NULL)
    {
        n = 0;
        for (int j = 1; j <= local_Ny; ++j)
        {
            for (int k = 1; k <= local_Nz; ++k)
            {
                u[idx(local_Nx + 1, j, k, alloc_Ny, alloc_Nz)] = recv_right_x[n];
                ++n;
            }
        }
    }
}

/************************************************************
 * @brief Perform blocking halo exchange in y-direction.
 ************************************************************/
void exchange_halo_y(double *u, int local_Nx, int local_Ny, int local_Nz,
                     int left_rank_y, int right_rank_y, MPI_Comm comm,
                     double *send_left_y, double *send_right_y,
                     double *recv_left_y, double *recv_right_y)
{
    const int alloc_Ny = local_Ny + 2;
    const int alloc_Nz = local_Nz + 2;
    int face_size = local_Nx * local_Nz;

    int n = 0;

    /********************************
     * Packing left face i = 1;
     ********************************/
    for (int i = 1; i <= local_Nx; ++i)
    {
        for (int k = 1; k <= local_Nz; ++k)
        {
            send_left_y[n] = u[idx(i, 1, k, alloc_Ny, alloc_Nz)];
            ++n;
        }
    }

    /****************************************
     * Packing right face i = local_Nx
     ****************************************/
    n = 0;
    for (int i = 1; i <= local_Nx; ++i)
    {
        for (int k = 1; k <= local_Nz; ++k)
        {
            send_right_y[n] = u[idx(i, local_Ny, k, alloc_Ny, alloc_Nz)];
            ++n;
        }
    }

    /**********************************
     * Exchange with left neighbour
     **********************************/
    MPI_Sendrecv(send_left_y, face_size, MPI_DOUBLE, left_rank_y, 200,
                 recv_right_y, face_size, MPI_DOUBLE, right_rank_y, 200,
                 comm, MPI_STATUS_IGNORE);

    /***********************************
     * Exchange with right neighbour
     ***********************************/
    MPI_Sendrecv(send_right_y, face_size, MPI_DOUBLE, right_rank_y, 201,
                 recv_left_y, face_size, MPI_DOUBLE, left_rank_y, 201,
                 comm, MPI_STATUS_IGNORE);

    /***************************************************
     * Unpack into layers
     *
     * This places data into the ghost halo cells
     *
     * Here, left halo goes into i = 0;
     ***************************************************/
    if (left_rank_y != MPI_PROC_NULL)
    {
        n = 0;
        for (int i = 1; i <= local_Nx; ++i)
        {
            for (int k = 1; k <= local_Nz; ++k)
            {
                u[idx(i, 0, k, alloc_Ny, alloc_Nz)] = recv_left_y[n];
                ++n;
            }
        }
    }

    /******************************************************
     * Unpack into layers part 2
     *
     * This places data into the ghost halo cells
     *
     * Here, right halo goes into i = local_Nx + 1;
     ******************************************************/
    if (right_rank_y != MPI_PROC_NULL)
    {
        n = 0;
        for (int i = 1; i <= local_Nx; ++i)
        {
            for (int k = 1; k <= local_Nz; ++k)
            {
                u[idx(i, local_Ny + 1, k, alloc_Ny, alloc_Nz)] = recv_right_y[n];
                ++n;
            }
        }
    }
}

/**************************************************************
 * @brief Perform blocking halo exchange in z-direction.
 **************************************************************/
void exchange_halo_z(double *u, int local_Nx, int local_Ny, int local_Nz,
                     int left_rank_z, int right_rank_z, MPI_Comm comm,
                     double *send_left_z, double *send_right_z,
                     double *recv_left_z, double *recv_right_z)
{
    const int alloc_Ny = local_Ny + 2;
    const int alloc_Nz = local_Nz + 2;
    int face_size = local_Nx * local_Ny;

    int n = 0;

    /*****************************
     * Packing left face i = 1;
     *****************************/
    for (int i = 1; i <= local_Nx; ++i)
    {
        for (int j = 1; j <= local_Ny; ++j)
        {
            send_left_z[n] = u[idx(i, j, 1, alloc_Ny, alloc_Nz)];
            ++n;
        }
    }

    /************************************
     * Packing right face i = local_Nx
     ************************************/
    n = 0;
    for (int i = 1; i <= local_Nx; ++i)
    {
        for (int j = 1; j <= local_Ny; ++j)
        {
            send_right_z[n] = u[idx(i, j, local_Nz, alloc_Ny, alloc_Nz)];
            ++n;
        }
    }

    /************************************
     * Exchange with left neighbour
     ************************************/
    MPI_Sendrecv(send_left_z, face_size, MPI_DOUBLE, left_rank_z, 300,
                 recv_right_z, face_size, MPI_DOUBLE, right_rank_z, 300,
                 comm, MPI_STATUS_IGNORE);

    /*************************************
     * Exchange with right neighbour
     *************************************/
    MPI_Sendrecv(send_right_z, face_size, MPI_DOUBLE, right_rank_z, 301,
                 recv_left_z, face_size, MPI_DOUBLE, left_rank_z, 301,
                 comm, MPI_STATUS_IGNORE);

    /****************************************************
     * Unpack into layers
     *
     * This places data into the ghost halo cells
     *
     * Here, left halo goes into i = 0;
     ****************************************************/
    if (left_rank_z != MPI_PROC_NULL)
    {
        n = 0;
        for (int i = 1; i <= local_Nx; ++i)
        {
            for (int j = 1; j <= local_Ny; ++j)
            {
                u[idx(i, j, 0, alloc_Ny, alloc_Nz)] = recv_left_z[n];
                ++n;
            }
        }
    }

    /*****************************************************
     * Unpack into layers part 2
     *
     * This places data into the ghost halo cells
     *
     * Here, right halo goes into i = local_Nx + 1;
     *****************************************************/
    if (right_rank_z != MPI_PROC_NULL)
    {
        n = 0;
        for (int i = 1; i <= local_Nx; ++i)
        {
            for (int j = 1; j <= local_Ny; ++j)
            {
                u[idx(i, j, local_Nz + 1, alloc_Ny, alloc_Nz)] = recv_right_z[n];
                ++n;
            }
        }
    }
}

/*****************************************************************************************
 * @brief Solve the 3D Poisson equation using MPI-based Jacobi iteration.
 *
 * This solver uses:
 * - 2nd-order central finite differences
 * - Jacobi iterative method
 * - 3D domain decomposition with halo exchange
 *
 * Algorithm per iteration:
 * 1. Start non-blocking halo exchange
 * 2. Compute strict interior points (no dependencies)
 * 3. Complete halo exchange
 * 4. Compute boundary-adjacent points
 * 5. Swap solution arrays
 * 6. Compute residual and perform global reduction
 *
 * Convergence criterion:
 * \f[
 * \|r\|_2 < \epsilon
 * \f]
 *
 * @param data Local problem data (grid, solution, forcing, MPI info).
 * @param epsilon Convergence tolerance.
 * @return Final global residual norm.
 *
 * @note Residual is computed using BLAS `dnrm2` and MPI_Allreduce.
 * @note Non-blocking communication overlaps computation and communication.
 *****************************************************************************************/
double solve_poisson_mpi(LocalProblemData &data, double epsilon)
{
    int global_Nx = data.global_Nx;
    int global_Ny = data.global_Ny;
    int global_Nz = data.global_Nz;

    int local_Nx = data.local_Nx;
    int local_Ny = data.local_Ny;
    int local_Nz = data.local_Nz;

    int alloc_Ny = data.alloc_Ny;
    int alloc_Nz = data.alloc_Nz;

    const int start_x = data.start_x;
    const int start_y = data.start_y;
    const int start_z = data.start_z;

    double *f = data.f;
    double *u = data.u;
    double *u_new = data.u_new;

    int owned_size = local_Nx * local_Ny * local_Nz;
    double *r_local = new double[owned_size]();

    const double hx = 1.0 / (global_Nx - 1);
    const double hy = 1.0 / (global_Ny - 1);
    const double hz = 1.0 / (global_Nz - 1);

    const double inv_hx2 = 1.0 / (hx * hx);
    const double inv_hy2 = 1.0 / (hy * hy);
    const double inv_hz2 = 1.0 / (hz * hz);

    const double denom = 2.0 * (inv_hx2 + inv_hy2 + inv_hz2);

    double residual = 1.0;
    int iter = 0;

    double t_exchange = 0.0;
    double t_update = 0.0;
    double t_residual = 0.0;

    MPI_Request reqs[12];

    /// Allocate halo buffers ONCE
    int face_x = local_Ny * local_Nz;
    double *send_left_x = new double[face_x];
    double *send_right_x = new double[face_x];
    double *recv_left_x = new double[face_x];
    double *recv_right_x = new double[face_x];

    int face_y = local_Nx * local_Nz;
    double *send_left_y = new double[face_y];
    double *send_right_y = new double[face_y];
    double *recv_left_y = new double[face_y];
    double *recv_right_y = new double[face_y];

    int face_z = local_Nx * local_Ny;
    double *send_left_z = new double[face_z];
    double *send_right_z = new double[face_z];
    double *recv_left_z = new double[face_z];
    double *recv_right_z = new double[face_z];

    while (residual > epsilon)
    {
        // ---------------------------------
        // Start non-blocking halo exchange
        // ---------------------------------
        double t0 = MPI_Wtime();
        start_halo_exchange(
            u,
            local_Nx, local_Ny, local_Nz,
            alloc_Ny, alloc_Nz,
            data.left_rank_x, data.right_rank_x,
            data.left_rank_y, data.right_rank_y,
            data.left_rank_z, data.right_rank_z,
            data.comm,
            send_left_x, send_right_x, recv_left_x, recv_right_x,
            send_left_y, send_right_y, recv_left_y, recv_right_y,
            send_left_z, send_right_z, recv_left_z, recv_right_z,
            reqs);
        t_exchange += MPI_Wtime() - t0;

        // ---------------------------------
        // Compute strict interior
        // ---------------------------------
        t0 = MPI_Wtime();
        for (int i = 2; i <= local_Nx - 1; ++i)
        {
            const int gi = start_x + (i - 1);
            for (int j = 2; j <= local_Ny - 1; ++j)
            {
                const int gj = start_y + (j - 1);
                for (int k = 2; k <= local_Nz - 1; ++k)
                {
                    const int gk = start_z + (k - 1);

                    const bool global_boundary =
                        (gi == 0 || gi == global_Nx - 1 ||
                         gj == 0 || gj == global_Ny - 1 ||
                         gk == 0 || gk == global_Nz - 1);

                    const size_t p = idx(i, j, k, alloc_Ny, alloc_Nz);
                    const size_t pxp = idx(i + 1, j, k, alloc_Ny, alloc_Nz);
                    const size_t pxm = idx(i - 1, j, k, alloc_Ny, alloc_Nz);
                    const size_t pyp = idx(i, j + 1, k, alloc_Ny, alloc_Nz);
                    const size_t pym = idx(i, j - 1, k, alloc_Ny, alloc_Nz);
                    const size_t pzp = idx(i, j, k + 1, alloc_Ny, alloc_Nz);
                    const size_t pzm = idx(i, j, k - 1, alloc_Ny, alloc_Nz);

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
            }
        }
        t_update += MPI_Wtime() - t0;

        // ---------------------------------
        // Finish exchange and unpack halos
        // ---------------------------------
        t0 = MPI_Wtime();
        finish_halo_exchange(
            u,
            local_Nx, local_Ny, local_Nz,
            alloc_Ny, alloc_Nz,
            data.left_rank_x, data.right_rank_x,
            data.left_rank_y, data.right_rank_y,
            data.left_rank_z, data.right_rank_z,
            recv_left_x, recv_right_x,
            recv_left_y, recv_right_y,
            recv_left_z, recv_right_z,
            reqs);
        t_exchange += MPI_Wtime() - t0;

        // ---------------------------------
        // Compute boundary-adjacent owned cells
        // ---------------------------------
        t0 = MPI_Wtime();
        for (int i = 1; i <= local_Nx; ++i)
        {
            const int gi = start_x + (i - 1);
            for (int j = 1; j <= local_Ny; ++j)
            {
                const int gj = start_y + (j - 1);
                for (int k = 1; k <= local_Nz; ++k)
                {
                    const int gk = start_z + (k - 1);

                    const bool on_local_boundary =
                        (i == 1 || i == local_Nx ||
                         j == 1 || j == local_Ny ||
                         k == 1 || k == local_Nz);

                    if (!on_local_boundary)
                    {
                        continue;
                    }

                    const bool global_boundary =
                        (gi == 0 || gi == global_Nx - 1 ||
                         gj == 0 || gj == global_Ny - 1 ||
                         gk == 0 || gk == global_Nz - 1);

                    const size_t p = idx(i, j, k, alloc_Ny, alloc_Nz);
                    const size_t pxp = idx(i + 1, j, k, alloc_Ny, alloc_Nz);
                    const size_t pxm = idx(i - 1, j, k, alloc_Ny, alloc_Nz);
                    const size_t pyp = idx(i, j + 1, k, alloc_Ny, alloc_Nz);
                    const size_t pym = idx(i, j - 1, k, alloc_Ny, alloc_Nz);
                    const size_t pzp = idx(i, j, k + 1, alloc_Ny, alloc_Nz);
                    const size_t pzm = idx(i, j, k - 1, alloc_Ny, alloc_Nz);

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
            }
        }
        t_update += MPI_Wtime() - t0;

        // ---------------------------------
        // Swap solution arrays
        // ---------------------------------
        double *temp = u;
        u = u_new;
        u_new = temp;

        // ---------------------------------
        // Residual computation
        // ---------------------------------
        t0 = MPI_Wtime();
        int n = 0;
        for (int i = 1; i <= local_Nx; ++i)
        {
            const int gi = start_x + (i - 1);
            for (int j = 1; j <= local_Ny; ++j)
            {
                const int gj = start_y + (j - 1);
                for (int k = 1; k <= local_Nz; ++k)
                {
                    const int gk = start_z + (k - 1);

                    const bool global_boundary =
                        (gi == 0 || gi == global_Nx - 1 ||
                         gj == 0 || gj == global_Ny - 1 ||
                         gk == 0 || gk == global_Nz - 1);

                    if (global_boundary)
                    {
                        continue;
                    }

                    const size_t p = idx(i, j, k, alloc_Ny, alloc_Nz);
                    const size_t pxp = idx(i + 1, j, k, alloc_Ny, alloc_Nz);
                    const size_t pxm = idx(i - 1, j, k, alloc_Ny, alloc_Nz);
                    const size_t pyp = idx(i, j + 1, k, alloc_Ny, alloc_Nz);
                    const size_t pym = idx(i, j - 1, k, alloc_Ny, alloc_Nz);
                    const size_t pzp = idx(i, j, k + 1, alloc_Ny, alloc_Nz);
                    const size_t pzm = idx(i, j, k - 1, alloc_Ny, alloc_Nz);

                    const double laplace =
                        (u[pxp] - 2.0 * u[p] + u[pxm]) * inv_hx2 +
                        (u[pyp] - 2.0 * u[p] + u[pym]) * inv_hy2 +
                        (u[pzp] - 2.0 * u[p] + u[pzm]) * inv_hz2;

                    r_local[n] = f[p] - laplace;
                    ++n;
                }
            }
        }

        double local_norm = F77NAME(dnrm2)(n, r_local, 1);
        double local_sq = local_norm * local_norm;
        double global_sq = 0.0;
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, data.comm);
        residual = sqrt(global_sq);
        t_residual += MPI_Wtime() - t0;

        ++iter;

        //int rank;
        //MPI_Comm_rank(data.comm, &rank);
        //if (rank == 0 && (iter == 1 || iter % 50 == 0))
        //{
        //    cout << "Iteration: " << iter << " Residual: " << residual << endl;
        //}
    }

    data.u = u;
    data.u_new = u_new;

    //double max_exchange = 0.0;
    //double max_update = 0.0;
    //double max_residual_time = 0.0;

    //MPI_Reduce(&t_exchange, &max_exchange, 1, MPI_DOUBLE, MPI_MAX, 0, data.comm);
    //MPI_Reduce(&t_update, &max_update, 1, MPI_DOUBLE, MPI_MAX, 0, data.comm);
    //MPI_Reduce(&t_residual, &max_residual_time, 1, MPI_DOUBLE, MPI_MAX, 0, data.comm);

    //int rank;
    //MPI_Comm_rank(data.comm, &rank);
    //if (rank == 0)
    //{
        //cout << "\nTiming summary (max over ranks):\n";
        //cout << "Halo exchange time: " << max_exchange << " s" << endl;
        //cout << "Jacobi update time: " << max_update << " s" << endl;
        //cout << "Residual time: " << max_residual_time << " s" << endl;
    //}

    delete[] send_left_x;
    delete[] send_right_x;
    delete[] recv_left_x;
    delete[] recv_right_x;

    delete[] send_left_y;
    delete[] send_right_y;
    delete[] recv_left_y;
    delete[] recv_right_y;

    delete[] send_left_z;
    delete[] send_right_z;
    delete[] recv_left_z;
    delete[] recv_right_z;

    delete[] r_local;
    return residual;
}