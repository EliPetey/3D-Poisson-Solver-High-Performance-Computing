#include <iostream>
#include <cmath>
#include <stdexcept>

#include <mpi.h>

#include "../src/poisson_solver_mpi.h"
#include "../src/problem_setup.h"

using namespace std;

/***********************************************************************************
 * @brief Select a simple 3D Cartesian process grid based on world size.
 *
 * This helper function assigns a fixed decomposition for common test sizes:
 * - 1 rank  → 1×1×1
 * - 4 ranks → 2×2×1
 * - 8 ranks → 2×2×2
 *
 * This keeps testing straightforward and avoids general factorisation logic.
 *
 * @param world_size Total number of MPI ranks.
 * @param Px Number of processes in x-direction (output).
 * @param Py Number of processes in y-direction (output).
 * @param Pz Number of processes in z-direction (output).
 *
 * @throws std::runtime_error If unsupported world size is used.
 ***********************************************************************************/
static void choose_process_grid(int world_size, int &Px, int &Py, int &Pz)
{
    if (world_size == 1)
    {
        Px = 1;
        Py = 1;
        Pz = 1;
    }
    else if (world_size == 4)
    {
        Px = 2;
        Py = 2;
        Pz = 1;
    }
    else if (world_size == 8)
    {
        Px = 2;
        Py = 2;
        Pz = 2;
    }
    else
    {
        throw runtime_error("Please run test_solver_mpi with 1, 4, or 8 MPI ranks.");
    }
}

/*******************************************************************************
 * @brief Compute local grid size for non-uniform decomposition.
 *
 * When the global size is not divisible by the number of processes,
 * the first few ranks receive one additional grid point.
 *
 * @param global_n Global grid size in one dimension.
 * @param Pdim Number of processes in that dimension.
 * @param coord Process coordinate in that dimension.
 * @return Local grid size.
 *******************************************************************************/
static int distribute(int global_n, int Pdim, int coord)
{
    const int base = global_n / Pdim;
    const int remainder = global_n % Pdim;
    return base + (coord < remainder ? 1 : 0);
}

/********************************************************************************
 * @brief Compute starting global index for a process.
 *
 * Accounts for uneven distribution when grid sizes are not divisible.
 *
 * @param global_n Global grid size.
 * @param Pdim Number of processes.
 * @param coord Process coordinate.
 * @return Starting global index of the local subdomain.
 ********************************************************************************/
static int local_start(int global_n, int Pdim, int coord)
{
    const int base = global_n / Pdim;
    const int remainder = global_n % Pdim;
    return coord * base + min(coord, remainder);
}

/********************************************************************************
 * @brief Initialise local subdomain using analytical test case.
 *
 * For each locally owned grid point:
 * - evaluates forcing term,
 * - applies Dirichlet boundary conditions where applicable.
 *
 * @param problem Local MPI problem data.
 * @param test_case Selected verification test case.
 *******************************************************************************/
static void initialise_local_test_case(LocalProblemData &problem, int test_case)
{
    const double hx = 1.0 / (problem.global_Nx - 1);
    const double hy = 1.0 / (problem.global_Ny - 1);
    const double hz = 1.0 / (problem.global_Nz - 1);

    for (int i = 1; i <= problem.local_Nx; ++i)
    {
        const int gi = problem.start_x + (i - 1);
        const double x = gi * hx;

        for (int j = 1; j <= problem.local_Ny; ++j)
        {
            const int gj = problem.start_y + (j - 1);
            const double y = gj * hy;

            for (int k = 1; k <= problem.local_Nz; ++k)
            {
                const int gk = problem.start_z + (k - 1);
                const double z = gk * hz;

                const size_t p = idx(i, j, k, problem.alloc_Ny, problem.alloc_Nz);

                problem.f[p] = forcing(test_case, x, y, z);

                const bool global_boundary =
                    (gi == 0 || gi == problem.global_Nx - 1 ||
                     gj == 0 || gj == problem.global_Ny - 1 ||
                     gk == 0 || gk == problem.global_Nz - 1);

                if (global_boundary)
                {
                    const double g = verification_solution(test_case, x, y, z);
                    problem.u[p] = g;
                    problem.u_new[p] = g;
                }
            }
        }
    }
}

/*************************************************************
 * @brief Compute global maximum absolute error.
 *
 * Each rank computes its local maximum error:
 * \f[
 * \max |u_{numerical} - u_{exact}|
 * \f]
 *
 * A global maximum is obtained using MPI_Allreduce.
 *
 * @param problem Local problem data.
 * @param test_case Verification test case.
 * @return Global maximum error across all MPI ranks.
 *************************************************************/
static double compute_global_max_error(const LocalProblemData &problem, int test_case)
{
    const double hx = 1.0 / (problem.global_Nx - 1);
    const double hy = 1.0 / (problem.global_Ny - 1);
    const double hz = 1.0 / (problem.global_Nz - 1);

    double local_max_err = 0.0;

    for (int i = 1; i <= problem.local_Nx; ++i)
    {
        const int gi = problem.start_x + (i - 1);
        const double x = gi * hx;

        for (int j = 1; j <= problem.local_Ny; ++j)
        {
            const int gj = problem.start_y + (j - 1);
            const double y = gj * hy;

            for (int k = 1; k <= problem.local_Nz; ++k)
            {
                const int gk = problem.start_z + (k - 1);
                const double z = gk * hz;

                const size_t p = idx(i, j, k, problem.alloc_Ny, problem.alloc_Nz);

                const double exact = verification_solution(test_case, x, y, z);
                const double err = fabs(problem.u[p] - exact);

                if (err > local_max_err)
                {
                    local_max_err = err;
                }
            }
        }
    }

    double global_max_err = 0.0;
    MPI_Allreduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, problem.comm);

    return global_max_err;
}

/*********************************************************************
 * @brief Construct a distributed Poisson problem for testing.
 *
 * This function:
 * - builds a Cartesian communicator,
 * - computes local domain sizes and offsets,
 * - determines neighbour ranks,
 * - allocates memory including halo layers,
 * - initialises data using analytical test case.
 *
 * @param problem Local problem data structure (output).
 * @param test_case Selected test case.
 * @param Nx Global grid size (x).
 * @param Ny Global grid size (y).
 * @param Nz Global grid size (z).
 * @param cart_comm Cartesian MPI communicator.
 *********************************************************************/
static void build_local_problem(LocalProblemData &problem,
                                int test_case,
                                int Nx, int Ny, int Nz,
                                MPI_Comm cart_comm)
{
    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);

    int world_size;
    MPI_Comm_size(cart_comm, &world_size);

    int Px, Py, Pz;
    choose_process_grid(world_size, Px, Py, Pz);

    int coords[3];
    MPI_Cart_coords(cart_comm, cart_rank, 3, coords);

    int left_rank_x, right_rank_x;
    int left_rank_y, right_rank_y;
    int left_rank_z, right_rank_z;

    MPI_Cart_shift(cart_comm, 0, 1, &left_rank_x, &right_rank_x);
    MPI_Cart_shift(cart_comm, 1, 1, &left_rank_y, &right_rank_y);
    MPI_Cart_shift(cart_comm, 2, 1, &left_rank_z, &right_rank_z);

    const int local_Nx = distribute(Nx, Px, coords[0]);
    const int local_Ny = distribute(Ny, Py, coords[1]);
    const int local_Nz = distribute(Nz, Pz, coords[2]);

    const int start_x = local_start(Nx, Px, coords[0]);
    const int start_y = local_start(Ny, Py, coords[1]);
    const int start_z = local_start(Nz, Pz, coords[2]);

    problem.global_Nx = Nx;
    problem.global_Ny = Ny;
    problem.global_Nz = Nz;

    problem.local_Nx = local_Nx;
    problem.local_Ny = local_Ny;
    problem.local_Nz = local_Nz;

    problem.alloc_Nx = local_Nx + 2;
    problem.alloc_Ny = local_Ny + 2;
    problem.alloc_Nz = local_Nz + 2;

    problem.start_x = start_x;
    problem.start_y = start_y;
    problem.start_z = start_z;

    problem.left_rank_x = left_rank_x;
    problem.right_rank_x = right_rank_x;
    problem.left_rank_y = left_rank_y;
    problem.right_rank_y = right_rank_y;
    problem.left_rank_z = left_rank_z;
    problem.right_rank_z = right_rank_z;

    problem.comm = cart_comm;

    const int padded_size = problem.alloc_Nx * problem.alloc_Ny * problem.alloc_Nz;
    problem.f = new double[padded_size]();
    problem.u = new double[padded_size]();
    problem.u_new = new double[padded_size]();

    initialise_local_test_case(problem, test_case);
}

/*************************************************************************
 * @brief Release dynamically allocated memory for local problem.
 *
 * Frees arrays and resets pointers to nullptr.
 *
 * @param problem Local problem data structure.
 *************************************************************************/
static void delete_local_problem(LocalProblemData &problem)
{
    delete[] problem.f;
    delete[] problem.u;
    delete[] problem.u_new;

    problem.f = nullptr;
    problem.u = nullptr;
    problem.u_new = nullptr;
}

/**********************************************************************
 * @brief Execute a single MPI test case.
 *
 * This function:
 * - builds the distributed problem,
 * - solves the Poisson equation,
 * - computes residual and error,
 * - verifies correctness across all ranks.
 *
 * Pass criteria:
 * - residual < epsilon
 * - max error < 1.0
 *
 * @param test_case Test case identifier.
 * @param Nx Global grid size (x).
 * @param Ny Global grid size (y).
 * @param Nz Global grid size (z).
 * @param epsilon Convergence tolerance.
 * @param cart_comm Cartesian MPI communicator.
 *
 * @return true if test passes globally, false otherwise.
 **********************************************************************/
static bool execute_case(int test_case, int Nx, int Ny, int Nz, double epsilon, MPI_Comm cart_comm)
{
    LocalProblemData data;
    build_local_problem(data, test_case, Nx, Ny, Nz, cart_comm);

    const double residual = solve_poisson_mpi(data, epsilon);
    const double error = compute_global_max_error(data, test_case);

    int rank;
    MPI_Comm_rank(cart_comm, &rank);

    if (rank == 0)
    {
        cout << "MPI Test: " << test_case
             << " Grid: " << Nx << "x" << Ny << "x" << Nz
             << " Residual: " << residual
             << " Max Error: " << error << endl;
    }

    const bool local_pass = (residual < epsilon) && (error < 1.0);
    int local_pass_int = local_pass ? 1 : 0;
    int global_pass_int = 0;

    MPI_Allreduce(&local_pass_int, &global_pass_int, 1, MPI_INT, MPI_MIN, cart_comm);

    delete_local_problem(data);

    return (global_pass_int == 1);
}

/************************************************************************
 * @brief Entry point for MPI Poisson solver test suite.
 *
 * This program:
 * - initialises MPI,
 * - constructs a Cartesian communicator,
 * - runs multiple verification test cases,
 * - reports pass/fail status.
 *
 * Supported configurations:
 * - 1, 4, or 8 MPI ranks
 *
 * Test cases:
 * - Case 1: Small grid, strict tolerance
 * - Case 2: Larger grid, relaxed tolerance
 * - Case 3: Anisotropic grid
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 if all tests pass, non-zero otherwise.
 ************************************************************************/
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int Px, Py, Pz;
    bool valid_world_size = true;

    try
    {
        choose_process_grid(world_size, Px, Py, Pz);
    }
    catch (const exception &e)
    {
        valid_world_size = false;
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            cerr << e.what() << endl;
        }
    }

    if (!valid_world_size)
    {
        MPI_Finalize();
        return 1;
    }

    int dims[3] = {Px, Py, Pz};
    int periods[3] = {0, 0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);

    int failed = 0;

    if (!execute_case(1, 32, 32, 32, 1e-8, cart_comm))
    {
        failed++;
    }

    if (!execute_case(2, 64, 64, 64, 1e-1, cart_comm))
    {
        failed++;
    }

    if (!execute_case(3, 32, 64, 128, 2e-1, cart_comm))
    {
        failed++;
    }

    int rank;
    MPI_Comm_rank(cart_comm, &rank);

    if (rank == 0)
    {
        if (failed == 0)
        {
            cout << "All MPI tests passed!" << endl;
        }
        else
        {
            cout << failed << " MPI tests failed!" << endl;
        }
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    return (failed == 0) ? 0 : 1;
}