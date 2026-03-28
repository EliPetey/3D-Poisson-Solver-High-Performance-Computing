#define _USE_MATH_DEFINES

#include <boost/program_options.hpp>
#include <mpi.h>
#include "poisson_solver_mpi.h"
#include "poisson_cuda.h"
#include "problem_setup.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdlib>

using namespace std;
namespace po = boost::program_options;

/****************************************************************************
 * @brief Broadcast a string from the root MPI rank to all other ranks.
 *
 * This function first broadcasts the string length, then resizes the
 * receiving buffers and broadcasts the actual string data.
 *
 * @param s String to broadcast (modified on non-root ranks).
 * @param root Rank of the broadcasting process.
 * @param comm MPI communicator.
 ****************************************************************************/
static void broadcast_string(string &s, int root, MPI_Comm comm)
{
    int len = 0;

    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == root)
    {
        len = static_cast<int>(s.size());
    }

    MPI_Bcast(&len, 1, MPI_INT, root, comm);

    s.resize(len);

    if (len > 0)
    {
        MPI_Bcast(s.data(), len, MPI_CHAR, root, comm);
    }
}

/**************************************************************************
 * @brief Compute local subdomain size for a given process coordinate.
 *
 * Handles non-uniform decomposition when global size is not divisible
 * by the number of processes. The first `remainder` ranks receive one
 * additional grid point.
 *
 * @param global_n Total number of grid points in a dimension.
 * @param Pdim Number of processes in that dimension.
 * @param coord Coordinate of the current process in that dimension.
 * @return Local number of grid points assigned to this process.
 **************************************************************************/
static int distribute(int global_n, int Pdim, int coord)
{
    const int base = global_n / Pdim;
    const int remainder = global_n % Pdim;
    return base + (coord < remainder ? 1 : 0);
}

/*****************************************************************************
 * @brief Compute starting global index for a process in 1D decomposition.
 *
 * Determines the offset into the global grid for the current process,
 * accounting for uneven decomposition.
 *
 * @param global_n Total number of grid points.
 * @param Pdim Number of processes.
 * @param coord Coordinate of this process.
 * @return Starting global index for this process.
 *****************************************************************************/
static int local_start(int global_n, int Pdim, int coord)
{
    const int base = global_n / Pdim;
    const int remainder = global_n % Pdim;
    return coord * base + min(coord, remainder);
}

/***********************************************************************
 * @brief Initialise a test case on the local MPI subdomain.
 *
 * Populates the forcing term and boundary conditions using analytic
 * functions for verification. Boundary values are set using the
 * known exact solution.
 *
 * @param problem Local problem data structure.
 * @param test_case Identifier for the test case (1–5).
 ***********************************************************************/
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

                const bool global_boundary = (gi == 0 || gi == problem.global_Nx - 1 || gj == 0 || gj == problem.global_Ny - 1 || gk == 0 || gk == problem.global_Nz - 1);

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

/****************************************************************************
 * @brief Initialise forcing term from file for local MPI subdomain.
 *
 * Each MPI rank reads the full file but only stores values belonging
 * to its assigned subdomain. Homogeneous Dirichlet boundary conditions
 * are applied.
 *
 * @param problem Local problem data structure.
 * @param filename Path to forcing input file.
 *
 * @throws std::runtime_error If file cannot be read or format is invalid.
 *****************************************************************************/

static void initialise_local_forcing_from_file(LocalProblemData &problem, const string &filename)
{
    ifstream fin(filename);
    if (!fin)
    {
        throw runtime_error("Error! Cannot open forcing file!");
    }

    int file_Nx, file_Ny, file_Nz;
    fin >> file_Nx >> file_Ny >> file_Nz;

    if (!fin)
    {
        throw runtime_error("Error! Failed to read Nx Ny Nz from forcing file!");
    }

    if (file_Nx != problem.global_Nx || file_Ny != problem.global_Ny || file_Nz != problem.global_Nz)
    {
        throw runtime_error("Error! Forcing file dimensions do not match broadcast global dimensions!");
    }

    const int total = file_Nx * file_Ny * file_Nz;

    for (int n = 0; n < total; ++n)
    {
        double x, y, z, value;
        fin >> x >> y >> z >> value;

        if (!fin)
        {
            throw runtime_error("Error! Forcing file contains invalid data points!");
        }

        const int gi = n / (file_Ny * file_Nz);
        const int gj = (n / file_Nz) % file_Ny;
        const int gk = n % file_Nz;

        const bool owned_x = (gi >= problem.start_x && gi < problem.start_x + problem.local_Nx);
        const bool owned_y = (gj >= problem.start_y && gj < problem.start_y + problem.local_Ny);
        const bool owned_z = (gk >= problem.start_z && gk < problem.start_z + problem.local_Nz);

        if (owned_x && owned_y && owned_z)
        {
            const int li = gi - problem.start_x + 1;
            const int lj = gj - problem.start_y + 1;
            const int lk = gk - problem.start_z + 1;

            const size_t p = idx(li, lj, lk, problem.alloc_Ny, problem.alloc_Nz);

            problem.f[p] = value;

            const bool global_boundary = (gi == 0 || gi == problem.global_Nx - 1 || gj == 0 || gj == problem.global_Ny - 1 || gk == 0 || gk == problem.global_Nz - 1);

            if (global_boundary)
            {
                problem.u[p] = 0.0;
                problem.u_new[p] = 0.0;
            }
        }
    }
}

/****************************************************************************
 * @brief Gather distributed solution and write to a single output file.
 *
 * Each rank sends its local solution block to rank 0, which reconstructs
 * the global 3D field and writes it to disk.
 *
 * Output format:
 * Nx Ny Nz
 * x y z u(x,y,z)
 *
 * @param filename Output file name.
 * @param problem Local problem data.
 * @param comm MPI communicator.
 *
 * @note Communication uses point-to-point MPI_Send/MPI_Recv.
 ****************************************************************************/
static void write_solution_mpi(const string &filename,
                               const LocalProblemData &problem,
                               MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int global_Nx = problem.global_Nx;
    const int global_Ny = problem.global_Ny;
    const int global_Nz = problem.global_Nz;

    const int local_Nx = problem.local_Nx;
    const int local_Ny = problem.local_Ny;
    const int local_Nz = problem.local_Nz;

    const int start_x = problem.start_x;
    const int start_y = problem.start_y;
    const int start_z = problem.start_z;

    const int alloc_Ny = problem.alloc_Ny;
    const int alloc_Nz = problem.alloc_Nz;

    const int local_count = local_Nx * local_Ny * local_Nz;

    /// Pack owned interior points only
    double *sendbuf = new double[local_count];
    int n = 0;
    for (int i = 1; i <= local_Nx; ++i)
    {
        for (int j = 1; j <= local_Ny; ++j)
        {
            for (int k = 1; k <= local_Nz; ++k)
            {
                sendbuf[n] = problem.u[idx(i, j, k, alloc_Ny, alloc_Nz)];
                ++n;
            }
        }
    }

    if (rank == 0)
    {
        double *global_u = new double[global_Nx * global_Ny * global_Nz]();

        /// First insert rank 0's own block
        n = 0;
        for (int i = 0; i < local_Nx; ++i)
        {
            for (int j = 0; j < local_Ny; ++j)
            {
                for (int k = 0; k < local_Nz; ++k)
                {
                    const int gi = start_x + i;
                    const int gj = start_y + j;
                    const int gk = start_z + k;

                    const size_t gp =
                        static_cast<size_t>(gi) * global_Ny * global_Nz +
                        static_cast<size_t>(gj) * global_Nz +
                        static_cast<size_t>(gk);

                    global_u[gp] = sendbuf[n];
                    ++n;
                }
            }
        }

        /// Receive and insert all other ranks
        for (int src = 1; src < size; ++src)
        {
            int meta[6];
            MPI_Recv(meta, 6, MPI_INT, src, 900, comm, MPI_STATUS_IGNORE);

            const int sx = meta[0];
            const int sy = meta[1];
            const int sz = meta[2];
            const int lx = meta[3];
            const int ly = meta[4];
            const int lz = meta[5];

            const int recv_count = lx * ly * lz;
            double *recvbuf = new double[recv_count];

            MPI_Recv(recvbuf, recv_count, MPI_DOUBLE, src, 901, comm, MPI_STATUS_IGNORE);

            int m = 0;
            for (int i = 0; i < lx; ++i)
            {
                for (int j = 0; j < ly; ++j)
                {
                    for (int k = 0; k < lz; ++k)
                    {
                        const int gi = sx + i;
                        const int gj = sy + j;
                        const int gk = sz + k;

                        const size_t gp =
                            static_cast<size_t>(gi) * global_Ny * global_Nz +
                            static_cast<size_t>(gj) * global_Nz +
                            static_cast<size_t>(gk);

                        global_u[gp] = recvbuf[m];
                        ++m;
                    }
                }
            }

            delete[] recvbuf;
        }

        /// Write single output file
        ofstream fout(filename);
        if (!fout)
        {
            delete[] global_u;
            delete[] sendbuf;
            throw runtime_error("Could not open output file: " + filename);
        }

        fout << global_Nx << " " << global_Ny << " " << global_Nz << "\n";

        const double hx = 1.0 / (global_Nx - 1);
        const double hy = 1.0 / (global_Ny - 1);
        const double hz = 1.0 / (global_Nz - 1);

        for (int i = 0; i < global_Nx; ++i)
        {
            const double x = i * hx;
            for (int j = 0; j < global_Ny; ++j)
            {
                const double y = j * hy;
                for (int k = 0; k < global_Nz; ++k)
                {
                    const double z = k * hz;

                    const size_t gp =
                        static_cast<size_t>(i) * global_Ny * global_Nz +
                        static_cast<size_t>(j) * global_Nz +
                        static_cast<size_t>(k);

                    fout << x << " " << y << " " << z << " " << global_u[gp] << "\n";
                }
            }
        }

        delete[] global_u;
    }
    else
    {
        int meta[6] = {
            start_x, start_y, start_z,
            local_Nx, local_Ny, local_Nz};

        MPI_Send(meta, 6, MPI_INT, 0, 900, comm);
        MPI_Send(sendbuf, local_count, MPI_DOUBLE, 0, 901, comm);
    }

    delete[] sendbuf;
}

/************************************************************************************
 * @brief Entry point for MPI-based Poisson solver with optional CUDA acceleration.
 *
 * Workflow:
 * 1. Initialise MPI
 * 2. Parse command-line arguments (rank 0)
 * 3. Broadcast configuration to all ranks
 * 4. Create Cartesian MPI topology
 * 5. Decompose domain across processes
 * 6. Initialise problem (analytic test case or file input)
 * 7. Solve Poisson equation using CUDA backend
 * 8. Gather and write global solution
 * 9. Finalise MPI
 *
 * Command-line options:
 * --test <1-5>        : Use built-in test case
 * --forcing <file>    : Use forcing from file
 * --Nx, --Ny, --Nz    : Grid dimensions (test mode only)
 * --Px, --Py, --Pz    : Process grid dimensions
 * --epsilon           : Convergence tolerance
 *
 * @param argc Argument count
 * @param argv Argument vector
 * @return Exit status (0 = success, 1 = error)
 ***********************************************************************************/
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int test = 1;
    int Nx = 32, Ny = 32, Nz = 32;
    int Px = 1, Py = 1, Pz = 1;
    double epsilon = 1e-8;
    LocalProblemData problem;

    string forcing_filename;

    int help_flag = 0;
    int error_flag = 0;
    int forcing_flag = 0;

    if (world_rank == 0)
    {
        po::options_description opts("Allowed options");
        opts.add_options()("help", "Print available options.")("forcing", po::value<string>(&forcing_filename), "input forcing file")("test", po::value<int>(&test)->default_value(1), "Test case to use (1-5)")("Nx", po::value<int>(&Nx)->default_value(32), "Number of grid points (x)")("Ny", po::value<int>(&Ny)->default_value(32), "Number of grid points (y)")("Nz", po::value<int>(&Nz)->default_value(32), "Number of grid points (z)")("epsilon", po::value<double>(&epsilon)->default_value(1e-8), "Residual threshold")("Px", po::value<int>(&Px)->default_value(1), "Number of processes (x)")("Py", po::value<int>(&Py)->default_value(1), "Number of processes (y)")("Pz", po::value<int>(&Pz)->default_value(1), "Number of processes (z)");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, opts), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            cout << opts << endl;
            help_flag = 1;
        }

        const bool test_present = vm.count("test") > 0 && !vm["test"].defaulted();

        bool forcing_present = vm.count("forcing") > 0;

        const bool Nx_user = vm.count("Nx") > 0 && !vm["Nx"].defaulted();

        const bool Ny_user = vm.count("Ny") > 0 && !vm["Ny"].defaulted();

        const bool Nz_user = vm.count("Nz") > 0 && !vm["Nz"].defaulted();

        forcing_flag = forcing_present ? 1 : 0;

        if (!help_flag)
        {
            if (test_present == forcing_present)
            {
                cerr << "Either one of --test or --forcing must be provided. Not BOTH!" << endl;
                ;
                error_flag = 1;
            }

            if (forcing_present && (Nx_user || Ny_user || Nz_user))
            {
                cerr << "You can only use --Nx --Ny --Nz only if --test is specified!" << endl;
                error_flag = 1;
            }

            if (test_present && (test < 1 || test > 5))
            {
                cerr << "Only use 1-5 for test case!" << endl;
                error_flag = 1;
            }

            if (Px * Py * Pz != world_size)
            {
                cerr << "Px, Py, Pz do not multiply to world rank!" << endl;
                error_flag = 1;
            }

            /// If forcing inputted, read dimensions from file on rank 0
            if (!error_flag && forcing_present)
            {
                ifstream fin(forcing_filename);
                if (!fin)
                {
                    cerr << "Error! Cannot open forcing file!" << endl;
                    error_flag = 1;
                }
                else
                {
                    fin >> Nx >> Ny >> Nz;
                    if (!fin)
                    {
                        cerr << "Error! Cannot read Nx Ny Nz from forcing file!" << endl;
                        error_flag = 1;
                    }
                }
            }
        }
    }

    MPI_Bcast(&help_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&error_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (help_flag || error_flag)
    {
        MPI_Finalize();
        return help_flag ? 0 : 1;
    }

    MPI_Bcast(&test, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Px, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Py, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Pz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&forcing_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /**
     * @section MPI_Decomposition MPI Domain Decomposition
     *
     * The global domain is decomposed into a 3D Cartesian process grid.
     * Each process owns a subdomain with halo (ghost) cells for neighbour exchange.
     *
     * Layout per dimension:
     * halo | interior | halo
     *
     * Communication directions:
     * +x / -x : neighbouring processes in x-direction
     * +y / -y : neighbouring processes in y-direction
     * +z / -z : neighbouring processes in z-direction
     *
     * Iterative solver workflow:
     * 1. Exchange halo regions
     * 2. Update interior grid points
     * 3. Compute local residual
     * 4. Perform global reduction (MPI_Allreduce)
     */

    if (forcing_flag)
    {
        broadcast_string(forcing_filename, 0, MPI_COMM_WORLD);
    }

    int dims[3] = {Px, Py, Pz};
    int periods[3] = {0, 0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);

    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);

    int coords[3];
    MPI_Cart_coords(cart_comm, cart_rank, 3, coords);

    int left_rank_x, right_rank_x;
    int left_rank_y, right_rank_y;
    int left_rank_z, right_rank_z;

    MPI_Cart_shift(cart_comm, 0, 1, &left_rank_x, &right_rank_x);
    MPI_Cart_shift(cart_comm, 1, 1, &left_rank_y, &right_rank_y);
    MPI_Cart_shift(cart_comm, 2, 1, &left_rank_z, &right_rank_z);

    /// Non-divisible distribution
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

    try
    {
        if (forcing_flag)
        {
            initialise_local_forcing_from_file(problem, forcing_filename);
        }
        else
        {
            initialise_local_test_case(problem, test);
        }
    }
    catch (const exception &e)
    {
        if (world_rank == 0)
        {
            cerr << "Initialisation error: " << e.what() << endl;
        }

        delete[] problem.f;
        delete[] problem.u;
        delete[] problem.u_new;

        MPI_Comm_free(&cart_comm);
        MPI_Finalize();
        return 1;
    }

    double t0 = MPI_Wtime();
    const double final_residual = solve_poisson_cuda(problem, epsilon);
    double total_time = MPI_Wtime() - t0;

    double max_total_time = 0.0;
    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, problem.comm);

    write_solution_mpi("solution.txt", problem, problem.comm);

    if (world_rank == 0)
    {
        cout << "Final residual: " << final_residual << endl;
        cout << "Total solve time: " << max_total_time << " s" << endl;
    }

    delete[] problem.f;
    delete[] problem.u;
    delete[] problem.u_new;

    MPI_Comm_free(&problem.comm);
    MPI_Finalize();
    return 0;
}
