#define _USE_MATH_DEFINES

#include <boost/program_options.hpp>
#include <mpi.h>
#include "poisson_solver_mpi.h"
#include "problem_setup.h"
#include "poisson_cuda.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdlib>

using namespace std;
namespace po = boost::program_options;

/*************************************************************************************
 * @brief Broadcast a string from the root MPI rank to all ranks in a communicator.
 *
 * The root rank first broadcasts the length of the string, after which all
 * ranks resize their local string buffer and receive the character data.
 *
 * @param s String to broadcast. On non-root ranks this is overwritten with the
 *          received value.
 * @param root Rank that owns the original string.
 * @param comm MPI communicator across which the string is broadcast.
 *************************************************************************************/
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

/***************************************************************************************
 * @brief Compute the number of grid points assigned to one process along a dimension.
 *
 * This supports non-uniform domain decomposition when the global number of
 * points is not exactly divisible by the number of processes. The first
 * `remainder` process coordinates receive one extra point.
 *
 * @param global_n Total number of grid points in the global dimension.
 * @param Pdim Number of processes along the dimension.
 * @param coord Cartesian coordinate of the current process in that dimension.
 * @return Number of grid points owned locally in the given dimension.
 ***************************************************************************************/
static int distribute(int global_n, int Pdim, int coord)
{
    const int base = global_n / Pdim;
    const int remainder = global_n % Pdim;
    return base + (coord < remainder ? 1 : 0);
}

/************************************************************************************
 * @brief Compute the starting global index for a process in a 1D decomposition.
 *
 * The returned index accounts for the non-uniform distribution produced by
 * @ref distribute, where earlier ranks may own one additional point.
 *
 * @param global_n Total number of grid points in the global dimension.
 * @param Pdim Number of processes along the dimension.
 * @param coord Cartesian coordinate of the current process in that dimension.
 * @return Global starting index of the local subdomain in that dimension.
 ************************************************************************************/
static int local_start(int global_n, int Pdim, int coord)
{
    const int base = global_n / Pdim;
    const int remainder = global_n % Pdim;
    return coord * base + min(coord, remainder);
}

/**********************************************************************************
 * @brief Initialise an analytic test problem on the local MPI subdomain.
 *
 * For each owned interior point, the forcing term is evaluated using the
 * selected test case. If the point lies on the global boundary, the exact
 * solution is also applied as a Dirichlet boundary condition.
 *
 * @param problem Local problem data describing the MPI subdomain, storage, and
 *                global grid dimensions.
 * @param test_case Identifier of the analytic test case to initialise.
 **********************************************************************************/
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

/****************************************************************************************
 * @brief Initialise the forcing term from an input file on the local MPI subdomain.
 *
 * Each rank reads the same forcing file, validates its dimensions against the
 * global problem size, and keeps only the entries belonging to its own local
 * block. Homogeneous Dirichlet boundary conditions are imposed on global
 * boundary points.
 *
 * @param problem Local problem data describing the MPI subdomain, storage, and
 *                global grid dimensions.
 * @param filename Path to the forcing input file.
 *
 * @throws std::runtime_error If the file cannot be opened, if the header cannot
 *         be read, if dimensions do not match the global problem, or if any
 *         data point is invalid.
 ****************************************************************************************/
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

/*********************************************************************************
 * @brief Gather the distributed solution and write it to a single output file.
 *
 * Each rank packs its owned interior values and sends them to rank 0. The root
 * rank reconstructs the full global solution array and writes the result in
 * text format together with the corresponding spatial coordinates.
 *
 * Output file format:
 * - First line: `Nx Ny Nz`
 * - Remaining lines: `x y z u`
 *
 * @param filename Name of the output file to write.
 * @param problem Local problem data for the current MPI rank.
 * @param comm MPI communicator containing all participating ranks.
 *
 * @throws std::runtime_error If the output file cannot be opened on rank 0.
 *
 * @note This routine uses explicit point-to-point communication with
 *       `MPI_Send` and `MPI_Recv`.
 ********************************************************************************/
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

/***********************************************************************************
 * @brief Entry point for the MPI-based 3D Poisson solver.
 *
 * This program solves a 3D Poisson problem on a Cartesian grid using MPI-based
 * domain decomposition. The problem can be initialised either from a built-in
 * analytic test case or from an external forcing file.
 *
 * High-level workflow:
 * - Initialise MPI
 * - Parse command-line options on rank 0
 * - Validate inputs and broadcast configuration
 * - Create a 3D Cartesian communicator
 * - Decompose the global domain among MPI ranks
 * - Allocate local arrays including halo cells
 * - Initialise the forcing term and boundary values
 * - Solve the Poisson equation using the MPI solver
 * - Gather and write the final solution
 * - Free resources and finalise MPI
 *
 * Supported command-line options:
 * - `--help`               Print available options
 * - `--test <1-5>`         Select built-in analytic test case
 * - `--forcing <file>`     Read forcing function from file
 * - `--Nx <int>`           Number of grid points in x (test mode only)
 * - `--Ny <int>`           Number of grid points in y (test mode only)
 * - `--Nz <int>`           Number of grid points in z (test mode only)
 * - `--Px <int>`           Number of MPI processes in x
 * - `--Py <int>`           Number of MPI processes in y
 * - `--Pz <int>`           Number of MPI processes in z
 * - `--epsilon <double>`   Convergence threshold for the residual
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * @return `0` on success, non-zero on failure.
 **********************************************************************************/
int main(int argc, char *argv[])
{

    /// Initialise MPI
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
        /// Define Program Options
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

        /******************************************************************************************************
         * Command Line Argument input criteria
         *
         * Defines the 5 boolean types for:
         *
         * 1. --test
         * 2. --forcing
         * 3. --User entered Nx
         * 4. --User entered Ny
         * 5. --User entered Nz
         * 6. --User entered Px, Py, Pz multiplies to same as world_rank
         *
         * Checks the following:
         *
         * 1. Either --forcing or --test should be provided, not both
         * 2. Only if --test provided, then --Nx, --Ny, --Nz can be specified
         * 3. If --test is given, only options 1-5 is allowed
         * 4. if --forcing specified, reads the forcing function using values of Nx Ny Nz from the file
         * 5. if Px, Py, Pz do not multiply to be world_rank, error thrown
         ******************************************************************************************************/

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

    /***************************************************************************************
     * @brief MPI Process.
     * MPI Process begins here:
     * Each iteration requires communication.
     * Each process gets Nx/Px, Ny/Py, Nz/Pz size
     *
     * Boundary points use halo cells to compute so as to not constantly query neighbours
     *
     * halo | interior | halo - 1D representation of MPI split
     *
     * +x --> right process
     * -x --> left process
     * +y --> front process
     * -y --> back process
     * +z --> top process
     * -z --> bottom process
     *
     * For example (+x plane):
     * SEND PLANE       |       RECEIVE PLANE
     * ------------------------------------------
     * i = local_Nx     |       i = local_Nx + 1
     *
     * ************************************************************************************
     * Workflow:
     * 1. Exchange halos
     * 2. compute interial update
     * 3. compute local residual
     * 4. MPI_All reduce residual
     **************************************************************************************/

    /**************************************************************************************
     * @brief Set up the Cartesian MPI process topology and local domain.
     *
     * The global grid is decomposed into a 3D Cartesian process grid of size
     * `Px x Py x Pz`. Each rank stores its interior region together with halo
     * cells used for neighbour communication during iterative updates.
     *
     * Solver workflow per iteration:
     * - Exchange halo regions with neighbouring ranks
     * - Update local interior values
     * - Compute local residual
     * - Perform a global reduction of the residual
     **************************************************************************************/

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

    /*********************************************************************************
     * @brief Solve the Poisson equation on the distributed domain.
     *
     * The linear system is solved iteratively using the MPI implementation of the
     * Poisson solver until the residual falls below the user-specified threshold.
     *********************************************************************************/
    double t0 = MPI_Wtime();
    const double final_residual = solve_poisson_mpi(problem, epsilon);
    double total_time = MPI_Wtime() - t0;

    double max_total_time = 0.0;
    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    if (world_rank == 0)
    {
        cout << "Total solve time: " << max_total_time << " s" << endl;
    }

    write_solution_mpi("solution.txt", problem, cart_comm);

    if (world_rank == 0)
    {
        cout << "Nx=" << Nx
             << " Ny=" << Ny
             << " Nz=" << Nz
             << " Px=" << Px
             << " Py=" << Py
             << " Pz=" << Pz
             << " Ranks=" << (Px * Py * Pz)
             << " TotalSolveTime=" << max_total_time
             << " Residual=" << final_residual
             << endl;
    }

    delete[] problem.f;
    delete[] problem.u;
    delete[] problem.u_new;

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
