CXX = g++
MPICXX = mpicxx
NVCC = nvcc

CXXFLAGS = -std=c++17 -Wall -O2
NVCCFLAGS = -O2 -ccbin $(MPICXX)
LDLIBS = -lboost_program_options -lblas

TARGET = poisson
MPI_TARGET = poisson-mpi
TEST_TARGET = test_solver
MPI_TEST_TARGET = test_solver_mpi
CUDA_TARGET = poisson-cuda

SRC = src/main.cpp src/poisson_solver.cpp src/problem_setup.cpp
OBJ = $(SRC:.cpp=.o)

MPI_SRC = src/main_mpi.cpp src/poisson_solver_mpi.cpp src/problem_setup.cpp
MPI_OBJ = $(MPI_SRC:.cpp=.o)

TEST_SRC = tests/test_solver.cpp src/poisson_solver.cpp src/problem_setup.cpp
TEST_OBJ = $(TEST_SRC:.cpp=.o)

MPI_TEST_SRC = tests/test_solver_mpi.cpp src/poisson_solver_mpi.cpp src/problem_setup.cpp
MPI_TEST_OBJ = $(MPI_TEST_SRC:.cpp=.o)

CUDA_SRC = src/main_cuda.cpp src/problem_setup.cpp src/poisson_cuda.cu
CUDA_OBJ = src/main_cuda.o src/problem_setup.o src/poisson_cuda.o

$(MPI_TEST_TARGET): $(MPI_TEST_OBJ)
	$(MPICXX) $(CXXFLAGS) -o $(MPI_TEST_TARGET) $(MPI_TEST_OBJ) $(LDLIBS)

$(TEST_TARGET): $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $(TEST_TARGET) $(TEST_OBJ) $(LDLIBS)

DOXYFILE = Doxyfile

.PHONY: all default clean tests tests-mpi doc

all: poisson poisson-mpi

default: all

poisson: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ) $(LDLIBS)

poisson-mpi: $(MPI_OBJ)
	$(MPICXX) $(CXXFLAGS) -o $(MPI_TARGET) $(MPI_OBJ) $(LDLIBS)

poisson-cuda: $(CUDA_OBJ)
	$(NVCC) -ccbin $(MPICXX) -o $(CUDA_TARGET) $(CUDA_OBJ) $(LDLIBS)

tests: $(TEST_TARGET)
	./$(TEST_TARGET)

tests-mpi: $(MPI_TEST_TARGET)
	mpirun -np 4 ./$(MPI_TEST_TARGET)

doc:
	doxygen $(DOXYFILE)

clean:
	rm -f $(OBJ) $(MPI_OBJ) $(TEST_OBJ) $(CUDA_OBJ) $(TEST_OBJ) $(MPI_TEST_OBJ) $(TARGET) $(MPI_TARGET) $(TEST_TARGET) $(MPI_TEST_TARGET) $(CUDA_TARGET) solution.txt
	rm -rf docs

src/main.o: src/main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

src/poisson_solver.o: src/poisson_solver.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

src/problem_setup.o: src/problem_setup.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

tests/test_solver.o: tests/test_solver.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

tests/test_solver_mpi.o: tests/test_solver_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

src/main_mpi.o: src/main_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

src/poisson_solver_mpi.o: src/poisson_solver_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

src/poisson_cuda.o: src/poisson_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

src/main_cuda.o: src/main_cuda.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@
