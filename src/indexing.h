#pragma once
#include <cstddef>

/**********************************************************************************************************************************************************************
 * #pragma once tells the compiler to only include this header file once per translation unit.
 * This is done to prevent redefinition errors and duplicate function or class definitions.
 *
 * It is a compile time protection against multiple inclusion which improves:
 *
 * 1. Safety
 * 2. Compile Speed
 *
 * inline indexing maps from 3D indices to 1D memory layout.
 *
 * Conceptually speaking, data is stored as:
 * u[i][j][k]
 *
 * Hence, a mapping (i, j, k) --> flat index is required.
 *
 * The current indexing is row major ordering where:
 *
 * i --> slowest
 * j --> middle
 * k --> fastest
 *
 * Hence, memory looks like [i][j][k] and index = i * (Ny * Nz) + j * (Nz) + k
 *
 * The function is declared as 'inline' to elimiate function call overhead,
 * as it is invoked frequently within performance critical loops.
 *
 * @param i     = Index in the x direction
 * @param j     = Index in the y direction
 * @param z     = Index in the z direction
 * @param Ny    = Number of grid points in y direction (including halos)
 * @param Nz    = Number of grid points in z direction (including halos)
 * @return Flattened 1D index corresponding to (i, j, k)
 *
 **********************************************************************************************************************************************************************/

inline std::size_t idx(int i, int j, int k, int Ny, int Nz)
{
    return static_cast<std::size_t>(i) * Ny * Nz + static_cast<std::size_t>(j) * Nz + static_cast<std::size_t>(k);
}