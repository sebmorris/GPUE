/*-------------vortex_3d.cu---------------------------------------------------//
*
* Purpose: This file intends to perform a convolution of 3d data for vortex
*          recognition in the GPUE code
*
*   Notes: We will be using the window method for convolutions because it has
*          a slightly better complxity case for a separable filter
*          (which the Sobel filter definitely is!)
*
*-----------------------------------------------------------------------------*/

#ifndef VORTEX_3D_H
#define VORTEX_3D_H

//#include "kernels.h"
#include <stdio.h>

// Kernel to return vortex positions

// Kernel to return spine of edges

// We need a central kernel with just inputs and outputs

__global__ void find_edges(double2* wfc, double* density, double* edges);

// Here, we will need 3 different convolution filters: row, col, depth
// these functions will only be used in this file (for now)
__device__ void convolve_row(double* density, double* edges, double* kernel,
                             int width, int height, int depth);


__device__ void convolve_col(double* density, double* edges, double* kernel,
                             int width, int height, int depth);

__device__ void convolve_depth(double* density, double* edges, double* kernel,
                               int width, int height, int depth);

#endif
