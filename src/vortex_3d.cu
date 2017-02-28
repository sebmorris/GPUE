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

#include "../include/vortex_3d.h"
#include "../include/kernels.h"

// Here, we will need 3 different convolution filters: row, col, depth
// these functions will only be used in this file (for now)
__device__ void convolve_row(double* density, double* edges, double* kernel,
                             int width, int height, int depth){
}


__device__ void convolve_col(double* density, double* edges, double* kernel,
                             int width, int height, int depth){
}

__device__ void convolve_depth(double* density, double* edges, double* kernel,
                               int width, int height, int depth){
}

// Kernel to transform a wavefunction to a field of edges
__global__ void find_edges(double2* wfc, double* density, double* edges){

    // first, we need to set the threading
    unsigned int gid = getGid3d3d();

    // Now we need to create the wfc density
    density[gid] = complexMagnitudeSquared(wfc[gid]);

    // Now we should do the convolutions, note that we simply need to
    // pass the convolved "sum" around. It should work out just fine (I think)

    // Defining the sobel kernel;
    double kernel_tri[3];
    double kernel_div[3];

    kernel_tri[0] = 1;
    kernel_tri[1] = 2;
    kernel_tri[2] = 1;

    kernel_div[0] = -1;
    kernel_div[1] = 0;
    kernel_div[2] = 1;

    // Note: The 256, 256 is arbitrarily set right now
    convolve_row(density, edges, kernel_tri, 256, 256, 256);
    convolve_col(density, edges, kernel_div, 256, 256, 256);
    convolve_depth(density, edges, kernel_div, 256, 256, 256);
}
/*
__device__ void convolve_row(double* density, double* edges, double* kernel,
                             int width, int height, int depth){

    // These definitions are somewhat arbitrary for now. 
    // Kernel is the Sobel filter, so radius 1, right?
    int kernel_radius = 1;
    int kernel_radius_aligned = 1;
    int tile_width = 128;

    // this will need to be updated to compile. 
    // The array size needs to be constant
    __shard__ float data[2*kernel_radius + tile_width];

    // Defining apron limits with respect to starting row
    const int tile_start  = blockIdx.x / tile_width;
    const int tile_end    = tile_start + tile_width + 1;
    const int apron_start = tile_start - kernel_radius;
    const int apron_end   = tile_end + kernel_radius;

    // Clamps for limits according to resolution limits
    // I don't know if I'm allowed to se std functions here...
    const int tile_end_clamp    = std::min(tile_end, width - 1);
    const int apron_start_clamp = std::min(apron_start, 0);
    const int apron_end_clamp   = std::min(apron_end, width - 1);

    const int row_start = blockIdx.y * width;

    const int apron_start_aligned = tile_start - kernel_radius_aligned;

    const int load_pos = apron_start_aligneda + threadIdx.x;

    if (load_pos >= apron_start){
        const int smem_pos = load_pos - apron_start;

        if (load_pos >= apron_start_clamped && load_pos <= apron_end_clamped){
            data[smem_pos] = density[row_start + load_pos];
        }
        else{
            data[smem_pos] = 0;
        }
    }

    __syncthreads();

    const int write_pos = tile_start + threadIdx.x;

    if (write_pos <= tile_end_clamp){
        const int smem_pos = write_pos - apron_Start;
        float sum = 0;

        sum += data[smem_pos -1] * kernel[0];
        sum += data[smem_pos] * kernel[1];
        sum += data[smem_pos +1] * kernel[2];

        edges[row_start + write_pos] = sum;
    }
}
*/
