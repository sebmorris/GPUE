///@endcond
//##############################################################################
/**
 *  @file    split_op.h
 *  @author  Lee J. O'Riordan (mlxd)
 *  @date    12/11/2015
 *  @version 0.1
 *
 *  @brief Host and device declarations for simulations.
 *
 *  @section DESCRIPTION
 *  These functions and variables are necessary for carrying out the GPUE
 *	simulations. This file will be re-written in an improved form in some
 *	future release.
 */
//##############################################################################

#ifndef SPLIT_OP_H
#define SPLIT_OP_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <ctype.h>
#include <getopt.h>
#include "tracker.h"
#include "ds.h"
#include "constants.h"

#ifdef __linux
	#include<omp.h>
#elif __APPLE__
	//printf("OpenMP support disabled due to Clang/LLVM being behind the trend.",);
#endif

 /**
 * @brief	Checks if CUDA operation has succeeded. Prints to stdout.
 * @ingroup	data
 * @param	result Function result code of CUDA operation
 * @param	c Descriptor of CUDA operation
 * @return	0 for success. See CUDA failure codes in cuda.h for other values.
 */
int isError(int result, char* c); //Checks to see if an error has occurred.

/**
* @brief	Performs parallel summation and renormalises the wavefunction
* @ingroup	data
* @param	gpuWfc GPU memory location for wavefunction
* @param	Parameter class
* @return	0 for success. See CUDA failure codes in cuda.h for other values.
*/
void parSum(double2* gpuWfc, Grid &par);

/**
* @brief        Performs parallel summation and renormalises the wavefunction
* @ingroup      data
* @param        gpuWfc GPU memory location for wavefunction
* @param        gpuParSum wavefunction density
* @param        Parameter class
* @return       0 for success. See CUDA failure codes in cuda.h for other values.
*/
void parSum(double* gpuWfc, double *gpuParSum, Grid &par);

/**
* @brief	Creates the optical lattice to match the vortex lattice constant
* @ingroup	data
* @param	centre Central vortex in condensate
* @param	V Trapping potential for condensate
* @param	vArray Vortex location array
* @param	theta_opt Offset angle for optical lattice relative to vortex lattice
* @param	intensity Optical lattice amplitude
* @param	v_opt Optical lattice memory address location
* @param	x X grid array
* @param	y Y grid array
* @param	Parameter class
*/
void optLatSetup(const std::shared_ptr<Vtx::Vortex> centre, const double* V,
                 std::vector<std::shared_ptr<Vtx::Vortex>> &vArray, double theta_opt,
                 double intensity, double* v_opt, const double *x, const double *y,
                 Grid &par);

/**
* @brief	Calculates the energy of the condensate.
* @ingroup	data
* @param	gpuWfc Device wavefunction array
* @param	Parameter class
* @return	$\langle \Psi | H | \Psi \rangle$
*/
double energy_calc(Grid &par, double2* wfc);

#endif
