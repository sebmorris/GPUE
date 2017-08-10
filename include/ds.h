///@endcond
//##############################################################################
/**
 *  @file    ds.h
 *  @author  Lee J. O'Riordan (mlxd)
 *  @date    12/11/2015
 *  @version 0.1
 *
 *  @brief Dastructure for simulation runtime parameters
 *
 *  @section DESCRIPTION
 *  This file keeps track of and generates a parameter file (INI format) of the
 *	simulation parameters used. The resulting file is read in by the Python
 *	post-proc/analysis functions.
 */
 //#############################################################################

#ifndef DS_H
#define DS_H
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <typeinfo>
#include <cassert>
#include <iostream>

/*----------------------------------------------------------------------------//
* CLASSES
*-----------------------------------------------------------------------------*/

struct pos{
    double x, y, z;
};

/**
 * @brief       Class to hold the variable map and grid information
 * @ingroup     data
 */
// NOTE: This is necessary if we ever want to do dynamic grid manipulation.
// NOTE: I do not like this integer double split for retrieval. Find solution.
// NOTE: Consider changing unordered maps to switches for performance
// NOTE: Add FileIO to public functions
class Grid{
    // Here we keep our variable map (unordered for performance)
    // and also grid information. Note that dx dy, and dz are in param_double
    private:
        std::unordered_map<std::string, int> param_int;
        std::unordered_map<std::string, double> param_double;
        std::unordered_map<std::string, double*> param_dstar;
        std::unordered_map<std::string, bool> param_bool;
        std::unordered_map<std::string, cufftDoubleComplex*> sobel;
        std::unordered_map<std::string, std::string> param_string;
        std::string data_dir;

        // List of all strings for parsing into the appropriate param map
        // 1 -> int, 2 -> double, 3 -> double*
        std::unordered_map<std::string, int> id_list;

    // Here we keep the functions to store variables and access grid data
    public:
        dim3 grid, threads;

        // placing grid parameters in public for now
        double *x, *y, *z, *xp, *yp, *zp;

        // Function to store sobel_fft operators into the sobel map
        void store(std::string id, cufftDoubleComplex* d2param);

        // Function to store integer into param_int
        void store(std::string id, int iparam);

        // Function to store double into param_double
        void store(std::string id, double dparam);

        // Function to store double* into param_dstar
        void store(std::string id, double *dsparam);

        // Function to store bool into param_bool
        void store(std::string id, bool bparam);

        // Function to store string into data_dir
        void store(std::string id, std::string sparam);

        // Function to retrieve integer value from param_int
        int ival(std::string id);

        // Function to retrieve double value from param_double
        double dval(std::string id);

        // Function to retrieve double star values from param_dstar
        double *dsval(std::string id);

        // Function to retrieve bool from param_bool
        bool bval(std::string id);

        // Fucntion to retrieve string from data_dir
        std::string sval(std::string id);

        // Function to call back the sobel operators
        cufftDoubleComplex *cufftDoubleComplexval(std::string id);

        // Function for file writing
        void write(std::string filename);

        // Two boolean functions to check whether a string exists in 
        // param_double or param_dstar
        bool is_double(std::string id);
        bool is_dstar(std::string id);

        // Function to print all available variables
        void print_map();

        // Key values for operators
        // Note that Vector potential only have a single string for x, y, z
        std::string Kfn, Vfn, Afn, Axfile, Ayfile, Azfile, Wfcfn;
};
typedef class Grid Grid;

/**
 * @brief       class to hold all necessary information about the operators
 * @ingroup     data
 */
class Op{
    private:
        typedef double (*functionPtr)(Grid&, Op&, int, int, int);
        std::unordered_map<std::string, double*> Op_dstar;
        std::unordered_map<std::string, const double*> Op_cdstar;
        std::unordered_map<std::string, cufftDoubleComplex*> Op_cdc;
        //K_fns.emplace("rotation_K", rotation_K);
        //V_fns["harmonic_V"] = harmonic_V;
        // double *V, *V_opt, *K, *xPy, *yPx, *xPy_gpu, *yPx_gpu;
        //cufftDoubleComplex *GK,*GV_half,*GV,*EK,*EV,*EV_opt,*GxPy,*GyPx,
        //                   *ExPy,*EyPx,*EappliedField,*K_gpu,*V_gpu;

    public:
        // Functions to store data
        void store(std::string id, const double *data);
        void store(std::string id, double *data);
        void store(std::string id, cufftDoubleComplex *data);

        // Functions to retrieve data
        double *dsval(std::string id);
        cufftDoubleComplex *cufftDoubleComplexval(std::string id);

        // Map for function pointers and keys K and V
        functionPtr K_fn;
        functionPtr V_fn;
        functionPtr Ax_fn;
        functionPtr Ay_fn;
        functionPtr Az_fn;

        void set_K_fn(std::string id);
        void set_V_fn(std::string id);
        void set_A_fns(std::string id);

        // Function to set K and V function pointers

};
typedef class Op Op;

/**
 * @brief       class to hold all necessary information about the wavefunction
 * @ingroup     data
 */
class Wave{
    private:
        typedef cufftDoubleComplex (*functionPtr)(Grid&, double, int, int, int);
        std::unordered_map<std::string, double*> Wave_dstar;
        std::unordered_map<std::string, cufftDoubleComplex*> Wave_cdc;
        //double *Energy, *energy_gpu, *r, *Phi, *Phi_gpu;
        //cufftDoubleComplex *wfc, *wfc0, *wfc_backup, *wfc_gpu, *par_sum;
    public:

        // functions to store data
        void store(std::string id, double *data);
        void store(std::string id, cufftDoubleComplex *data);

        // Functions to retrieve data
        double *dsval(std::string id);
        cufftDoubleComplex *cufftDoubleComplexval(std::string id);

        functionPtr Wfc_fn;
        void set_wfc_fn(std::string id);
};
typedef class Wave Wave;

/*----------------------------------------------------------------------------//
* AUX
*-----------------------------------------------------------------------------*/
// I didn't know where to place these functions for now, so the'll be here
cufftHandle generate_plan_other2d(Grid &par);
cufftHandle generate_plan_other3d(Grid &par, int axis);

void set_fns(Grid &par, Op &opr, Wave &wave);

#endif
