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
* GPU STACK
* Note: It is clear that a stack implementation on GPU's for tree traversal is
*       not a universally good idea; however, there is an array of these trees,
*       so there is still a need for parallelization. I feel a massively
*       parallel CPU system would be a bit better for this purpose, but we
*       cannot afford the transfer time every timestep.
*-----------------------------------------------------------------------------*/
/*

struct stack {
    void **data;
    size_t top, capacity, size;
};

stack get_stack(size_t size) {
    stack stk;

    stk.data = malloc(4 * size);
    stk.capacity = 4;
    stk.top = 0;

    return stk;
}

bool stack_empty(stack *stk){
    return (stk->top == 0);
}

void stack_push(stack *stk, void *element) {
    if (stk->top == stk->capacity) {
        stk->capacity *= 2;
        stk->data = realloc(stk->data, stk->capacity * sizeof(stk->data[0]));
    }

    stk->data[stk->top++] = element;
}

void *stack_pop(struct stack *stk) {
    if (stack_empty(stk)) {
        return NULL;
    }

    return stk->data[--stk->top];
}

void free_stack(struct stack stk) {
    free(stk.data);
}
*/

/*----------------------------------------------------------------------------//
* CLASSES
*-----------------------------------------------------------------------------*/

struct pos{
    double x, y, z;
};

typedef double (*fnPtr) (double, double);

struct EqnNode{
    double val = 0;
    bool is_dynamic = false;
    char var = '0';

    EqnNode *left, *right;

    int op_num;
    bool has_op = false;
};

// For ease of allocating, we will store the entire GPU tree into an array that
// simply connects elements inside
struct EqnNode_gpu{
    double val = 0;
    bool is_dynamic = false;
    char var = '0';

    int left = -1;
    int right = -1;

    int op_num;
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
        typedef void (*functionPtrA)(double*, double*, double*, 
                                     double,  double,  double, 
                                     double,  double,  double, 
                                     double, double, double*);
        typedef void (*functionPtrV)(double*, double*, double*, double*,
                                     double*, double*, double*, double*);
        typedef void (*functionPtrwfc)(double*, double*, double*, 
                                       double*, double, double*, double2*);
        std::unordered_map<std::string, int> param_int;
        std::unordered_map<std::string, double> param_double;
        std::unordered_map<std::string, double*> param_dstar;
        std::unordered_map<std::string, bool> param_bool;
        std::unordered_map<std::string, cufftDoubleComplex*> sobel;
        std::unordered_map<std::string, std::string> param_string;
        std::unordered_map<std::string, EqnNode_gpu*> param_ast;
        std::unordered_map<std::string, EqnNode> param_ast_cpu;

        // List of all strings for parsing into the appropriate param map
        // 1 -> int, 2 -> double, 3 -> double*
        std::unordered_map<std::string, int> id_list;

    // Here we keep the functions to store variables and access grid data
    public:
        dim3 grid, threads;

        // Map for function pointers and keys K and V
        functionPtrV V_fn;
        functionPtrA Ax_fn, Ay_fn, Az_fn;
        functionPtrwfc wfc_fn;

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

        // Function to store asts into data_dir
        void store(std::string id, EqnNode_gpu *ensparam);

        // Function to store asts into data_dir
        void store(std::string id, EqnNode astparam);

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

        // Function to call back ast
        EqnNode_gpu *astval(std::string id);

        // Function to call back ast
        EqnNode ast_cpuval(std::string id);

        // Function for file writing
        void write(std::string filename);

        // Two boolean functions to check whether a string exists in 
        // param_double or param_dstar
        bool is_double(std::string id);
        bool is_dstar(std::string id);
        bool is_ast_gpu(std::string id);
        bool is_ast_cpu(std::string id);

        // Function to print all available variables
        void print_map();

        // function to set A functions
        void set_A_fn(std::string id);

        // function to set V functions
        void set_V_fn(std::string id);

        // function to set V functions
        void set_wfc_fn(std::string id);

        // Key values for operators
        // Note that Vector potential only have a single string for x, y, z
        std::string Kfn, Vfn, Afn, Axfile, Ayfile, Azfile, Wfcfn;
};
typedef class Grid Grid;

void generate_plan_other2d(cufftHandle *plan_fft1d, Grid &par);
void generate_plan_other3d(cufftHandle *plan_fft1d, Grid &par, int axis);
void set_fns(Grid &par);

#endif
