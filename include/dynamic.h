#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <string>
#include "../include/ds.h"

typedef double (*fnPtr) (double, double);


struct EqnNode{
    double val = 0;
    bool is_dynamic = false;
    char var = '0';

    EqnNode *left, *right;

    fnPtr op = NULL;
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

EqnNode parse_eqn(Grid &par, std::string eqn_string);
void find_element_num(EqnNode eqn_tree, int &element_num);

double evaluate_eqn(EqnNode *eqn, double x, double y, double z, 
                    double time);

void tree_to_array(EqnNode eqn, EqnNode_gpu *eqn_array, int &element_num);

void allocate_equation(EqnNode_gpu *eqn_cpu, EqnNode_gpu *eqn_gpu, int n);

/*----------------------------------------------------------------------------//
* GPU KERNELS
*-----------------------------------------------------------------------------*/

__device__ double evaluate_eqn_gpu(EqnNode_gpu *eqn, double x, double y,
                                   double z, double time, int element_num);

__global__ void find_field(double *field, double dx, double dy, double dz, 
                           double time, EqnNode_gpu *eqn);

__global__ void zeros(double *field, int n);

__device__ double poly_j(int v, double x, int n);

#endif
