#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <string>
#include "../include/ds.h"

EqnNode parse_eqn(Grid &par, std::string eqn_string, std::string val_str);
void find_element_num(EqnNode eqn_tree, int &element_num);

double evaluate_eqn(EqnNode *eqn, double x, double y, double z, 
                    double time);

void tree_to_array(EqnNode eqn, EqnNode_gpu *eqn_array, int &element_num);

void allocate_equation(EqnNode_gpu *eqn_cpu, EqnNode_gpu *eqn_gpu, int n);

void parse_param_file(Grid &par);

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
