#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <string>
#include "../include/ds.h"

struct EqnNode{
    double val;
    bool is_dynamic = false;
    char var = '0';

    EqnNode *left, *right;

    //typedef void (*functionPtr)(EqnNode *, int, int, int, double);
    typedef double (*functionPtr)(double, double);
    functionPtr op = NULL;
};

// For ease of allocating, we will store the entire GPU tree into an array that
// simply connects elements inside
struct EqnNode_gpu{
    double val;
    bool is_dynamic = false;
    char var = '0';

    int left = -1;
    int right = -1;

    //typedef void (*functionPtr)(EqnNode *, int, int, int, double);
    typedef __device__ double (*functionPtr)(double, double);
    functionPtr op = NULL;
};

EqnNode parse_eqn(Grid &par, std::string eqn_string);
void find_element_num(EqnNode eqn_tree, int &element_num);

double evaluate_eqn(EqnNode *eqn, double x, double y, double z, 
                    double time);
__device__ double evaluate_eqn_gpu(EqnNode eqn, double x, double y, double z, 
                                   double time, int &element_num);

void tree_to_array(EqnNode eqn, EqnNode_gpu *eqn_array, int &element_num);

#endif
