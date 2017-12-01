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

EqnNode parse_eqn(Grid &par, std::string eqn_string);

double evaluate_eqn(EqnNode eqn, double x, double y, double z, 
                    double time);
void evaluate_eqn_gpu(EqnNode eqn, double x, double y, double z, 
                      double time);

void allocate_eqn(EqnNode *eqn, EqnNode *eqn_gpu);

#endif
