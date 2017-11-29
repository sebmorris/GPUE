#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <string>
#include "../include/ds.h"

struct EqnNode{
    double val;

    EqnNode *left, *right;

    //typedef void (*functionPtr)(EqnNode *, int, int, int, double);
    typedef double (*functionPtr)(double, double);
    functionPtr op;
};

EqnNode parse_eqn(Grid &par, std::string eqn_string);

void evaluate_eqn(EqnNode eqn, int xid, int yid, int zid, double time);
void evaluate_eqn_gpu(EqnNode eqn, int xid, int yid, int zid, double time);

void allocate_eqn(EqnNode *eqn, EqnNode *eqn_gpu);

#endif
