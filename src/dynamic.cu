#include <algorithm>
#include <limits>
#include <stack>

#include "../include/dynamic.h"
#include "../include/kernels.h"

// Simple functions to subtract, add, multiply and divide
double subtract(double a, double b){
    return a-b;
}

__device__ double subtract_gpu(double a, double b){
    return b - a;
}

double add(double a, double b){
    return a+b;
}

__device__ double add_gpu(double a, double b){
    return a + b;
}

double multiply(double a, double b){
    return a*b;
}

__device__ double multiply_gpu(double a, double b){
    return a * b;
}

double divide(double a, double b){
    return a/b;
}

__device__ double divide_gpu(double a, double b){
    return a/b;
}

double cos(double a, double b){
    return cos(a);
}

__device__ double cos_gpu(double a, double b){
    return cos(a);
}

// We assume that we have already removed unnecessary spaces and such from 
// our eqn_string
EqnNode parse_eqn(Grid &par, std::string eqn_string){

    // boolean value iff first minus
    bool minus = false;

    // Because this will be called recursively, we need to return if the string
    // length is 0
    if (eqn_string.length() == 0){
        std::cout << "There's nothing here!" << '\n';
        exit(1);
    }

    // vector of all possibe mathematical operators (not including functions)
    std::vector<std::string> moperators(4);
    moperators = {
        "-", "+", "/", "*"
    };

    // we need a map for these operators
    typedef double (*functionPtr_mop)(double, double);
    std::unordered_map<char, functionPtr_mop> moperator_map;
    moperator_map['-'] = subtract;
    moperator_map['+'] = add;
    moperator_map['*'] = multiply;
    moperator_map['/'] = divide;

    // And another vector for brackets of various types which indicate recursive
    // parsing of the equation
    std::vector<char> mbrackets;
    mbrackets = {
        '(', '[', ']', ')'
    };

    // vector of all possible mathematical functions... more to come
    std::vector<std::string> mfunctions(5);
    mfunctions = {
        "sin", "cos", "exp", "tan", "erf", "sqrt"
    };

    // We also need a specific map for the functions above
    typedef double (*functionPtr)(double, double);
    std::unordered_map<std::string, functionPtr> mfunctions_map;
    mfunctions_map["cos"] = cos;
/*
    mfunctions_map["sin"] = sin;
    mfunctions_map["cos"] = cos;
    mfunctions_map["tan"] = tan;
    mfunctions_map["exp"] = exp;
    mfunctions_map["erf"] = erf;
    mfunctions_map["sqrt"] = sqrt;
*/

    // first, we need to parse the equation string and remove parentheses
    // Then we'll sort according to the math operators (mops)
    int half = mbrackets.size() / 2;
    std::stack<int> open_bra;
    std::vector<int> ignored_positions;
    for (int i = 0; i < eqn_string.size(); ++i){
        for (int j = 0; j < mbrackets.size() / 2; ++j){
            if (eqn_string[i] == mbrackets[j]){
                open_bra.push(i);
            }
        }

        // Now we need to look for the closing bracket
        for (int j = mbrackets.size()/2; j < mbrackets.size(); ++j){
            if (eqn_string[i] == mbrackets[j]){
                ignored_positions.push_back(open_bra.top());
                ignored_positions.push_back(i);
                open_bra.pop();
            }
        }
    }


    // If parentheses cover the entire expression, we 
    //    1. Remove the parentheses
    //    2. subtract 1 from bra_positions
    std::string temp_string = eqn_string;
    if (ignored_positions.size() > 0){
        if (ignored_positions[ignored_positions.size()-1] 
                == temp_string.size() - 1 &&
            ignored_positions[ignored_positions.size()-2] == 0){
            ignored_positions.erase(ignored_positions.end()-1);
            ignored_positions.erase(ignored_positions.end()-1);

            eqn_string = eqn_string.substr(1, eqn_string.size() - 2);
    
            for (int i = 0; i < ignored_positions.size(); ++i){
                --ignored_positions[i];
            }
        }
        temp_string = eqn_string;
    
        // Now we remove the parentheses from the eqn_string
        int offset = 0;
        std::vector<int> temp_positions = ignored_positions;
        for (int i = 0; i < temp_positions.size(); i += 2){
            temp_string.erase(temp_positions[i],
                              temp_positions[i+1] - temp_positions[i]+1);
            for (int j = i+2; j < temp_positions.size(); ++j){
                if (temp_positions[j] > temp_positions[i]){
                    temp_positions[j] 
                        -= temp_positions[i+1] - temp_positions[i] + 1;
                }
            }
        }
    }

    // Creating the EqnNode
    EqnNode eqn_tree;

    bool only_nums = 
        (temp_string.find_first_not_of("0123456789") 
            == std::string::npos);
    if (only_nums){
        eqn_tree.val = atof(temp_string.c_str());
        return eqn_tree;
    }
    else if(temp_string.size() == 1){
        if(temp_string[0] == 'x'){
            eqn_tree.is_dynamic = true;
            eqn_tree.var = 'x';
            return eqn_tree;
        }
        else if(temp_string[0] == 'y'){
            eqn_tree.is_dynamic = true;
            eqn_tree.var = 'y';
            return eqn_tree;
        }
        else if(temp_string[0] == 'z'){
            eqn_tree.is_dynamic = true;
            eqn_tree.var = 'z';
            return eqn_tree;
        }
        else if(temp_string[0] == 't'){
            eqn_tree.is_dynamic = true;
            eqn_tree.var = 't';
            return eqn_tree;
        }
    }


    // We'll need to parse the equation string in reverse PEMDAS
    // So we go through the moperators, then mbrackets / mfunctions
    bool mop_found = false;
    int mop_point = 0;
    while (!mop_found){
        for (auto &mop : moperators){
            if (temp_string.find(mop) < temp_string.size()){
                mop_point = temp_string.find(mop);
                mop_found = true;
            } 
        }
        if (!mop_found){
            if(auto it = mfunctions_map.find(temp_string)
                    != mfunctions_map.end()){

                mop_point = temp_string.size()-1;
                eqn_tree.op = mfunctions_map[temp_string];
                eqn_tree.left = (EqnNode *)malloc(sizeof(EqnNode));
                eqn_tree.left[0] = parse_eqn(par, 
                                    eqn_string.substr(mop_point+1, 
                                    eqn_string.size() - mop_point-1));

            }
            else{
                eqn_tree.val = par.dval(temp_string);
            }
            return eqn_tree;
                
        }
    }

    // Now we need to find the mop_point position in the eqn_string
    // We know the temp_string and how many positions we removed and where.
    if (ignored_positions.size() > 0){
        int count = 0;
        for (int i = 0; i <= mop_point; ++i){
            for (int j = 0; j < ignored_positions.size(); j += 2){
                if (ignored_positions[j] == i){
                    count += ignored_positions[j+1] - ignored_positions[j];
                }
            }
            count++;
        }

        mop_point = count;
    }

    // Now we need to store the operator into the eqn_tree
    eqn_tree.op = moperator_map[eqn_string[mop_point]];

    // Now we need to parse the left and right banches...
    eqn_tree.left = (EqnNode *)malloc(sizeof(EqnNode));
    eqn_tree.left[0] = parse_eqn(par, eqn_string.substr(0, mop_point));

    eqn_tree.right = (EqnNode *)malloc(sizeof(EqnNode));
    eqn_tree.right[0] = parse_eqn(par, eqn_string.substr(mop_point+1, 
                                       eqn_string.size() - mop_point-1));

    return eqn_tree;
}

double evaluate_eqn(EqnNode *eqn, double x, double y, double z, 
                    double time){

    if (eqn->op == NULL){
        if (eqn->is_dynamic){
            if(eqn->var == 'x'){
                return x;
            }
            if(eqn->var == 'y'){
                return y;
            }
            if(eqn->var == 'z'){
                return z;
            }
            if(eqn->var == 't'){
                return time;
            }
        }
        else{
            return eqn->val;
        }
    }

    double val1 = evaluate_eqn(eqn->left, x, y, z, time);
    double val2 = evaluate_eqn(eqn->right, x, y, z, time);
    return eqn->op(val1, val2);

}

void tree_to_array(EqnNode eqn, EqnNode_gpu *eqn_array, int &element_num){

    eqn_array[element_num].val = eqn.val;
    eqn_array[element_num].var = eqn.var;
    eqn_array[element_num].is_dynamic = eqn.is_dynamic;

    // Now to create a map for all the functions
    typedef double (*functionPtr1)(double, double);
    typedef __device__ double (*functionPtr2)(double, double);

    std::unordered_map<functionPtr1, functionPtr2> ptr_map1, ptr_map2;
    ptr_map1[add] = add_gpu;
    ptr_map1[subtract] = subtract_gpu;
    ptr_map1[multiply] = multiply_gpu;
    ptr_map1[divide] = divide_gpu;

    ptr_map2[cos] = cos_gpu;

    bool only_left = false;
    auto it = ptr_map1.find(eqn.op);
    auto it2 = ptr_map2.find(eqn.op);
    if (it != ptr_map1.end()){
        std::cout << "found math function" << '\n';
        eqn_array[element_num].op = ptr_map1[eqn.op];
    }
    else if (it2 != ptr_map2.end()){
        std::cout << "found cos function" << '\n';
        eqn_array[element_num].op = ptr_map2[eqn.op];
        only_left = true;
    }
    else{
        eqn_array[element_num].op = NULL;
    }

    if (eqn.op == NULL){
        std::cout << "leaf node: " << eqn.val << '\t' << element_num << '\n';
        eqn_array[element_num].left = -1;
        eqn_array[element_num].right = -1;
        return;
    }
    else{
        int temp = element_num;
        element_num++;
        eqn_array[temp].left = element_num;
        tree_to_array(eqn.left[0], eqn_array, element_num);

        if (!only_left){
            element_num++;
            eqn_array[temp].right = element_num;
            tree_to_array(eqn.right[0], eqn_array, element_num);
        }
        else{
            eqn_array[temp].right = -1;
        }
    }
}

void find_element_num(EqnNode eqn_tree, int &element_num){

    element_num++;
    if (eqn_tree.op == NULL){
        return;
    }
    else{
        find_element_num(eqn_tree.right[0], element_num);
        find_element_num(eqn_tree.left[0], element_num);
    }
}

/*----------------------------------------------------------------------------//
* GPU KERNELS
*-----------------------------------------------------------------------------*/
__device__ double evaluate_eqn_gpu(EqnNode_gpu *eqn, double x, double y,
                                   double z, double time, int element_num){

    if (eqn[element_num].right < 0 &&
        eqn[element_num].left < 0){
        if (eqn[element_num].is_dynamic){
            if(eqn[element_num].var == 'x'){
                return x;
            }
            if(eqn[element_num].var == 'y'){
                return y;
            }
            if(eqn[element_num].var == 'z'){
                return z;
            }
            if(eqn[element_num].var == 't'){
                return time;
            }
        }
        else{
            return eqn[element_num].val;
        }
    }

    double val1;
    double val2;
    if (eqn[element_num].left > 0){
        val1 = evaluate_eqn_gpu(eqn, x, y, z, time, eqn[element_num].left);
    }
    else{
        val1 = 0;
    }

    if (eqn[element_num].right > 0){
        val2 = evaluate_eqn_gpu(eqn, x, y, z, time, eqn[element_num].right);
    }
    else{
        val2 = 0;
    }

    //return multiply_gpu(val1, val2);
    return eqn[element_num].op(val1, val2);

}

__global__ void find_field(double *field, double dx, double dy, double dz, 
                           double time, EqnNode_gpu *eqn){
    int gid = getGid3d3d();
    int xid = blockIdx.x*blockDim.x + threadIdx.x;
    int yid = blockIdx.y*blockDim.y + threadIdx.y;
    int zid = blockIdx.z*blockDim.z + threadIdx.z;

    field[gid] = evaluate_eqn_gpu(eqn, dx*xid, dy*yid, dz*zid, time, 0);
}

__global__ void zeros(double *field, int n){
    int xid = blockDim.x*blockIdx.x + threadIdx.x;

    if (xid < n){
        field[xid] = 0;
    }

}
