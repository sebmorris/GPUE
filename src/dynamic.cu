#include <algorithm>
#include <limits>

#include "../include/dynamic.h"

// Simple functions to subtract, add, multiply and divide
double subtract(double a, double b){
    return a-b;
}

double add(double a, double b){
    return a+b;
}

double multiply(double a, double b){
    return a*b;
}

double divide(double a, double b){
    return a/b;
}
// We assume that we have already removed unnecessary spaces and such from 
// our eqn_string
EqnNode parse_eqn(Grid &par, std::string eqn_string){
    // boolean value iff first minus
    bool minus = false;

    //std::cout << equation << '\n';

    // Because this will be called recursively, we need to return if the string
    // length is 0
    if (eqn_string.length() == 0){
        std::cout << "There's nothing here!" << '\n';
        //return;
    }

    // vector of all possibe mathematical operators (not including functions)
    std::vector<std::string> moperators(4);
    moperators = {
        "-", "+", "/", "*"
    };

    // we need a map for these operators
    typedef double (*functionPtr_mop)(double, double);
    std::unordered_map<std::string, functionPtr_mop> moperator_map;
    moperator_map["-"] = subtract;
    moperator_map["+"] = add;
    moperator_map["*"] = multiply;
    moperator_map["/"] = divide;

    // And another vector for brackets of various types which indicate recursive
    // parsing of the equation
    std::vector<std::string> mbrackets;
    mbrackets = {
        "(", "[", "]", ")"
    };

    // vector of all possible mathematical functions... more to come
    std::vector<std::string> mfunctions(5);
    mfunctions = {
        "sin", "cos", "exp", "tan", "erf", "sqrt"
    };

    // We also need a specific map for the functions above
    typedef double (*functionPtr)(double);
    std::unordered_map<std::string, functionPtr> mfunctions_map;
    mfunctions_map["sin"] = sin;
    mfunctions_map["cos"] = cos;
    mfunctions_map["tan"] = tan;
    mfunctions_map["exp"] = exp;
    mfunctions_map["erf"] = erf;
    mfunctions_map["sqrt"] = sqrt;

    // first, we need to parse the equation string and remove parentheses
    // Then we'll sort according to the math operators (mops)
    std::string temp_string = eqn_string;
    int half = mbrackets.size() / 2;
    std::vector<int> open_bra, close_bra;
    for (int i = 0; i < mbrackets.size(); ++i){
        if (i < half){
            for (int j = 0; j < eqn_string.size(); ++j){
                if (eqn_string[j] == mbrackets[i][0]){
                    open_bra.push_back(j);
                }
            }
        }

        // Now we need to look for the closing bracket
        if (i >= half){
            for (int j = 0; j < eqn_string.size(); ++j){
                if (eqn_string[j] == mbrackets[i][0]){
                    close_bra.push_back(j);
                }
            }
        }
    }


    // Sorting should get all the parentheses 
    std::sort(open_bra.begin(), open_bra.end());
    std::sort(close_bra.begin(), close_bra.end());

    // Finding how many brackets we can ignore for further subdivision
    int num_close = 0;
    int num_open = 0;
    int j = 0;
    int position = open_bra[0];

    std::vector<int> ignored_positions;

    // We go through all the closing brackets, then:
    //    1. Search for all opening brackets that come before it
    //    2. If number of open brackets and closing brackets are equal, 
    //       we write out and save our count.
    //    3. We then reset counts and continue
    //    4. If there are no more opening brackets, we set the position high
    for (int i = 0; i < close_bra.size(); ++i){
        while(position < close_bra[i]){
            num_open++;
            j++;
            if (j < open_bra.size()){
                position = open_bra[j];
            }
            else{
                position = std::numeric_limits<int>::max();
            }
        }

        num_close++;
        if (num_open == num_close){
            ignored_positions.push_back(open_bra[j - num_open]);
            ignored_positions.push_back(close_bra[i]);
            num_close = 0;
            num_open = 0;
        }
    }

    std::cout << "check" << '\n';
    for (int i = 0; i < ignored_positions.size(); ++i){
        std::cout << ignored_positions[i] << '\n';
    }

    // Now we remove the parentheses from the eqn_string
    int offset = 0;
    for (int i = 0; i < ignored_positions.size(); i += 2){
        temp_string.erase(ignored_positions[i]-offset,
                          ignored_positions[i+1] - ignored_positions[i]+1);
        offset += ignored_positions[i+1] - ignored_positions[i]+1;
    }

    std::cout << temp_string << '\n';

    // We'll need to parse the equation string in reverse PEMDAS
    // So we go through the moperators, then mbrackets / mfunctions
    for (auto &mop : moperators){
        
    }

/*

    // Check for parentheses
    for (auto &mbra : mbrackets){
        //std::cout << equation.substr(0,1) << '\n';
        if (equation.substr(0,1) == mbra){
            if (mbra == ")" || mbra == "]"){
                //std::cout << "could not find matching " << mbra << "!\n";
                exit(0);
            }
            else if (mbra == "("){
                int brapos = equation.find(")");
                std::string new_eqn = equation.substr(1,brapos-1);
                parse_equation(par, new_eqn, val, i, j, k);
                equation = equation.substr(brapos+1, equation.size());
            }
            else if (mbra == "["){
                int brapos = equation.find("]");
                std::string new_eqn = equation.substr(1,brapos-1);
                parse_equation(par, new_eqn, val, i, j, k);
                equation = equation.substr(brapos, equation.size());
            }
        }
    }


    // We will have values and operators, but some operators will need to 
    // recursively call this function (think exp(), sin(), cos())...
    // We now need to do some silly sorting to figure out which operator 
    // comes first and where it is
    size_t index = equation.length();
    std::string currmop = "";
    size_t moppos;
    for (auto &mop : moperators){
        moppos = equation.find(mop);
        if (moppos < equation.length()){
            if (moppos < index){ // && moppos > 0){
                currmop = mop;
                index = moppos;
            }
            //else if(moppos == 0 && mop == "-"){
                //minus = true;
                //equation = equation.substr(1,equation.size());
            //}
        }
    }

    //std::cout << currmop << '\t' << index << '\n';

    // Now we do a similar thing for the mbrackets
    // Sharing moppos from above
    for (auto &mbra : mbrackets){
        moppos = equation.find(mbra);
        if (moppos < equation.length()){
            if (moppos < index){
                currmop = mbra;
                index = moppos;
            }
        }
    }

    // Now we need to get the string we are working with...
    std::string item = equation.substr(0,index);

    // now we need to find the string in either mfunctions or par
    // First, we'll check mfunctions

    // Now we need to check to see if the string is in mfunctions
    auto it = mfunctions_map.find(item);
    if (it != mfunctions_map.end()){
        int openbracket, closebracket;
        openbracket = index;
        closebracket = equation.find(equation[openbracket]);
        std::string ineqn = equation.substr(openbracket + 1, 
                                            closebracket - 1);
        double inval = 1;
        parse_equation(par, ineqn, inval, i, j, k);
        val = mfunctions_map[item](inval);

        // now we need to parse the rest of the string...
        ineqn = equation.substr(closebracket, equation.size());
        parse_equation(par, ineqn, val, i, j, k);
    }

    // Now we need to do a similar thing for all the maps in par.
    else if (par.is_double(item)){
        val = par.dval(item);
    }
    else if (par.is_dstar(item)){
        if (item == "x" || item == "px"){
            val = par.dsval(item)[i];
        }
        if (item == "y" || item == "py"){
            val = par.dsval(item)[j];
        }
        if (item == "z" || item == "pz"){
            val = par.dsval(item)[k];
        }
    }
    else if (item.find_first_not_of("0123456789.") > item.size() &&
             item.size() > 0){
        //std::cout << item << '\n';
        val = std::stod(item);
    }
    else if (item.size() > 0){
        std::cout << "could not find string " << item << "! please use one of "
                  << "the following variables:" << '\n';
        par.print_map();
    }

    if (minus){
        val *= -1;
    }

    //std::cout << item << '\t' << currmop << '\n';

    // Now to deal with the operator at the end
    if (currmop == "+"){
        double inval = 1;
        std::string new_eqn = equation.substr(index+1,equation.size());
        //std::cout << new_eqn << '\n';
        parse_equation(par, new_eqn, inval, i, j, k);
        val += inval;
    }
    if (currmop == "-"){
        double inval = 1;
        std::string new_eqn = equation.substr(index+1,equation.size());
        //std::cout << new_eqn << '\n';
        parse_equation(par, new_eqn, inval, i, j, k);
        val -= inval;
    }
    if (currmop == "*"){
        double inval = 1;
        std::string new_eqn = equation.substr(index+1,equation.size());
        //std::cout << new_eqn << '\n';
        parse_equation(par, new_eqn, inval, i, j, k);
        val *= inval;
    }
    if (currmop == "/"){
        double inval = 1;
        std::string new_eqn = equation.substr(index+1,equation.size());
        //std::cout << new_eqn << '\n';
        parse_equation(par, new_eqn, inval, i, j, k);
        val /= inval;
    }
*/
}

void allocate_eqn(EqnNode *eqn, EqnNode *eqn_gpu){
}
