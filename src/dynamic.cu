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

    std::cout << eqn_string << '\n';

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
    int half = mbrackets.size() / 2;
    std::vector<int> open_bra, close_bra;
    for (int i = 0; i < mbrackets.size(); ++i){
        if (i < half){
            for (int j = 0; j < eqn_string.size(); ++j){
                if (eqn_string[j] == mbrackets[i]){
                    open_bra.push_back(j);
                }
            }
        }

        // Now we need to look for the closing bracket
        if (i >= half){
            for (int j = 0; j < eqn_string.size(); ++j){
                if (eqn_string[j] == mbrackets[i]){
                    close_bra.push_back(j);
                }
            }
        }
    }

 
    // If parentheses cover the entire expression, we 
    //    1. Remove the parentheses
    //    2. subtract 1 from bra_positions
    std::string temp_string = eqn_string;
    std::vector<int> ignored_positions;
    if (open_bra.size() > 0 && close_bra.size() > 0){
        if (open_bra[0] == 0 && 
            close_bra[close_bra.size() -1] == eqn_string.size() - 1){
            eqn_string = eqn_string.substr(1, eqn_string.size() - 2);
            open_bra.erase(open_bra.begin());
            close_bra.erase(close_bra.end()-1);
    
            for (int i = 0; i < close_bra.size(); ++i){
                --open_bra[i];
                --close_bra[i];
            }
        }
        temp_string = eqn_string;
    
        // Finding how many brackets we can ignore for further subdivision
        int num_close = 0;
        int num_open = 0;
        int j = 0;
        int position = open_bra[0];
    
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
    
        // Now we remove the parentheses from the eqn_string
        int offset = 0;
        for (int i = 0; i < ignored_positions.size(); i += 2){
            temp_string.erase(ignored_positions[i]-offset,
                              ignored_positions[i+1] - ignored_positions[i]+1);
            offset += ignored_positions[i+1] - ignored_positions[i]+1;
        }
    }

    // Creating the EqnNode
    EqnNode eqn_tree;

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
            bool only_nums = 
                (temp_string.find_first_not_of("0123456789") 
                    == std::string::npos);
            if (only_nums){
                eqn_tree.val = atof(temp_string.c_str());
            }
            else if(temp_string.size() == 1){
                if(temp_string[0] == 'x'){
                    eqn_tree.is_dynamic = true;
                    eqn_tree.var = 'x';
                }
                else if(temp_string[0] == 'y'){
                    eqn_tree.is_dynamic = true;
                    eqn_tree.var = 'y';
                }
                else if(temp_string[0] == 'z'){
                    eqn_tree.is_dynamic = true;
                    eqn_tree.var = 'z';
                }
                else if(temp_string[0] == 't'){
                    eqn_tree.is_dynamic = true;
                    eqn_tree.var = 't';
                }
                else{
                    eqn_tree.val = par.dval(temp_string);
                }
                
            }
            else{
                eqn_tree.val = par.dval(temp_string);
            }
            return eqn_tree;
        }
    }

    std::cout << mop_point << '\n';

    // Now we need to find the mop_point position in the eqn_string
    // We know the temp_string and how many positions we removed and where.
    if (ignored_positions.size() > 0){
        int index = 0;
        int count = 0;
        for (int i = 0; i <= mop_point; ++i){
            if (ignored_positions[index] == i){
                count += ignored_positions[index+1] - ignored_positions[index];
                if (index < ignored_positions.size() - 2){
                    index += 2;
                }
            }
            count++;
        }

        mop_point = count;
    }

    std::cout << mop_point << '\n';
    std::cout << eqn_string[mop_point] << '\n';

    // Now we need to store the operator into the eqn_tree
    eqn_tree.op = moperator_map[eqn_string[mop_point]];

    // Now we need to parse the left and right banches...
    eqn_tree.left = (EqnNode *)malloc(sizeof(EqnNode));
    eqn_tree.right = (EqnNode *)malloc(sizeof(EqnNode));
    eqn_tree.left[0] = parse_eqn(par, eqn_string.substr(0, mop_point));
    eqn_tree.right[0] = parse_eqn(par, eqn_string.substr(mop_point+1, 
                                       eqn_string.size() - mop_point-1));

    return eqn_tree;

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
