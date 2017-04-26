
#include "../include/ds.h"
#include "../include/operators.h"

/*----------------------------------------------------------------------------//
* AUX
*-----------------------------------------------------------------------------*/

// I didn't know where to place these functions for now, so the'll be here
cufftHandle generate_plan_other2d(Grid &par){
    // We need a number of propertied for this fft / transform
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");

    cufftHandle plan_fft1d;

    int batch = yDim;
    int rank = 1;
    int n[] = {xDim, yDim};
    int idist = 1;
    int odist = 1;
    int inembed[] = {xDim,yDim};
    int onembed[] = {xDim,yDim};
    int istride = yDim;
    int ostride = yDim;

    cufftResult result;

    result = cufftPlanMany(&plan_fft1d, rank, n, inembed, istride, 
                           idist, onembed, ostride, odist, 
                           CUFFT_Z2Z, batch);

    if(result != CUFFT_SUCCESS){
        printf("Result:=%d\n",result);
        printf("Error: Could not execute cufftPlanfft1d(%s ,%d ,%d ).\n", 
               "plan_1d", (unsigned int)xDim, (unsigned int)yDim);
        return -1;
    }

    return plan_fft1d;

}

// other plan for 3d case
// Note that we have 3 cases to deal with here: x, y, and z
cufftHandle generate_plan_other3d(Grid &par, int axis){
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int zDim = par.ival("zDim");

    cufftResult result;
    cufftHandle plan_fft1d;

    // Along first dimension (z)
    if (axis == 0){
        result = cufftPlan1d(&plan_fft1d, xDim, CUFFT_Z2Z, yDim*xDim);
/*
        int batch = xDim*yDim;
        int rank = 1;
        int n[] = {xDim, yDim, zDim};
        int idist = xDim;
        int odist = xDim;
        int inembed[] = {xDim, yDim, zDim};
        int onembed[] = {xDim, yDim, zDim};
        int istride = 1;
        int ostride = 1;
    
        result = cufftPlanMany(&plan_fft1d, rank, n, inembed, istride, 
                               idist, onembed, ostride, odist, 
                               CUFFT_Z2Z, batch);
*/
    }

    // Along second dimension (y)
    // This one is a bit complicated because of how the data is aligned
    else if (axis == 1){
        int batch = yDim;
        int rank = 1;
        int n[] = {xDim, yDim};
        int idist = 1;
        int odist = 1;
        int inembed[] = {xDim, yDim};
        int onembed[] = {xDim, yDim};
        int istride = yDim;
        int ostride = yDim;
    
        result = cufftPlanMany(&plan_fft1d, rank, n, inembed, istride, 
                               idist, onembed, ostride, odist, 
                               CUFFT_Z2Z, batch);
        //result = cufftPlan2d(&plan_fft1d, xDim, yDim, CUFFT_Z2Z);
        
    }

    // Along third dimension (x)
    else if (axis == 2){

        int batch = xDim*yDim;
        int rank = 1;
        int n[] = {xDim, yDim, zDim};
        int idist = 1;
        int odist = 1;
        int inembed[] = {xDim, yDim, zDim};
        int onembed[] = {xDim, yDim, zDim};
        int istride = xDim*yDim;
        int ostride = xDim*yDim;
    
        result = cufftPlanMany(&plan_fft1d, rank, n, inembed, istride, 
                               idist, onembed, ostride, odist, 
                               CUFFT_Z2Z, batch);
    }

    if(result != CUFFT_SUCCESS){
        printf("Result:=%d\n",result);
        printf("Error: Could not execute cufftPlan3d(%s ,%d ,%d ).\n", 
               "plan_1d", (unsigned int)xDim, (unsigned int)yDim);
        return -1;
    }

    return plan_fft1d;
}


/*----------------------------------------------------------------------------//
* GRID
*-----------------------------------------------------------------------------*/

// Function to store sobel_fft operators and stuff
void Grid::store(std::string id, cufftDoubleComplex *d2param){
    sobel[id] = d2param;
}

// Function to store integer into Grid->param_int
void Grid::store(std::string id, int iparam){
    param_int[id] = iparam;
}

// Function to store double into Grid->param_double
void Grid::store(std::string id, double dparam){
    param_double[id] = dparam;
}

// Function to store double* into param_dstar
void Grid::store(std::string id, double *dsparam){
    param_dstar[id] = dsparam;
}

// Function to store bool into param_bool
void Grid::store(std::string id, bool bparam){
    param_bool[id] = bparam;
}

// Function to store string into data_dir
void Grid::store(std::string id, std::string sparam){
    param_string[id] = sparam;
}

// Two boolean functions to check whether a string exists in 
// param_double or param_dstar
bool Grid::is_double(std::string id){
    auto it = param_double.find(id);
    if (it != param_double.end()){
        return true;
    }
    else {
        return false;
    }
}

bool Grid::is_dstar(std::string id){
    auto it = param_dstar.find(id);
    if (it != param_dstar.end()){
        return true;
    }
    else {
        return false;
    }
}

// Function to retrieve integer from Grid->param_int
int Grid::ival(std::string id){
    return param_int[id];
}

// Function to retrieve double from Grid->param_double
double Grid::dval(std::string id){
    auto it = param_double.find(id);
    if (it == param_double.end()){
        std::cout << "ERROR: could not find string " << id 
                  << " in Grid::param_double." << '\n';
        assert(it != param_double.end());
    }
    return it->second;
}

// Function to retrieve double star values from param_dstar
double *Grid::dsval(std::string id){
    auto it = param_dstar.find(id);
    if (it == param_dstar.end()){
        std::cout << "ERROR: could not find string " << id
                  << " in Grid::param_dstar." << '\n';
        assert(it != param_dstar.end());
    }
    return it->second;
}

// Function to retrieve bool values from param_bool
bool Grid::bval(std::string id){
    auto it = param_bool.find(id);
    if (it == param_bool.end()){
        std::cout << "ERROR: could not find string " << id 
                  << " in Grid::param_bool." << '\n';
        assert(it != param_bool.end());
    }
    return it->second;
}

// Function to retrieve string from data_dir
// Note: There is only one string value in the Grid struct... 
//       We must add an unordered map for strings if further strings are desired
std::string Grid::sval(std::string id){
    auto it = param_string.find(id);
    if (it == param_string.end()){
        std::cout << "ERROR: could not find string " << id 
                  << " in Grid::param_string." << '\n';
        assert(it != param_string.end());
    }
    return it->second;
}

// Function to call back the sobel operators
cufftDoubleComplex *Grid::cufftDoubleComplexval(std::string id){
    auto it = sobel.find(id);
    if (it == sobel.end()){
        std::cout << "ERROR: could not find string " << id
                  << " in Grid::sobel." << '\n';
        assert(it != sobel.end());
    }
    return it->second;
}

// Function for file writing (to replace fileIO::writeOutParam)
void Grid::write(std::string filename){
    std::ofstream output;
    output.open(filename);
    //Needed to recognise Params.dat as .ini format for python post processing
    output << "[Params]" <<"\n";

    // We simply iterate through the int and double param maps
    for (auto item : param_double){
        output << item.first << "=" << item.second << '\n';
        std::cout << item.first << "=" << item.second << '\n';
    }

    for (auto item : param_int){
        output << item.first << "=" << item.second << '\n';
        std::cout << item.first << "=" << item.second << '\n';
    }

    output.close();
}

// Function to print all available variables
void Grid::print_map(){
    for (auto item : param_double){
        std::cout << item.first << '\n';
    }
    for (auto item : param_dstar){
       std::cout << item.first << '\n';
    }
}

/*----------------------------------------------------------------------------//
* CUDA
*-----------------------------------------------------------------------------*/

// Functions to store data in Cuda class
void Cuda::store(std::string id, cudaError_t errin){
    err = errin;
}

void Cuda::store(std::string id, cufftResult resultin){
    result = resultin;
}

void Cuda::store(std::string id, cufftHandle planin){
    plan_map[id] = planin;
}

void Cuda::store(std::string id, cudaStream_t streamin){
    if (id == "streamA"){
        streamA = streamin;
    }
    else if (id == "streamB"){
        streamB = streamin;
    }
    else if (id == "streamC"){
        streamC = streamin;
    }
    else if (id == "streamD"){
        streamD = streamin;
    }
    else{
        std::cout << "Error: stream not found!" << '\n';
    }
}

void Cuda::store(std::string id, dim3 dim3in){
    if (id == "grid"){
        grid = dim3in;
    }
    else if (id == "threads"){
        threads = dim3in;
    }
}

// Functions to retrieve data from Cuda class
// Note: There are definitely more elegant ways to do this.
cudaError_t Cuda::cudaError_tval(std::string id){
    return err;
}

cufftResult Cuda::cufftResultval(std::string id){
    return result;
}

// Returns nothing if called incorrectly.
cufftHandle Cuda::cufftHandleval(std::string id){
    auto it = plan_map.find(id);
    if (it == plan_map.end()){
        std::cout << "ERROR: could not find string " << id 
                  << " in Cuda::plan_map." << '\n';
        assert(it != plan_map.end());
    }
    return it->second;
}

// Returns nothing if called incorrectly
cudaStream_t Cuda::cudaStream_tval(std::string id){
    if (id == "streamA"){
        return streamA;
    }
    else if (id == "streamB"){
        return streamB;
    }
    else if (id == "streamC"){
        return streamC;
    }
    else if (id == "streamD"){
        return streamD;
    }
    else{
        std::cout << "Error: stream not found!" << '\n';
        exit(1);
    }
}

dim3 Cuda::dim3val(std::string id){
    if (id == "grid"){
        return grid;
    }
    else if (id == "threads"){
        return threads;
    }
    else{
        std::cout << "Item " << id << " Not found in Cuda!" << '\n';
        exit(1);
    }
}

/*----------------------------------------------------------------------------//
* OP
*-----------------------------------------------------------------------------*/


// Functions to store data in the Op class
void Op::store(std::string id, const double *paramCD){
    Op_cdstar[id] = paramCD;
}

void Op::store(std::string id, double *data){
    Op_dstar[id] = data;
}

void Op::store(std::string id, cufftDoubleComplex *data){
    Op_cdc[id] = data;
}

// Functions to retrieve data from the Op class
double *Op::dsval(std::string id){
    auto it = Op_dstar.find(id);
    if (it == Op_dstar.end()){
        std::cout << "ERROR: could not find string " << id 
                  << " in Op::Op_dstar." << '\n';
        assert(it != Op_dstar.end());
    }
    return it->second;
}

cufftDoubleComplex *Op::cufftDoubleComplexval(std::string id){
    auto it = Op_cdc.find(id);
    if (it == Op_cdc.end()){
        std::cout << "ERROR: could not find string " << id 
                  << " in Op::Op_cdc." << '\n';
        assert(it != Op_cdc.end());
    }
    return it->second;
}

// Function to set the K function for simulation based on distribution selected
void Op::set_K_fn(std::string id){
    if (id == "rotation_K"){
        K_fn = rotation_K;
    }
    else if(id == "rotation_K3d"){
        K_fn = rotation_K3d;
    }
    else if(id == "rotation_gauge_K"){
        K_fn = rotation_gauge_K;
    }
    else if(id == "rotation_K_dimensionless"){
        K_fn = rotation_K_dimensionless;
    }
}

void Op::set_V_fn(std::string id){
    if (id == "2d"){
        V_fn = harmonic_V;
    }
    else if(id == "torus"){
        V_fn = torus_V;
    }
    else if(id == "3d"){
        V_fn = harmonic_V3d;
    }
    else if(id == "harmonic_gauge_V"){
        K_fn = harmonic_gauge_V;
    }
    else if(id == "harmonic_V_dimensionless"){
        K_fn = harmonic_V_dimensionless;
    }
}

void Op::set_A_fns(std::string id){
    // 3d functions first
    if (id == "rotation"){
        Ax_fn = rotation_Ax;
        Ay_fn = rotation_Ay;
        Az_fn = rotation_Az;
    }
    else if (id == "constant"){
        Ax_fn = constant_A;
        Ay_fn = constant_A;
        Az_fn = constant_A;
    }
    else if (id == "ring"){
        Ax_fn = ring_Ax;
        Ay_fn = ring_Ay;
        Az_fn = constant_A;
    }

    // 2d functions
    else if (id == "dynamic"){
        Ax_fn = dynamic_Ax;
        Ay_fn = dynamic_Ay;
        Az_fn = constant_A;
    }
    else if (id == "fiber2d"){
        Ax_fn = fiber2d_Ax;
        Ay_fn = fiber2d_Ay;
        Az_fn = constant_A;
    }
    else if (id == "test"){
        Ax_fn = test_Ax;
        Ay_fn = test_Ay;
        Az_fn = constant_A;
    }
    else if (id == "test"){
        Ax_fn = test_Ax;
        Ay_fn = test_Ay;
        Az_fn = constant_A;
    }

    // If reading from file, these will be set later
    else if (id == "file"){
        Ax_fn = nullptr;
        Ay_fn = nullptr;
        Az_fn = nullptr;
    }
}

// Function to set functionPtrs without an unordered map
void set_fns(Grid &par, Op &opr, Wave &wave){

    // There are 3 different function distributions to keep in mind:
    // Kfn, Vfn, Afn, wfcfn

    // Kfn
    opr.set_K_fn(par.Kfn);

    // Vfn
    opr.set_V_fn(par.Vfn);

    // Afn
    opr.set_A_fns(par.Afn);

    // Wfcfn
    wave.set_wfc_fn(par.Wfcfn);

}

/*----------------------------------------------------------------------------//
* WAVE
*-----------------------------------------------------------------------------*/

// Functions to store data in the Wave class
void Wave::store(std::string id, double *data){
    Wave_dstar[id] = data;
}

void Wave::store(std::string id, cufftDoubleComplex *data){
    Wave_cdc[id] = data;
}

// Functions to retrieve data from the Wave class
double *Wave::dsval(std::string id){
    auto it = Wave_dstar.find(id);
    if (it == Wave_dstar.end()){
        std::cout << "ERROR: could not find string " << id 
                  << " in Wave::Wave_dstar." << '\n';
        assert(it != Wave_dstar.end());
    }
    return it->second;
}

cufftDoubleComplex *Wave::cufftDoubleComplexval(std::string id){
    auto it = Wave_cdc.find(id);
    if (it == Wave_cdc.end()){
        std::cout << "ERROR: could not find string " << id 
                  << " in Wave::Wave_cdc." << '\n';
        assert(it != Wave_cdc.end());
    }
    return it->second;
}

// Function to set functionPtr for wfc
void Wave::set_wfc_fn(std::string id){
    if (id == "2d"){
        Wfc_fn = standard_wfc_2d;
    }
    else if(id == "3d"){
        Wfc_fn = standard_wfc_3d;
    }
    else if(id == "torus"){
        Wfc_fn = torus_wfc;
    }
}

/*----------------------------------------------------------------------------//
* MISC
*-----------------------------------------------------------------------------*/

/*
// Template function to print all values in map
template <typename T> void print_map(std::unordered_map<std::string, T> map){
    std::cout << "Contents of map are: " << '\n';
    std::cout << "key: " << '\t' << "element: " << '\n';
    for (auto element : map){
        std::cout << element.first << '\t' << element.second << '\n';
    }
}
*/
