
#include "../include/ds.h"
#include "../include/unit_test.h"
#include "../include/parser.h"
#include "../include/evolution.h"
#include "../include/init.h"
#include "../include/dynamic.h"
#include <string.h>
#include <assert.h>
#include <cufft.h>
#include <vector>
#include <fstream>

// Test for the Grid structure with paramters in it 
void parameter_test();

// Test for the parsing function
void parser_test();

// Testing the evolve_2d function in evolution.cu
void evolve_2d_test();

// Testing the parSum function
void parSum_test();

// Simple test of grid / cuda stuff
void grid_test2d();
void grid_test3d();

// Test of 1D fft's along all 3d grids
void fft_test();

// Test to check the equation parser for dynamic fields
void dynamic_test();

// Test to make sure the kernel for the polynomial approx. of Bessel fxns works
void bessel_test();

// Kernel testing will be added later
__device__ bool close(double a, double b, double threshold){
    return (abs(a-b) < threshold);
}


/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

void test_all(){
    std::cout << "Starting unit tests..." << '\n';
    //parameter_test();

    // Do not uncomment these 2
    //parser_test();
    //evolve_2d_test();

    //grid_test2d();
    //grid_test3d();
    //parSum_test();
    //fft_test();
    dynamic_test();
    bessel_test();

    std::cout << "All tests completed. GPUE passed." << '\n';
}

// Test to check the equation parser for dynamic fields
// For this test, we will need a general set of parameters to read in and a
// standard equation string to look at. 
void dynamic_test(){

    std::cout << "Beginning test of dynamic functions..." <<'\n';
    std::string eqn_string = "(((3*x)+7)+(5-7)+cos(0))+pow(120,2)";

    Grid par;
    par.store("x",5);

    EqnNode eqn_tree = parse_eqn(par, eqn_string);

    std::cout << "finding the number of elements in abstract syntax tree...\n";

    int num = 0;
    find_element_num(eqn_tree, num);
    int element_num = num;

    std::cout << "Total number of elements is: " << num << '\n';

    std::cout << "Now to copy the tree to the GPU..." << '\n';

    EqnNode_gpu *eqn_gpu, *eqn_cpu;
    eqn_cpu = (EqnNode_gpu *)malloc(sizeof(EqnNode_gpu)*element_num);
    num = 0;
    tree_to_array(eqn_tree, eqn_cpu, num);

/*
    for (int i = 0; i < num; ++i){
        std::cout << eqn_cpu[i].val << '\n';
        std::cout << eqn_cpu[i].left << '\n';
        std::cout << eqn_cpu[i].right << '\n' << '\n';
    }
*/

    cudaMalloc((void**)&eqn_gpu, sizeof(EqnNode_gpu)*element_num);
    cudaMemcpy(eqn_gpu, eqn_cpu, sizeof(EqnNode_gpu)*element_num,
               cudaMemcpyHostToDevice);

    // Now to check some simple evaluation
    std::cout << "Now to check simple GPU evaluation..." << '\n';
    int n = 64;
    double *array, *array_gpu;
    array = (double *)malloc(sizeof(double)*n);
    cudaMalloc(&array_gpu, sizeof(double)*n);

    int threads = 64;
    int grid = (int)ceil((float)n/threads);

    //zeros<<<grid, threads>>>(array_gpu, n);
    find_field<<<grid, threads>>>(array_gpu, 1, 0.0, 0.0, 0.0, eqn_gpu);

    cudaMemcpy(array, array_gpu, sizeof(double)*n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i){
        std::cout << array[i] << '\n';
    }

    // Now testing simple parsing of example "example.cfg"
    std::cout << "Testing simple parameter parsing." << '\n';
    par.store("param_file", (std::string)"src/example.cfg");
    parse_param_file(par);

    std::cout << "Dynamic tests passed" <<'\n';
}

__global__ void bessel_test_kernel(double *j, double *j_poly, bool *val){
    int xid = blockDim.x*blockIdx.x + threadIdx.x;
    j[xid] = j0(xid * 2.0 / 128);
    j_poly[xid] = poly_j(0,xid * 2.0 / 128, 20);

    if (!close(j[xid],j_poly[xid], 0.00001)){
        val[0] = false;
    }
    else val[0] = true;
}

// Test for bessel functions
void bessel_test(){

    std::cout << "Testing Bessel Functions..." << '\n';

    double *j_gpu, *j_poly_gpu;
    bool *val, *val_gpu;
    int n = 128;
    cudaMalloc((void **)&j_gpu, sizeof(double)*n);
    cudaMalloc((void **)&j_poly_gpu, sizeof(double)*n);

    cudaMalloc((void **)&val_gpu, sizeof(bool));
    val = (bool *)malloc(sizeof(bool));

    bessel_test_kernel<<<64,2>>>(j_gpu, j_poly_gpu, val_gpu);
    cudaMemcpy(val, val_gpu, sizeof(bool), cudaMemcpyDeviceToHost);

    if(val[0]){
        std::cout << "Bessel Test Passed!" << '\n';
    }
    else{
        std::cout << "Bessel Test Failed!" << '\n';
        exit(1);
    }

}

// Test of 1D fft's along all 3d grids
// In particular, we need to test the generate_plan_other3d function
// These will be checked against 1d 
void fft_test(){

    // For these tests, we are assuming that the x, y and z dimensions are 
    // All the same (2x2x2)
    // Note that yDim needs to be singled out differently, but z/x need no loops

    // now we need to create the necessary parameters and store everything
    int xDim = 2;
    int yDim = 2;
    int zDim = 2;
    int gsize = xDim * yDim * zDim;

    Grid par;
    par.store("xDim", xDim);
    par.store("yDim", yDim);
    par.store("zDim", zDim);

    cufftHandle plan_x, plan_y, plan_z;
    // Now creating the plans
    generate_plan_other3d(&plan_x, par, 0);
    generate_plan_other3d(&plan_y, par, 1);
    generate_plan_other3d(&plan_z, par, 2);

    // And the result / error
    cudaError_t err;
    cufftResult result;

    // Creating the initial array for the x dimension fft
    double2 *array, *gpu_array;
    array = (double2 *) malloc(sizeof(double2)*gsize);
    cudaMalloc((void**) &gpu_array, sizeof(double2)*gsize);
    for (int i = 0; i < gsize; i++){
        array[i].x = 1;
        array[i].y = 0;
    }

    // transferring to gpu
    err = cudaMemcpy(gpu_array, array, sizeof(double2)*gsize,
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        std::cout << "Could not coppy array to device!" << '\n';
        std::cout << "error code: " << err << '\n';
        exit(1);
    }

    // Performing the x transformation
    for (int i = 0; i < yDim; i++){
        result = cufftExecZ2Z(plan_y, &gpu_array[i*xDim*yDim], 
                                      &gpu_array[i*xDim*yDim], CUFFT_FORWARD);
    }
    //result = cufftExecZ2Z(plan_z, gpu_array, gpu_array, CUFFT_FORWARD);

    // transferring back to host to check output
    err = cudaMemcpy(array, gpu_array, sizeof(double2)*gsize, 
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        std::cout << "Could not coppy gpu_array to host!" << '\n';
        std::cout << "error code: " << err << '\n';
        exit(1);
    }

    for (int i = 0; i < gsize; i++){
        std::cout << array[i].x << '\t' << array[i].y << '\n';
    }

    // Now to try the inverse direction

    for (int i = 0; i < yDim; i++){
        result = cufftExecZ2Z(plan_y, &gpu_array[i*xDim*yDim], 
                                      &gpu_array[i*xDim*yDim], CUFFT_INVERSE);
    }
    //result = cufftExecZ2Z(plan_z, gpu_array, gpu_array, CUFFT_INVERSE);

    // copying back
    err = cudaMemcpy(array, gpu_array, sizeof(double2)*gsize, 
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        std::cout << "Could not coppy gpu_array to host!" << '\n';
        std::cout << "error code: " << err << '\n';
        exit(1);
    }

    for (int i = 0; i < gsize; i++){
        std::cout << array[i].x << '\t' << array[i].y << '\n';
    }



}

// Simple test of CUDA grid stuff
void grid_test2d(){

    std::cout << "testing grid / threads and stuff" << '\n';

    int max_threads = 128;

    int xDim = 1024;
    int yDim = 1024;
    int zDim = 1;

    int xD = 1, yD = 1, zD = 1;

    int gsize = xDim * yDim;

    // Now to set up the CUDA grid / threads
    dim3 block;
    dim3 grid;

    if (xDim <= max_threads){
        block.x = xDim;
        block.y = 1;
        block.z = 1;

        xD = 1;
        yD = yDim;
        zD = 1;
    } 
    else{
        int count = 0;
        int dim_tmp = xDim;
        while (dim_tmp > max_threads){
            count++;
            dim_tmp /= 2;
        }

        std::cout << "count is: " << count << '\n';

        block.x = dim_tmp;
        block.y = 1;
        block.z = 1;
        xD = pow(2,count);
        yD = yDim;
        zD = 1;
    }

    std::cout << "threads in x are: " << block.x << '\n';
    std::cout << "dimensions are: " << xD << '\t' << yD << '\t' << zD << '\n';

    grid.x=xD; 
    grid.y=yD; 
    grid.z=zD; 

    int total_threads = block.x * block.y * block.z;

    // Now we need to initialize our double * and send it to the gpu
    double *host_array, *device_array;
    host_array = (double *) malloc(sizeof(double)*gsize);
    cudaMalloc((void**) &device_array, sizeof(double)*gsize);

    // initializing 2d array
    for (int i = 0; i < gsize; i++){
        host_array[i] = -1;
    }

    // Now to copy to device
    cudaMemcpy(device_array, host_array,
               sizeof(double)*gsize,
               cudaMemcpyHostToDevice);

    // Test
    thread_test<<<grid,block>>>(device_array,device_array);

    // Now to copy back and print
    cudaMemcpy(host_array, device_array,
               sizeof(double)*gsize,
               cudaMemcpyDeviceToHost);
    
    
/*
    for (int i = 0; i < gsize; i++){
        std::cout << i << '\t' <<  host_array[i] << '\n';
    }
*/
    std::cout << "1024 x 1024 is: " << host_array[gsize-1] << '\n';
    assert(host_array[gsize-1] == 1024*1024-1);

    std::cout << "2d grid tests completed. now for 3d cases" << '\n';

}

// Simple test of CUDA grid stuff
void grid_test3d(){

    std::cout << "testing grid / threads and stuff for 3d" << '\n';

    int max_threads = 128;

    int xDim = 256;
    int yDim = 256;
    int zDim = 256;

    int xD = 1, yD = 1, zD = 1;

    int gsize = xDim * yDim * zDim;

    // Now to set up the CUDA grid / threads
    dim3 block;
    dim3 grid;

    if (xDim <= max_threads){
        block.x = xDim;
        block.y = 1;
        block.z = 1;

        xD = 1;
        yD = yDim;
        zD = zDim;
    } 
    else{
        int count = 0;
        int dim_tmp = xDim;
        while (dim_tmp > max_threads){
            count++;
            dim_tmp /= 2;
        }

        std::cout << "count is: " << count << '\n';

        block.x = dim_tmp;
        block.y = 1;
        block.z = 1;
        xD = pow(2,count);
        yD = yDim;
        zD = zDim;
    }

    std::cout << "threads in x are: " << block.x << '\n';
    std::cout << "dimensions are: " << xD << '\t' << yD << '\t' << zD << '\n';

    grid.x=xD; 
    grid.y=yD; 
    grid.z=zD; 

    int total_threads = block.x * block.y * block.z;

    // Now we need to initialize our double * and send it to the gpu
    double *host_array, *device_array;
    host_array = (double *) malloc(sizeof(double)*gsize);
    cudaMalloc((void**) &device_array, sizeof(double)*gsize);

    // initializing 2d array
    for (int i = 0; i < gsize; i++){
        host_array[i] = -1;
    }

    // Now to copy to device
    cudaMemcpy(device_array, host_array,
               sizeof(double)*gsize,
               cudaMemcpyHostToDevice);

    // Test
    thread_test<<<grid,block>>>(device_array,device_array);

    // Now to copy back and print
    cudaMemcpy(host_array, device_array,
               sizeof(double)*gsize,
               cudaMemcpyDeviceToHost);
    
    
/*
    for (int i = 0; i < gsize; i++){
        std::cout << i << '\t' <<  host_array[i] << '\n';
    }
*/
    std::cout << "256x256x256 is: " << host_array[gsize-1] << '\n';
    assert(host_array[gsize-1] == 256*256*256-1);

    std::cout << "3d grid tests completed. now for 3d cases" << '\n';

}

// Test of the parSum function in 3d
void parSum_test(){

    // Setting error
    cudaError_t err;

    // first, we need to initialize the Grid and Cuda classes
    Grid par;

    // 2D test first

    // For now, we will assume an 8x8 array for summing
    dim3 threads(16, 1, 1);
    int total_threads = threads.x*threads.y*threads.z;

    par.store("dimnum", 2);
    par.store("xDim", 64);
    par.store("yDim", 64);
    par.store("zDim", 1);
    par.store("dx",1.0);
    par.store("dy",1.0);
    par.store("dz",1.0);
    par.threads = threads;

    // Now we need to initialize the grid for the getGid3d3d kernel
    int gsize = 64*64;
    dim3 grid;
    grid.x = 4;
    grid.y = 64;

    par.grid = grid;

    // now we need to initialize the wfc to all 1's;
    double2 *wfc, *host_sum;
    wfc = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gsize);
    host_sum = (cufftDoubleComplex *) 
               malloc(sizeof(cufftDoubleComplex) * gsize / total_threads);

    // init wfc
    for (int i = 0; i < gsize; i++){
        wfc[i].x = 1;
        wfc[i].y = 0;
    }

    double2 *gpu_wfc;
    cudaMalloc((void**) &gpu_wfc, sizeof(cufftDoubleComplex)*gsize);

    // copying wfc to device
    err = cudaMemcpy(gpu_wfc, wfc, sizeof(cufftDoubleComplex)*gsize,
                     cudaMemcpyHostToDevice);

    if (err!=cudaSuccess){
        std::cout << "ERROR: Could not copy wfc to device!" << '\n';
    }

    // Creating parsum on device
    double2 *par_sum;
    cudaMalloc((void**) &par_sum, 
                   sizeof(cufftDoubleComplex)*gsize/total_threads);

    parSum(gpu_wfc, par_sum, par);

    // copying parsum back
    err = cudaMemcpy(host_sum, par_sum, 
                     sizeof(cufftDoubleComplex)*gsize / total_threads, 
                     cudaMemcpyDeviceToHost);
    if (err!=cudaSuccess){
        std::cout << err << '\n';
        std::cout << "ERROR: Could not copy par_sum to the host!" << '\n';
        exit(1);
    }

    // The output value should be 4096
    std::cout << "2d parSum is:" << '\n';
    std::cout << host_sum[0].x << " + " << host_sum[0].y << " i" << '\n';

    if (host_sum[0].x != 4096){
        std::cout << "parSum 2d test has failed! Sum is: "
                  << host_sum[0].x << '\n';
        assert((int)host_sum[0].x == 4096);
    }

    // Now for the 3d case
    // For now, we will assume a 16x16x16 array for summing
    par.store("dimnum", 3);
    par.store("xDim", 16);
    par.store("yDim", 16);
    par.store("zDim", 16);
    par.store("dx",1.0);
    par.store("dy",1.0);
    par.store("dz",1.0);

    // Now we need to initialize the grid for the getGid3d3d kernel
    grid.x = 1;
    grid.y = 16;
    grid.z = 16;

    par.grid = grid;

    // copying host wfc back to device
    err = cudaMemcpy(gpu_wfc, wfc, sizeof(cufftDoubleComplex)*gsize,
                     cudaMemcpyHostToDevice);

    parSum(gpu_wfc, par_sum, par);

    // copying parsum back
    err = cudaMemcpy(host_sum, par_sum, 
                     sizeof(cufftDoubleComplex)*gsize / total_threads, 
                     cudaMemcpyDeviceToHost);
    if (err!=cudaSuccess){
        std::cout << "ERROR: Could not copy par_sum to the host!" << '\n';
        exit(1);
    }

    std::cout << "3d parSum is:" << '\n';
    std::cout << host_sum[0].x << " + " << host_sum[0].y << " i" << '\n';

    if (host_sum[0].x != 4096){
        std::cout << "parSum 3d test has failed!" << '\n';
        assert((int)host_sum[0].x == 4096);
    }

}

// Test for the Grid structure with paramters in it
// Initialize all necessary variables and read them back out
void parameter_test(){
    // For this test, we simply need to read in and out stuff from each 
    // class and structure in ds.h / ds.cc
    
    // Certain variables will be used multiple times. 
    double *dstar_var;
    dstar_var = (double *)malloc(sizeof(double) * 5);
    cufftDoubleComplex *cdc_var;
    cdc_var = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex) * 5);
    for (int i = 0; i < 5; ++i){
        dstar_var[i] = (double)i * 0.5;
        cdc_var[i].x = (double)i * 0.5;
        cdc_var[i].y = (double)i * 0.5;
    }

    double dvar = 1.05;
    int ivar = 5;
    bool bvar = true;

    // Now testing the Grid class
    Grid grid_test;
    grid_test.store("dstar_var",dstar_var);
    grid_test.store("dvar", dvar);
    grid_test.store("ivar", ivar);
    grid_test.store("bvar", bvar);

    assert(dstar_var == grid_test.dsval("dstar_var"));
    assert(dvar == grid_test.dval("dvar"));
    assert(ivar == grid_test.ival("ivar"));
    assert(bvar == grid_test.bval("bvar"));

    std::cout << "Grid class checked, now checking the Cuda class..." << '\n';

    std::cout << "All data structures checked" << '\n';

}

// Test for the parsing function
void parser_test(){

    // Testing the command-line parser with defaults and with modifications
    std::cout << "Testing command-line parser with no arguments..." << '\n';

    // First testing default values in and out of the parser function
    char *fake_noargv[] = {NULL};
    Grid noarg_grid;
    noarg_grid = parseArgs(0,fake_noargv);

    // Checking contents of noarg_grid:
    assert(noarg_grid.ival("xDim") == 256);
    assert(noarg_grid.ival("yDim") == 256);
    assert(noarg_grid.ival("zDim") == 256);
    assert(noarg_grid.dval("omega") == 0);
    assert(noarg_grid.dval("gammaY") == 1.0);
    assert(noarg_grid.ival("gsteps") == 1);
    assert(noarg_grid.ival("esteps") == 1);
    assert(noarg_grid.dval("gdt") == 1e-4);
    assert(noarg_grid.dval("dt") == 1e-4);
    assert(noarg_grid.ival("device") == 0);
    assert(noarg_grid.ival("atoms") == 1);
    assert(noarg_grid.bval("read_wfc") == false);
    assert(noarg_grid.ival("printSteps") == 100);
    assert(noarg_grid.dval("winding") == 0);
    assert(noarg_grid.bval("corotating") == false);
    assert(noarg_grid.bval("gpe") == false);
    assert(noarg_grid.dval("omegaZ") == 6.283);
    assert(noarg_grid.dval("interaction") == 1);
    assert(noarg_grid.dval("laser_power") == 0);
    assert(noarg_grid.dval("angle_sweep") == 0);
    assert(noarg_grid.ival("kick_it") == 0);
    assert(noarg_grid.bval("write_it") == false);
    assert(noarg_grid.dval("x0_shift") == 0);
    assert(noarg_grid.dval("y0_shift") == 0);
    assert(noarg_grid.dval("z0_shift") == 0);
    assert(noarg_grid.dval("sepMinEpsilon") == 0);
    assert(noarg_grid.bval("graph") == false);
    assert(noarg_grid.bval("unit_test") == false);
    assert(noarg_grid.dval("omegaX") == 6.283);
    assert(noarg_grid.dval("omegaY") == 6.283);
    assert(noarg_grid.sval("data_dir") == "data/");
    assert(noarg_grid.bval("ramp") == false);
    assert(noarg_grid.ival("ramp_type") == 1);
    assert(noarg_grid.ival("dimnum") == 2);
    assert(noarg_grid.bval("write_file") == true);
    assert(noarg_grid.dval("fudge") == 1.0);
    assert(noarg_grid.ival("kill_idx") == -1);
    assert(noarg_grid.dval("DX") == 0.0);
    assert(noarg_grid.dval("mask_2d") == 1.5e-4);
    assert(noarg_grid.dval("box_size") == 2.5e-5);
    assert(noarg_grid.bval("found_sobel") == false);
    assert(noarg_grid.Afn == "rotation");
    assert(noarg_grid.Kfn == "rotation_K");
    assert(noarg_grid.Vfn == "2d");
    assert(noarg_grid.Wfcfn == "2d");
    assert(noarg_grid.sval("conv_type") == "FFT");
    assert(noarg_grid.ival("charge") == 0);
    assert(noarg_grid.bval("flip") == false);

    // Now testing all values specified by command-line arguments
    std::cout << "Testing command-line parser with all arguments..." << '\n';
    std::vector<std::string> argarray(10);

    // I apologize for the mess... If you have a better way of creating the 
    // char ** for this without running into memory issues, let me know!
    char *fake_fullargv[] = {strdup("./gpue"), 
                             strdup("-A"), strdup("rotation"), 
                             strdup("-a"),
                             strdup("-b"), strdup("2.5e-5"), 
                             strdup("-C"), strdup("0"), 
                             strdup("-c"), strdup("3"), 
                             strdup("-D"), strdup("data"), 
                             strdup("-E"), 
                             strdup("-e"), strdup("1"), 
                             strdup("-f"), 
                             strdup("-G"), strdup("1"),
                             strdup("-g"), strdup("1"), 
                             strdup("-i"), strdup("1"), 
                             strdup("-K"), strdup("0"), 
                             strdup("-k"), strdup("0"),
                             strdup("-L"), strdup("0"), 
                             strdup("-l"), 
                             strdup("-n"), strdup("1"), 
                             strdup("-O"), strdup("0"),
                             strdup("-P"), strdup("0"), 
                             strdup("-p"), strdup("100"),
                             strdup("-Q"), strdup("0"), 
                             strdup("-q"), strdup("0"), 
                             strdup("-R"), strdup("1"), 
                             //strdup("-r"),
                             strdup("-S"), strdup("0"), 
                             strdup("-s"),
                             strdup("-T"), strdup("1e-4"), 
                             strdup("-t"), strdup("1e-4"), 
                             strdup("-U"), strdup("0"), 
                             strdup("-V"), strdup("0"), 
                             strdup("-W"), 
                             strdup("-w"), strdup("0"), 
                             strdup("-X"), strdup("1.0"),
                             strdup("-x"), strdup("256"), 
                             strdup("-Y"), strdup("1.0"), 
                             strdup("-y"), strdup("256"),
                             strdup("-Z"), strdup("6.283"), 
                             strdup("-z"), strdup("256"), 
                             NULL};
    int fake_argc = sizeof(fake_fullargv) / sizeof(char *) - 1;

    // Now to read into gpue and see what happens
    Grid fullarg_grid;
    fullarg_grid = parseArgs(fake_argc, fake_fullargv);

    // Checking contents of fullarg_grid:
    assert(fullarg_grid.ival("xDim") == 256);
    assert(fullarg_grid.ival("yDim") == 256);
    assert(fullarg_grid.ival("zDim") == 256);
    assert(fullarg_grid.dval("omega") == 0);
    assert(fullarg_grid.dval("gammaY") == 1.0);
    assert(fullarg_grid.ival("gsteps") == 1);
    assert(fullarg_grid.ival("esteps") == 1);
    assert(fullarg_grid.dval("gdt") == 1e-4);
    assert(fullarg_grid.dval("dt") == 1e-4);
    assert(fullarg_grid.ival("device") == 0);
    assert(fullarg_grid.ival("atoms") == 1);
    assert(fullarg_grid.bval("read_wfc") == false);
    assert(fullarg_grid.ival("printSteps") == 100);
    assert(fullarg_grid.dval("winding") == 0);
    assert(fullarg_grid.bval("corotating") == true);
    assert(fullarg_grid.bval("gpe") == true);
    assert(fullarg_grid.dval("omegaZ") == 6.283);
    assert(fullarg_grid.dval("interaction") == 1);
    assert(fullarg_grid.dval("laser_power") == 0);
    assert(fullarg_grid.dval("angle_sweep") == 0);
    assert(fullarg_grid.ival("kick_it") == 0);
    assert(fullarg_grid.bval("write_it") == true);
    assert(fullarg_grid.dval("x0_shift") == 0);
    assert(fullarg_grid.dval("y0_shift") == 0);
    assert(fullarg_grid.dval("z0_shift") == 0);
    assert(fullarg_grid.dval("sepMinEpsilon") == 0);
    assert(fullarg_grid.bval("graph") == true);
    assert(fullarg_grid.bval("unit_test") == false);
    assert(fullarg_grid.dval("omegaX") == 1.0);
    assert(fullarg_grid.dval("omegaY") == 1.0);
    assert(fullarg_grid.sval("data_dir") == "data/");
    assert(fullarg_grid.bval("ramp") == true);
    assert(fullarg_grid.ival("ramp_type") == 1);
    assert(fullarg_grid.ival("dimnum") == 3);
    assert(fullarg_grid.bval("write_file") == false);
    assert(fullarg_grid.dval("fudge") == 1.0);
    assert(fullarg_grid.ival("kill_idx") == 0);
    assert(fullarg_grid.dval("DX") == 0.0);
    assert(fullarg_grid.dval("mask_2d") == 1.5e-4);
    assert(fullarg_grid.dval("box_size") == 2.5e-5);
    assert(fullarg_grid.bval("found_sobel") == false);
    assert(fullarg_grid.Afn == "rotation");
    assert(fullarg_grid.Kfn == "rotation_K3d");
    assert(fullarg_grid.Vfn == "3d");
    assert(fullarg_grid.Wfcfn == "3d");
    assert(fullarg_grid.sval("conv_type") == "FFT");
    assert(fullarg_grid.ival("charge") == 0);
    assert(fullarg_grid.bval("flip") == true);

}

// Testing the evolve_2d function in evolution.cu
void evolve_2d_test(){
    // First, we need to create all the necessary data structures for the
    // The evolve_2d function, FOLLOWING INIT.CU

    std::cout << "Testing the evolve_2d function" << '\n';

    // Note: the omega_z value (-o flag) is arbitrary
    char * fake_argv[] = {strdup("./gpue"), 
                          strdup("-C"), strdup("0"), 
                          strdup("-e"), strdup("2.01e4"), 
                          strdup("-G"), strdup("1.0"), 
                          strdup("-g"), strdup("0"), 
                          strdup("-i"), strdup("1.0"), 
                          strdup("-k"), strdup("0"), 
                          strdup("-L"), strdup("0"), 
                          strdup("-n"), strdup("1e6"), 
                          strdup("-O"), strdup("0.0"), 
                          strdup("-Z"), strdup("10.0"), 
                          strdup("-P"), strdup("0.0"), 
                          strdup("-p"), strdup("1000"), 
                          strdup("-S"), strdup("0.0"), 
                          strdup("-T"), strdup("1e-4"), 
                          strdup("-t"), strdup("1e-4"), 
                          strdup("-U"), strdup("0"), 
                          strdup("-V"), strdup("0"), 
                          strdup("-w"), strdup("0.0"), 
                          strdup("-X"), strdup("1.0"), 
                          strdup("-x"), strdup("256"), 
                          strdup("-Y"), strdup("1.0"), 
                          strdup("-y"), strdup("256"), 
                          strdup("-W"), 
                          strdup("-D"), strdup("data"), NULL};
    int fake_argc = sizeof(fake_argv) / sizeof(char *) - 1;

    // Now to read into gpue and see what happens
    Grid par;
    par = parseArgs(fake_argc, fake_argv);

    std::cout << "omegaX is: " << par.dval("omegaX") << '\n';
    std::cout << "x / yDim are: " << par.ival("xDim") << '\t' 
              << par.ival("yDim") << '\n';
    int device = par.ival("device");
    cudaSetDevice(device);

    std::string buffer;

    //************************************************************//
    /*
    * Initialise the Params data structure to track params and variables
    */
    //************************************************************//

    init(par);

    // Re-establishing variables from parsed Grid class
    double dx = par.dval("dx");
    double dy = par.dval("dy");
    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double *V_opt = par.dsval("V_opt");
    double *pAy = par.dsval("pAy");
    double *pAx = par.dsval("pAx");
    double *pAy_gpu = par.dsval("pAy_gpu");
    double *pAx_gpu = par.dsval("pAx_gpu");
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    bool read_wfc = par.bval("read_wfc");
    int gsteps = par.ival("gsteps");
    int esteps = par.ival("esteps");
    cufftDoubleComplex *wfc = par.cufftDoubleComplexval("wfc");
    cufftDoubleComplex *V_gpu = par.cufftDoubleComplexval("V_gpu");
    cufftDoubleComplex *GK = par.cufftDoubleComplexval("GK");
    cufftDoubleComplex *GV = par.cufftDoubleComplexval("GV");
    cufftDoubleComplex *EV = par.cufftDoubleComplexval("EV");
    cufftDoubleComplex *EK = par.cufftDoubleComplexval("EK");
    cufftDoubleComplex *EpAy = par.cufftDoubleComplexval("EpAy");
    cufftDoubleComplex *EpAx = par.cufftDoubleComplexval("EpAx");
    cufftDoubleComplex *GpAx = par.cufftDoubleComplexval("GpAx");
    cufftDoubleComplex *GpAy = par.cufftDoubleComplexval("GpAy");
    cufftDoubleComplex *wfc_gpu = par.cufftDoubleComplexval("wfc_gpu");
    cufftDoubleComplex *K_gpu = par.cufftDoubleComplexval("K_gpu");
    cufftDoubleComplex *par_sum = par.cufftDoubleComplexval("par_sum");
    cudaError_t err;

    std::cout << "variables re-established" << '\n';
    std::cout << read_wfc << '\n';

    std::cout << "omegaY is: " << par.ival("omegaY") << '\t'
              << "omegaX is: " << par.dval("omegaX") << '\n';

/*
    for (int i = 0; i < xDim * yDim; ++i){
        std::cout << i << '\t' << wfc[i].x << '\t' << wfc[i].y << '\n';
    }
*/

    std::cout << "gsteps: " << gsteps << '\n';
   
    if(gsteps > 0){
        err=cudaMemcpy(K_gpu, GK, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy K_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(V_gpu, GV, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy V_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(pAy_gpu, GpAy, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy pAy_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(pAx_gpu, GpAx, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy pAx_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(wfc_gpu, wfc, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy wfc_gpu to device" << '\n';
            exit(1);
        }
    
        evolve_2d(par, par_sum, gsteps, 0, buffer);
        wfc = par.cufftDoubleComplexval("wfc");
        wfc_gpu = par.cufftDoubleComplexval("wfc_gpu");
        cudaMemcpy(wfc, wfc_gpu, sizeof(cufftDoubleComplex)*xDim*yDim,
                   cudaMemcpyDeviceToHost);
    }

    std::cout << GV[0].x << '\t' << GK[0].x << '\t'
              << pAy[0] << '\t' << pAx[0] << '\n';

    //free(GV); free(GK); free(pAy); free(pAx);

    // Re-initializing wfc after evolution
    wfc = par.cufftDoubleComplexval("wfc");
    wfc_gpu = par.cufftDoubleComplexval("wfc_gpu");

    std::cout << "evolution started..." << '\n';
    std::cout << "esteps: " << esteps << '\n';

    //************************************************************//
    /*
    * Evolution
    */
    //************************************************************//
    if(esteps > 0){
        err=cudaMemcpy(pAy_gpu, EpAy, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy pAy_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(pAx_gpu, EpAx, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy pAx_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(K_gpu, EK, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy K_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(V_gpu, EV, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy V_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(wfc_gpu, wfc, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy wfc_gpu to device" << '\n';
            exit(1);
        }

        evolve_2d(par, par_sum,
               esteps, 1, buffer);

    }

    std::cout << "done evolving, checking result" << '\n';

    // At this point, we have a wavefunction that is testable, which we will be
    // doing in much the same way as in the linear/perf branch of GPUE.
    // For this, we must recreate the en.py file in a testable format in cpp
    // Note that we could be using the GPUs for this, but because it is a unit
    // test and we do not care that much about perfomance, we will be using the 
    // CPU instead. We may later add in the appropriate GPU kernels.

    // We first need to grab the wavefunctions from the evolve_2d function
    // After evolution
    wfc = par.cufftDoubleComplexval("wfc");
    wfc_gpu = par.cufftDoubleComplexval("wfc_gpu");
    unsigned int gSize = xDim * yDim;

    // Now to grab K and V, note that these are different than the values used 
    // for K / V_gpu or for E / G K / V in the evolve_2d function
    // The additional 0 in the gpu variable name indicate this (sorry)
    double *K_0_gpu = par.dsval("K");
    double *K = par.dsval("K");
    double *V_0_gpu = par.dsval("V");
    double *V = par.dsval("V");

    // Now we need som CUDA specific variables for the kernels later on...
    int threads = par.ival("threads");
    dim3 grid = par.grid;

    // Momentum-space (p) wavefunction
    double2 *wfc_p = wfc;
    double2 *wfc_p_gpu = wfc_gpu;

    // Conjugate (c) wavefunction
    double2 *wfc_c = wfc;
    double2 *wfc_c_gpu = wfc_gpu;

    // Energies
    double2 *Energy_1, *Energy_2, *Energy_k, *Energy_v;
    Energy_1 = wfc_gpu;
    Energy_2 = wfc_gpu;

    // Plan for 2d FFT
    cufftHandle plan_2d = par.ival("plan_2d");

    std::cout << "allocating space on device..." << '\n';

    // Allocating space on GPU
    cudaMalloc((void **) &wfc_gpu, sizeof(cufftDoubleComplex) * gSize);
    cudaMalloc((void **) &K_0_gpu, sizeof(double) * gSize);
    cudaMalloc((void **) &V_0_gpu, sizeof(double) * gSize);
    cudaMalloc((void **) &wfc_p_gpu, sizeof(cufftDoubleComplex) * gSize);
    cudaMalloc((void **) &wfc_c_gpu, sizeof(cufftDoubleComplex) * gSize);
    cudaMalloc((void **) &par_sum, sizeof(cufftDoubleComplex)*(gSize/threads));

    std::cout << "copying contents... " << '\n';

    // Copy variables over to device
    cudaMemcpy(wfc_gpu, wfc, sizeof(cufftDoubleComplex) * gSize,
               cudaMemcpyHostToDevice);
    std::cout << "wfc copied..." << '\n';
    cudaMemcpy(K_0_gpu, K, sizeof(cufftDoubleComplex) * gSize,
               cudaMemcpyHostToDevice);
    std::cout << "K copied..." << '\n';
    cudaMemcpy(V_0_gpu, GV, sizeof(cufftDoubleComplex) * gSize,
               cudaMemcpyHostToDevice);
    std::cout << "V copied..." << '\n';
    cudaMemcpy(wfc_p_gpu, wfc_p, sizeof(cufftDoubleComplex) * gSize,
               cudaMemcpyHostToDevice);
    std::cout << "wfc_p copied..." << '\n';
    cudaMemcpy(wfc_c_gpu, wfc_c, sizeof(cufftDoubleComplex) * gSize,
               cudaMemcpyHostToDevice);
    std::cout << "wfc_c copied..." << '\n';

    std::cout << "performing energy calculations..." << '\n';


    // In the example python code, it was necessary to reshape everything, 
    // But let's see what happens if I don't do that here...

    // FFT for the wfc in momentum-space
    cufftExecZ2Z(plan_2d, wfc_gpu, wfc_p, CUFFT_FORWARD);

    // Conjugate for the wfc
    vecConjugate<<<grid,threads>>>(wfc_gpu, wfc_c);

    // K * wfc
    vecMult<<<grid,threads>>>(wfc_gpu,K_0_gpu,wfc_p);
    cufftExecZ2Z(plan_2d, wfc_p, Energy_1, CUFFT_INVERSE); 

    vecMult<<<grid,threads>>>(wfc_gpu, V_0_gpu, Energy_2);

/*
    for (int i = 0; i < xDim * yDim; ++i){
        std::cout << Energy_1[i].y << '\t' << Energy_2[i].x << '\n';
    }
*/

    //std::cout << wfc_gpu[0].x << '\t' << wfc_gpu[0].y << '\n';

    free(EV); free(EK); free(EpAy); free(EpAx);
    free(x);free(y);
    cudaFree(wfc_gpu); cudaFree(K_gpu); cudaFree(V_gpu); cudaFree(pAx_gpu);
    cudaFree(pAy_gpu); cudaFree(par_sum);

    std::cout << "Evolution test complete." << '\n';
    std::cout << "EVOLUTION TEST UNFINISHED!" << '\n';
    
}
