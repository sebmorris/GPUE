
#include "../include/init.h"

int init(Op &opr, Grid &par, Wave &wave){

    set_fns(par, opr, wave);

    // Re-establishing variables from parsed Grid class
    // Initializes uninitialized variables to 0 values
    std::string data_dir = par.sval("data_dir");
    int dimnum = par.ival("dimnum");
    int N = par.ival("atoms");
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int zDim = par.ival("zDim");
    bool write_file = par.bval("write_file");
    dim3 threads;
    unsigned int gSize = xDim*yDim;
    if (dimnum == 3){
        gSize *= zDim;
    }
    double omega = par.dval("omega");
    double gdt = par.dval("gdt");
    double dt = par.dval("dt");
    double omegaX = par.dval("omegaX");
    double omegaY = par.dval("omegaY");
    double omegaZ = par.dval("omegaZ");
    double gammaY = par.dval("gammaY"); //Aspect ratio of trapping geometry.
    double l = par.dval("winding");
    double box_size = par.dval("box_size");
    double *Energy;
    double *r;
    double *V_opt;
    double *Bz;
    double *Bx;
    double *By;
    double *Energy_gpu;
    cufftDoubleComplex *wfc;
    if (par.bval("read_wfc") == true){
        wfc = par.cufftDoubleComplexval("wfc");
    }
    cufftDoubleComplex *EV_opt;
    cufftDoubleComplex *wfc_backup;
    cufftDoubleComplex *EappliedField;
    cufftDoubleComplex *par_sum;
    cudaMalloc((void**) &par_sum, sizeof(double2)*gSize);

    std::cout << "gSize is: " << gSize << '\n';
    cufftResult result;
    cufftHandle plan_1d;
    cufftHandle plan_2d;
    cufftHandle plan_3d;
    cufftHandle plan_other2d;
    cufftHandle plan_dim2;
    cufftHandle plan_dim3;

    std::string buffer;
    double Rxy; //Condensate scaling factor.
    double a0x, a0y, a0z; //Harmonic oscillator length in x and y directions

    generate_grid(par);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    double mass = 1.4431607e-25; //Rb 87 mass, kg
    par.store("mass",mass);
    double a_s = 4.76e-9;
    par.store("a_s",a_s);

    double sum = 0.0;

    a0x = sqrt(HBAR/(2*mass*omegaX));
    a0y = sqrt(HBAR/(2*mass*omegaY));
    a0z = sqrt(HBAR/(2*mass*omegaZ));
    par.store("a0x",a0x);
    par.store("a0y",a0y);
    par.store("a0z",a0z);

    // Let's go ahead and define the gDensConst here
    // N*4*HBAR*HBAR*PI*(4.67e-9/mass)*sqrt(mass*(omegaZ)/(2*PI*HBAR)
    double gDenConst = N*4*HBAR*HBAR*PI*(4.67e-9/mass);
    if (dimnum == 2){
        gDenConst*= sqrt(mass*(omegaZ)/(2*PI*HBAR));
    }
    par.store("gDenConst", gDenConst);

    Rxy = pow(15,0.2)*pow(N*a_s*sqrt(mass*omegaZ/HBAR),0.2);
    par.store("Rxy",Rxy);

    //std::cout << "Rxy is: " << Rxy << '\n';
    double xMax, yMax, zMax;
    if (box_size > 0){
        xMax = box_size;
        yMax = box_size;
        zMax = box_size;
    }
    else{
        xMax = 6*Rxy*a0x;
        yMax = 6*Rxy*a0y;
        zMax = 6*Rxy*a0z;
    }
    par.store("xMax",xMax);
    par.store("yMax",yMax);
    par.store("zMax",zMax);

    double pxMax, pyMax, pzMax;
    pxMax = (PI/xMax)*(xDim>>1);
    pyMax = (PI/yMax)*(yDim>>1);
    pzMax = (PI/zMax)*(zDim>>1);
    par.store("pyMax",pyMax);
    par.store("pxMax",pxMax);
    par.store("pzMax",pzMax);

    double dx = xMax/(xDim>>1);
    double dy = yMax/(yDim>>1);
    double dz = zMax/(zDim>>1);
    if (dimnum == 2){
        dz = 1;
    }
    par.store("dx",dx);
    par.store("dy",dy);
    par.store("dz",dz);

    double dpx, dpy, dpz;
    dpx = PI/(xMax);
    dpy = PI/(yMax);
    dpz = PI/(zMax);
    //std::cout << "yMax is: " << yMax << '\t' << "xMax is: " << xMax << '\n';
    //std::cout << "dpx and dpy are:" << '\n';
    //std::cout << dpx << '\t' << dpy << '\n';
    par.store("dpx",dpx);
    par.store("dpy",dpy);
    par.store("dpz",dpz);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    /* Initialise wavefunction, momentum, position, angular momentum,
       imaginary and real-time evolution operators . */
    Energy = (double*) malloc(sizeof(double) * gSize);
    r = (double *) malloc(sizeof(double) * gSize);
    V_opt = (double *) malloc(sizeof(double) * gSize);
    EV_opt = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    Bz = (double *) malloc(sizeof(double) * gSize);
    Bx = (double *) malloc(sizeof(double) * gSize);
    By = (double *) malloc(sizeof(double) * gSize);
    EappliedField = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) *
                                                         gSize);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    #ifdef __linux
    int cores = omp_get_num_procs();
    par.store("Cores_Total",cores);

    // Assuming dev system specifics (Xeon with HT -> cores detected / 2)
    par.store("Cores_Max",cores/2);
    omp_set_num_threads(cores/2);

    //#pragma omp parallel for private(j)
    #endif

    par.store("gSize", xDim*yDim*zDim);
    generate_fields(par);
    double *K = par.dsval("K");
    double *Ax = par.dsval("Ax");
    double *Ay = par.dsval("Ay");
    double *Az = par.dsval("Az");
    double *V = par.dsval("V");

    double *pAx = par.dsval("pAx");
    double *pAy = par.dsval("pAy");
    double *pAz = par.dsval("pAz");

    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double *z = par.dsval("z");

    double2 *GpAx = par.cufftDoubleComplexval("GpAx");
    double2 *GpAy = par.cufftDoubleComplexval("GpAy");
    double2 *GpAz = par.cufftDoubleComplexval("GpAz");
    double2 *EpAx = par.cufftDoubleComplexval("EpAx");
    double2 *EpAy = par.cufftDoubleComplexval("EpAy");
    double2 *EpAz = par.cufftDoubleComplexval("EpAz");

    double2 *GV = par.cufftDoubleComplexval("GV");
    double2 *EV = par.cufftDoubleComplexval("EV");
    double2 *GK = par.cufftDoubleComplexval("GK");
    double2 *EK = par.cufftDoubleComplexval("EK");

    wfc = par.cufftDoubleComplexval("wfc");

    int index = 0;
    for(int i=0; i < gSize; i++ ){
        sum+=sqrt(wfc[i].x*wfc[i].x + wfc[i].y*wfc[i].y);
    }

    if (write_file){
        if (dimnum == 2){
            Bz = curl2d(par, Ax, Ay);
        }
        if (dimnum == 3){
            std::cout << "Calculating the 3d curl..." << '\n';
                    Bx = curl3d_x(par, Ax, Ay, Az);
                    By = curl3d_y(par, Ax, Ay, Az);
                    Bz = curl3d_z(par, Ax, Ay, Az);
                    std::cout << "Finished calculating Curl" << '\n';
        }
        std::cout << "writing initial variables to file..." << '\n';
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
        //hdfWriteDouble(xDim, V, 0, "V_0"); //HDF COMING SOON!
        //hdfWriteComplex(xDim, wfc, 0, "wfc_0");
        FileIO::writeOutDouble(buffer, data_dir + "V",V,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "K",K,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "pAy",pAy,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "pAx",pAx,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "Ax",Ax,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "Ay",Ay,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "Az",Az,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "x",x,xDim,0);
        FileIO::writeOutDouble(buffer, data_dir + "y",y,yDim,0);
        FileIO::writeOutDouble(buffer, data_dir + "z",z,zDim,0);
        FileIO::writeOutDouble(buffer, data_dir + "Bz",Bz,gSize,0);
        if (dimnum == 3){
            FileIO::writeOutDouble(buffer, data_dir + "Bx",Bx,gSize,0);
            FileIO::writeOutDouble(buffer, data_dir + "By",By,gSize,0);
        }
        FileIO::writeOut(buffer, data_dir + "WFC",wfc,gSize,0);
        FileIO::writeOut(buffer, data_dir + "EpAz",EpAz,gSize,0);
        FileIO::writeOut(buffer, data_dir + "EpAy",EpAy,gSize,0);
        FileIO::writeOut(buffer, data_dir + "EpAx",EpAx,gSize,0);
        FileIO::writeOut(buffer, data_dir + "GK",GK,gSize,0);
        FileIO::writeOut(buffer, data_dir + "GV",GV,gSize,0);
        FileIO::writeOut(buffer, data_dir + "GpAx",GpAx,gSize,0);
        FileIO::writeOut(buffer, data_dir + "GpAy",GpAy,gSize,0);
        FileIO::writeOut(buffer, data_dir + "GpAz",GpAz,gSize,0);
    }

    if (par.bval("read_wfc") == false){
        sum=sqrt(sum*dx*dy*dz);
        for (int i = 0; i < gSize; i++){
            wfc[i].x = (wfc[i].x)/(sum);
            wfc[i].y = (wfc[i].y)/(sum);
        }
    }

    result = cufftPlan2d(&plan_2d, xDim, yDim, CUFFT_Z2Z);
    if(result != CUFFT_SUCCESS){
        printf("Result:=%d\n",result);
        printf("Error: Could not execute cufftPlan2d(%s, %d, %d).\n", "plan_2d",
                (unsigned int)xDim, (unsigned int)yDim);
        return -1;
    }
    generate_plan_other2d(&plan_other2d, par);

    generate_plan_other3d(&plan_1d, par, 0);
    generate_plan_other3d(&plan_dim2, par, 1);
    generate_plan_other3d(&plan_dim3, par, 2);
    result = cufftPlan3d(&plan_3d, xDim, yDim, zDim, CUFFT_Z2Z);
    if(result != CUFFT_SUCCESS){
        printf("Result:=%d\n",result);
        printf("Error: Could not execute cufftPlan3d(%s, %d, %d, %d).\n", 
                "plan_3d",
                (unsigned int)xDim, (unsigned int)yDim, (unsigned int) zDim);
        return -1;
    }

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    //std::cout << GV[0].x << '\t' << GK[0].x << '\t'
    //          << pAy[0] << '\t' << pAx[0] << '\n';

    //std::cout << "storing variables..." << '\n';

    // Storing variables that have been initialized
    // Re-establishing variables from parsed Grid class
    // Initializes uninitialized variables to 0 values
    par.store("Energy", Energy);
    par.store("r", r);
    par.store("Energy_gpu", Energy_gpu);
    par.store("wfc", wfc);
    par.store("EV_opt", EV_opt);
    par.store("V_opt", V_opt);
    par.store("wfc_backup", wfc_backup);
    par.store("EappliedField", EappliedField);
    par.store("par_sum", par_sum);

    par.store("result", result);
    par.store("plan_1d", plan_1d);
    par.store("plan_2d", plan_2d);
    par.store("plan_other2d", plan_other2d);
    par.store("plan_3d", plan_3d);
    par.store("plan_dim2", plan_dim2);
    par.store("plan_dim3", plan_dim3);

    std::cout << "variables stored" << '\n';

    return 0;
}

// initializing all variables for 3d
int init_3d(Op &opr, Grid &par, Wave &wave){

    int max_threads = 128;

    // Setting functions for operators
    set_fns(par, opr, wave);
    //par.set_fns();
    //par.set_fns();

    // Re-establishing variables from parsed Grid class
    // Initializes uninitialized variables to 0 values
    std::string data_dir = par.sval("data_dir");
    int N = par.ival("atoms");
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int zDim = par.ival("zDim");
    dim3 threads(max_threads,1,1);
    bool write_file = par.bval("write_file");
    unsigned int gSize = xDim*yDim*zDim;
    double omega = par.dval("omega");
    double gdt = par.dval("gdt");
    double dt = par.dval("dt");
    double omegaX = par.dval("omegaX");
    double omegaY = par.dval("omegaY");
    double omegaZ = par.dval("omegaZ");
    double box_size = par.dval("box_size");
    double gammaY = par.dval("gammaY"); //Aspect ratio of trapping geometry.
    double l = par.dval("winding");
    double *x;
    double *y;
    double *z;
    double *xp;
    double *yp;
    double *zp;
    double *Energy;
    double *r;
    double *V;
    double *V_opt;
    double *Phi;
    double *Phi_gpu;
    double *K;
    double *pAy;
    double *pAx;
    double *pAz;
    double *Ax;
    double *Ay;
    double *Az;
    double *pAy_gpu;
    double *pAx_gpu;
    double *pAz_gpu;
    double *Energy_gpu;
    cufftDoubleComplex *wfc;
    if (par.bval("read_wfc") == true){
        wfc = par.cufftDoubleComplexval("wfc");
    }
    cufftDoubleComplex *V_gpu;
    cufftDoubleComplex *EV_opt;
    cufftDoubleComplex *wfc_backup;
    cufftDoubleComplex *GK;
    cufftDoubleComplex *GV;
    cufftDoubleComplex *GpAx;
    cufftDoubleComplex *GpAy;
    cufftDoubleComplex *GpAz;
    cufftDoubleComplex *EV;
    cufftDoubleComplex *EK;
    cufftDoubleComplex *EpAy;
    cufftDoubleComplex *EpAx;
    cufftDoubleComplex *EpAz;
    cufftDoubleComplex *EappliedField;
    cufftDoubleComplex *wfc_gpu;
    cufftDoubleComplex *K_gpu;
    cufftDoubleComplex *par_sum;

    //std::cout << omegaX << '\t' << omegaY << '\n';
    //std::cout << "xDim is: " << xDim << '\t' <<  "yDim is: " << yDim << '\t'
    //          << "zDim is: " << zDim << '\n';

    cufftResult result;
    //cufftHandle plan_1d;
    cufftHandle plan_3d;

    dim3 grid = par.grid;

    std::string buffer;

    double Rxy; //Condensate scaling factor.
    double a0x, a0y, a0z; //Harmonic oscillator length in x and y directions

    int xD = 1, yD = 1, zD = 1;

    if (xDim <= max_threads){
        threads.x = xDim;
        threads.y = 1;
        threads.z = 1;

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

        threads.x = dim_tmp;
        threads.y = 1;
        threads.z = 1;
        xD = pow(2,count);
        yD = yDim;
        zD = zDim;
    }

    std::cout << "threads in x are: " << threads.x << '\n';
    std::cout << "dimensions are: " << xD << '\t' << yD << '\t' << zD << '\n';

    grid.x=xD;
    grid.y=yD;
    grid.z=zD;

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    int i, j, k; //Used in for-loops for indexing


    double mass = 1.4431607e-25; //Rb 87 mass, kg
    par.store("mass",mass);
    double a_s = 4.76e-9;
    par.store("a_s",a_s);

    // Let's go ahead and define the gDensConst here
    // N*4*HBAR*HBAR*PI*(4.67e-9/mass)*sqrt(mass*(omegaZ)/(2*PI*HBAR)
    double gDenConst = N*4*HBAR*HBAR*PI*(4.76e-9/mass);
    par.store("gDenConst", gDenConst);


    double sum = 0.0;

    a0x = pow(HBAR/(2*mass*omegaX), 0.5);
    a0y = pow(HBAR/(2*mass*omegaY), 0.5);
    a0z = pow(HBAR/(2*mass*omegaZ), 0.5);
    par.store("a0x",a0x);
    par.store("a0y",a0y);
    par.store("a0z",a0z);

    //std::cout << "a0x and y are: " << a0x << '\t' << a0y << '\n';

    //std::cout << N << '\t' << a_s << '\t' << mass << '\t' << omegaZ << '\n';

    Rxy = pow(15,0.2)*pow(N*a_s*sqrt(mass*omegaZ/HBAR),0.2);
    par.store("Rxy",Rxy);
    double bec_length = sqrt( HBAR/(mass*sqrt( omegaX*omegaX *
                                               ( 1 - omega*omega) ) ));

    //std::cout << "Rxy is: " << Rxy << '\n';
    double xMax = box_size;
    double yMax = box_size;
    double zMax = box_size;
    par.store("xMax",xMax);
    par.store("yMax",yMax);
    par.store("zMax",zMax);

    double pxMax, pyMax, pzMax;
    pxMax = (PI/xMax)*(xDim>>1);
    pyMax = (PI/yMax)*(yDim>>1);
    pzMax = (PI/zMax)*(zDim>>1);
    par.store("pyMax",pyMax);
    par.store("pxMax",pxMax);
    par.store("pzMax",pzMax);

    double dx = xMax/(xDim>>1);
    double dy = yMax/(yDim>>1);
    double dz = zMax/(zDim>>1);
    par.store("dx",dx);
    par.store("dy",dy);
    par.store("dz",dz);

    double dpx, dpy, dpz;
    dpx = PI/(xMax);
    dpy = PI/(yMax);
    dpz = PI/(zMax);
    //std::cout << "yMax is: " << yMax << '\t' << "xMax is: " << xMax << '\n';
    //std::cout << "dpx and dpy are:" << '\n';
    //std::cout << dpx << '\t' << dpy << '\n';
    par.store("dpx",dpx);
    par.store("dpy",dpy);
    par.store("dpz",dpz);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    //double *x,*y,*xp,*yp;
    x = (double *) malloc(sizeof(double) * xDim);
    y = (double *) malloc(sizeof(double) * yDim);
    z = (double *) malloc(sizeof(double) * zDim);
    xp = (double *) malloc(sizeof(double) * xDim);
    yp = (double *) malloc(sizeof(double) * yDim);
    zp = (double *) malloc(sizeof(double) * zDim);

    //std::cout << "dx and dy are: " << '\n';
    //std::cout << dx << '\t' << dy << '\n';
    // creating x,y,z,xp,yp,zp
    for(i=0; i<xDim/2; ++i){
        x[i] = -xMax + i*dx;
        x[i + (xDim/2)] = i*dx;

        xp[i] = i*dpx;
        xp[i + (xDim/2)] = -pxMax + i*dpx;

    }
    for(i=0; i<yDim/2; ++i){
        y[i] = -yMax + i*dy;
        y[i + (yDim/2)] = i*dy;

        yp[i] = i*dpy;
        yp[i + (yDim/2)] = -pyMax + i*dpy;

    }
    for(i=0; i<zDim/2; ++i){
        z[i] = -zMax + i*dz;
        z[i + (zDim/2)] = i*dz;

        zp[i] = i*dpz;
        zp[i + (zDim/2)] = -pzMax + i*dpz;

    }

    par.store("x", x);
    par.store("y", y);
    par.store("z", z);
    par.store("xp", xp);
    par.store("yp", yp);
    par.store("zp", zp);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    Energy = (double*) malloc(sizeof(double) * gSize);
    r = (double *) malloc(sizeof(double) * gSize);
    Phi = (double *) malloc(sizeof(double) * gSize);
    if (par.bval("read_wfc") == false){
        wfc = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    }
    wfc_backup = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) *
                                               (gSize/threads.x));
    K = (double *) malloc(sizeof(double) * gSize);
    V = (double *) malloc(sizeof(double) * gSize);
    V_opt = (double *) malloc(sizeof(double) * gSize);
    GK = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    GV = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    GpAx = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    GpAy = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    GpAz = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    EK = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    EV = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    EV_opt = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    pAy = (double *) malloc(sizeof(double) * gSize);
    pAx = (double *) malloc(sizeof(double) * gSize);
    pAz = (double *) malloc(sizeof(double) * gSize);
    Ax = (double *) malloc(sizeof(double) * gSize);
    Ay = (double *) malloc(sizeof(double) * gSize);
    Az = (double *) malloc(sizeof(double) * gSize);
    EpAy = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    EpAx = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    EpAz = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    EappliedField = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) *
                                                         gSize);

    cudaMalloc((void**) &Energy_gpu, sizeof(double) * gSize);
    cudaMalloc((void**) &wfc_gpu, sizeof(cufftDoubleComplex) * gSize);
    cudaMalloc((void**) &Phi_gpu, sizeof(double) * gSize);
    cudaMalloc((void**) &K_gpu, sizeof(cufftDoubleComplex) * gSize);
    cudaMalloc((void**) &V_gpu, sizeof(cufftDoubleComplex) * gSize);
    cudaMalloc((void**) &pAy_gpu, sizeof(cufftDoubleComplex) * gSize);
    cudaMalloc((void**) &pAx_gpu, sizeof(cufftDoubleComplex) * gSize);
    cudaMalloc((void**) &pAz_gpu, sizeof(cufftDoubleComplex) * gSize);
    cudaMalloc((void**) &par_sum, sizeof(cufftDoubleComplex)*(gSize/threads.x));
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    //std::cout << "all variables malloc'd" << '\n';

    #ifdef __linux
    int cores = omp_get_num_procs();
    par.store("Cores_Total",cores);

    // Assuming dev system specifics (Xeon with HT -> cores detected / 2)
    par.store("Cores_Max",cores/2);
    omp_set_num_threads(cores/2);
    //std::cout << "GAMMAY IS: " << gammaY << '\n';
    //#pragma omp parallel for private(k)
    #endif
    int index;
    // Setting Ax, Ay, and Az if from file
    if (par.Afn == "file"){
        file_A(par.Axfile, Ax, omega);
        par.store("Ax",Ax);

        file_A(par.Ayfile, Ay, omega);
        par.store("Ay", Ay);

        file_A(par.Azfile, Az, omega);
        par.store("Az", Az);

        std::cout << "finished reading Ax / Ay / Az from file" << '\n';
    }
    for( i=0; i < xDim; i++ ){
        for( j=0; j < yDim; j++ ){
            for( k=0; k < zDim; k++ ){
                index = (i * yDim * zDim) + (j * zDim) + k;
                if (par.Afn == "rotation"){
                    Phi[index] = fmod(l*atan2(y[j], x[i]),2*PI);
                }
                else if (par.Vfn == "torus"){
                    double xOffset = par.dval("x0_shift");
                    double yOffset = par.dval("y0_shift");
                    double rMax = par.dval("xMax");
                    double fudge = par.dval("fudge");
                    double x_loc = x[i] - xOffset;
                    double y_loc = y[j] - yOffset;
                    double radius = sqrt(x_loc*x_loc + y_loc*y_loc)
                                    - 0.5*rMax*fudge;
                    Phi[index] = fmod(l*atan2(z[k], radius),2*PI);
                }
                else{
                    Phi[index] = 0.0;
                }

                if (par.bval("read_wfc") != true){
                    wfc[index] = wave.Wfc_fn(par, Phi[index],i,j,k);
                    sum+=sqrt(wfc[index].x*wfc[index].x + 
                              wfc[index].y*wfc[index].y);
                }
                
                V[index] = opr.V_fn(par, opr, i, j, k);
                K[index] = opr.K_fn(par, opr, i, j, k);
    
                GV[index].x = exp( -V[index]*(gdt/(2*HBAR)));
                GK[index].x = exp( -K[index]*(gdt/HBAR));
                GV[index].y = 0.0;
                GK[index].y = 0.0;

                // Ax and Ay will be calculated here but are used only for
                // debugging. They may be needed later for magnetic field calc
                if (par.Afn != "file"){
                    Ax[index] = opr.Ax_fn(par, opr, i, j, k);
                    Ay[index] = opr.Ay_fn(par, opr, i, j, k);
                    Az[index] = opr.Az_fn(par, opr, i, j, k);
                }
                
                pAy[index] = pAy_fn(par, opr, i, j, k);
                pAx[index] = pAx_fn(par, opr, i, j, k);
                pAz[index] = pAz_fn(par, opr, i, j, k);
    
                GpAx[index].x = exp(-pAx[index]*gdt);
                GpAx[index].y = 0;
                GpAy[index].x = exp(-pAy[index]*gdt);
                GpAy[index].y = 0;
                GpAz[index].x = exp(-pAz[index]*gdt);
                GpAz[index].y = 0;

                EV[index].x=cos( -V[index]*(dt/(2*HBAR)));
                EV[index].y=sin( -V[index]*(dt/(2*HBAR)));
                EK[index].x=cos( -K[index]*(dt/HBAR));
                EK[index].y=sin( -K[index]*(dt/HBAR));

                EpAy[index].x=cos(-pAy[index]*dt);
                EpAy[index].y=sin(-pAy[index]*dt);
                EpAx[index].x=cos(-pAx[index]*dt);
                EpAx[index].y=sin(-pAx[index]*dt);
                EpAz[index].x=cos(-pAz[index]*dt);
                EpAz[index].y=sin(-pAz[index]*dt);

            }
        }
    }

    if (write_file){
        // Calculating the curl 
        std::cout << "Calculating the 3d curl..." << '\n';
        double *Bx = curl3d_x(par, Ax, Ay, Az);
        double *By = curl3d_y(par, Ax, Ay, Az);
        double *Bz = curl3d_z(par, Ax, Ay, Az);
        std::cout << "Finished calculating Curl" << '\n';

        std::cout << "writing initial variables to file..." << '\n';
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
        //hdfWriteDouble(xDim, V, 0, "V_0"); //HDF COMING SOON!
        //hdfWriteComplex(xDim, wfc, 0, "wfc_0");
        //FileIO::writeOutDouble(buffer, data_dir + "V_opt",V_opt,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "V",V,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "K",K,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "pAy",pAy,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "pAx",pAx,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "pAz",pAz,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "Ax",Ax,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "Ay",Ay,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "Az",Az,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "Bz",Bz,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "By",By,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "Bx",By,gSize,0);
        FileIO::writeOut(buffer, data_dir + "WFC",wfc,gSize,0);
        FileIO::writeOut(buffer, data_dir + "EpAy",EpAy,gSize,0);
        FileIO::writeOut(buffer, data_dir + "EpAx",EpAx,gSize,0);
        FileIO::writeOut(buffer, data_dir + "EpAz",EpAz,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "Phi",Phi,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "r",r,gSize,0);
        FileIO::writeOutDouble(buffer, data_dir + "x",x,xDim,0);
        FileIO::writeOutDouble(buffer, data_dir + "y",y,yDim,0);
        FileIO::writeOutDouble(buffer, data_dir + "z",z,zDim,0);
        FileIO::writeOutDouble(buffer, data_dir + "px",xp,xDim,0);
        FileIO::writeOutDouble(buffer, data_dir + "py",yp,yDim,0);
        FileIO::writeOutDouble(buffer, data_dir + "pz",zp,zDim,0);
        FileIO::writeOut(buffer, data_dir + "GK",GK,gSize,0);
        FileIO::writeOut(buffer, data_dir + "GV",GV,gSize,0);
        FileIO::writeOut(buffer, data_dir + "GpAx",GpAx,gSize,0);
        FileIO::writeOut(buffer, data_dir + "GpAy",GpAy,gSize,0);
        FileIO::writeOut(buffer, data_dir + "GpAz",GpAz,gSize,0);
    }

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    //std::cout << "wrote initial variables" << '\n';

    //free(V);
    free(K); free(r); free(Phi);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    if (par.bval("read_wfc") == false){
        sum=sqrt(sum*dx*dy*dz);
        //#pragma omp parallel for reduction(+:sum) private(j)
        for (i = 0; i < xDim; i++){
            for (j = 0; j < yDim; j++){
                for (k = 0; k < zDim; k++){
                    index = i * yDim * zDim + j * zDim + k;
                    wfc[index].x = (wfc[index].x)/(sum);
                    wfc[index].y = (wfc[index].y)/(sum);
                }
            }
        }
    }

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    //std::cout << "xDim is: " << xDim << '\t' << "yDim is: " << yDim << '\n';
    //std::cout << "plan_2d is: " << plan_2d << '\n';
    result = cufftPlan3d(&plan_3d, xDim, yDim, zDim, CUFFT_Z2Z);
    //std::cout << "found result" << '\n';
    if(result != CUFFT_SUCCESS){
        printf("Result:=%d\n",result);
        printf("Error: Could not execute cufftPlan3d(%s, %d, %d, %d).\n", 
                "plan_3d",
                (unsigned int)xDim, (unsigned int)yDim, (unsigned int) zDim);
        return -1;
    }

    cufftHandle plan_1d;
    generate_plan_other3d(&plan_1d, par, 0);
    cufftHandle plan_dim2;
    generate_plan_other3d(&plan_dim2, par, 1);
    cufftHandle plan_dim3;
    generate_plan_other3d(&plan_dim3, par, 2);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    //std::cout << GV[0].x << '\t' << GK[0].x << '\t'
    //          << pAy[0] << '\t' << pAx[0] << '\n';

    //std::cout << "storing variables..." << '\n';

    // Storing variables that have been initialized
    // Re-establishing variables from parsed Grid class
    // Initializes uninitialized variables to 0 values
    par.store("omega", omega);
    par.store("gdt", gdt);
    par.store("dt", dt);
    par.store("omegaX", omegaX);
    par.store("omegaY", omegaY);
    par.store("omegaZ", omegaZ);
    par.store("dx", dx);
    par.store("dy", dy);
    par.store("dz", dz);
    par.store("xMax", xMax);
    par.store("yMax", yMax);
    par.store("zMax", zMax);
    par.store("winding", l);
    par.store("x", x);
    par.store("y", y);
    par.store("z", z);
    par.store("xp", xp);
    par.store("yp", yp);
    par.store("zp", zp);
    par.store("Energy", Energy);
    par.store("r", r);
    par.store("V", V);
    par.store("V_opt", V_opt);
    par.store("Phi", Phi);
    par.store("Phi_gpu", Phi_gpu);
    par.store("K", K);
    par.store("pAy", pAy);
    par.store("pAx", pAx);
    par.store("pAz", pAz);
    par.store("Energy_gpu", Energy_gpu);
    par.store("atoms", N);
    par.store("wfc", wfc);
    par.store("V_gpu", V_gpu);
    par.store("EV_opt", EV_opt);
    par.store("wfc_backup", wfc_backup);
    par.store("GK", GK);
    par.store("GV", GV);
    par.store("GpAx", GpAx);
    par.store("GpAy", GpAy);
    par.store("GpAz", GpAz);
    par.store("EV", EV);
    par.store("EK", EK);
    par.store("EpAy", EpAy);
    par.store("EpAx", EpAx);
    par.store("EpAz", EpAz);
    par.store("EappliedField", EappliedField);
    par.store("wfc_gpu", wfc_gpu);
    par.store("K_gpu", K_gpu);
    par.store("pAy_gpu", pAy_gpu);
    par.store("pAx_gpu", pAx_gpu);
    par.store("pAz_gpu", pAz_gpu);
    par.store("par_sum", par_sum);

    par.store("result", result);
    par.store("plan_1d", plan_1d);
    par.store("plan_3d", plan_3d);
    par.store("plan_dim2", plan_dim2);
    par.store("plan_dim3", plan_dim3);

    par.threads = threads;
    par.grid = grid;

    std::cout << "variables stored" << '\n';

    return 0;
}


int main(int argc, char **argv){

    Grid par = parseArgs(argc,argv);
    //Grid par2 = parseArgs(argc,argv);
    Wave wave;
    Op opr;

    int device = par.ival("device");
    int dimnum = par.ival("dimnum");
    cudaSetDevice(device);

    std::string buffer;
    time_t start,fin;
    time(&start);
    printf("Start: %s\n", ctime(&start));

    //************************************************************//
    /*
    * Initialise the Params data structure to track params and variables
    */
    //************************************************************//

    // If we want to read in a wfc, we may also need to imprint a phase. This
    // will be done in the init_2d and init_3d functions
    // We need a number of parameters for now
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int zDim = par.ival("zDim");
    if(par.bval("read_wfc") == true){

        // Initializing the wfc
        int gSize = xDim * yDim * zDim;
        cufftDoubleComplex *wfc;
        //wfc = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);

        std::string infile = par.sval("infile");
        std::string infilei = par.sval("infilei");
        printf("Loading wavefunction...");
        wfc=FileIO::readIn(infile,infilei,gSize);
        par.store("wfc",wfc);
        printf("Wavefunction loaded.\n");
        //std::string data_dir = par.sval("data_dir");
        //FileIO::writeOut(buffer, data_dir + "WFC_CHECK",wfc,gSize,0);
    }

    if (dimnum == 2){
        init(opr, par, wave);
    }
    else{
        //init_3d(opr, par2, wave);
        init(opr, par, wave);
    }

/*
    std::cout
    << par.ival("plan_3d") << par2.ival("plan_3d") << '\n'
    << par.ival("plan_1d") << par2.ival("plan_1d") << '\n'
    << par.ival("plan_2d") << par2.ival("plan_2d") << '\n'
    << par.ival("plan_other2d") << par2.ival("plan_other2d") << '\n'
    << par.ival("plan_dim1") << par2.ival("plan_dim1") << '\n'
    << par.ival("plan_dim2") << par2.ival("plan_dim2") << '\n';

    std::cout 
    << (par.cufftDoubleComplexval("GpAx") == par2.cufftDoubleComplexval("GpAx")) << '\n'
    << (par.cufftDoubleComplexval("GpAy") == par2.cufftDoubleComplexval("GpAy")) << '\n'
    << (par.cufftDoubleComplexval("GpAz") == par2.cufftDoubleComplexval("GpAz")) << '\n'
    << (par.dsval("V") == par2.dsval("V")) << '\n'
    << (par.dsval("K") == par2.dsval("K")) << '\n';
*/
    //std::cout << "initialized" << '\n';


    // Re-establishing variables from parsed Grid class
    // Note that 3d variables are set to nullptr's unless needed
    //      This might need to be fixed later
    std::string data_dir = par.sval("data_dir");
    double dx = par.dval("dx");
    double dy = par.dval("dy");
    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double *V_opt = par.dsval("V_opt");
    double *pAy = par.dsval("pAy");
    double *pAx = par.dsval("pAx");
    double2 *pAy_gpu;
    double2 *pAx_gpu;
    double2 *pAz_gpu;
    double2 *V_gpu;
    double2 *K_gpu;
    int gsteps = par.ival("gsteps");
    int esteps = par.ival("esteps");
    cufftDoubleComplex *wfc = par.cufftDoubleComplexval("wfc");
    cufftDoubleComplex *GK = par.cufftDoubleComplexval("GK");
    cufftDoubleComplex *GV = par.cufftDoubleComplexval("GV");
    cufftDoubleComplex *GpAx = par.cufftDoubleComplexval("GpAx");
    cufftDoubleComplex *GpAy = par.cufftDoubleComplexval("GpAy");
    cufftDoubleComplex *GpAz = nullptr;
    cufftDoubleComplex *EV = par.cufftDoubleComplexval("EV");
    cufftDoubleComplex *EK = par.cufftDoubleComplexval("EK");
    cufftDoubleComplex *EpAy = par.cufftDoubleComplexval("EpAy");
    cufftDoubleComplex *EpAx = par.cufftDoubleComplexval("EpAx");
    cufftDoubleComplex *EpAz = nullptr;
    cufftDoubleComplex *wfc_gpu = par.cufftDoubleComplexval("wfc_gpu");
    cufftDoubleComplex *par_sum = par.cufftDoubleComplexval("par_sum");
    cudaError_t err;
    int gsize = xDim * yDim;

    // Special variables for the 3d case
    if (dimnum == 3){
        double dz = par.dval("dz");
        double *z = par.dsval("z");
        double *pAz = par.dsval("pAz");
        cufftDoubleComplex *GpAz = par.cufftDoubleComplexval("GpAz");
        cufftDoubleComplex *EpAz = par.cufftDoubleComplexval("EpAz");
        gsize = xDim*yDim*zDim;
    }
    cudaMalloc((void**) &V_gpu, sizeof(double2)*gsize);
    cudaMalloc((void**) &K_gpu, sizeof(double2)*gsize);
    cudaMalloc((void**) &pAx_gpu, sizeof(double2)*gsize);
    cudaMalloc((void**) &pAy_gpu, sizeof(double2)*gsize);
    cudaMalloc((void**) &pAz_gpu, sizeof(double2)*gsize);

    std::cout << "variables re-established" << '\n';
    //std::cout << read_wfc << '\n';

    //************************************************************//
    /*
    * Groundstate finder section
    */
    //************************************************************//
    if (par.bval("write_file")){
        FileIO::writeOutParam(buffer, par, data_dir + "Params.dat");
    }

    if(gsteps > 0){
        err=cudaMemcpy(K_gpu, GK, sizeof(cufftDoubleComplex)*gsize,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy K_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(V_gpu, GV, sizeof(cufftDoubleComplex)*gsize,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy V_gpu to device" << '\n';
            exit(1);
        }
        if (par.bval("write_file")){
            FileIO::writeOut(buffer, data_dir + "GK1",GK,gsize,0);
            FileIO::writeOut(buffer, data_dir + "GV1",GV,gsize,0);
        }
        err=cudaMemcpy(pAy_gpu, GpAy, sizeof(cufftDoubleComplex)*gsize,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy pAy_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(pAx_gpu, GpAx, sizeof(cufftDoubleComplex)*gsize,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy pAx_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(wfc_gpu, wfc, sizeof(cufftDoubleComplex)*gsize,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy wfc_gpu to device" << '\n';
            exit(1);
        }
        par.store("pAx", pAx);
        par.store("pAy", pAy);
        par.store("GK", GK);
        par.store("GV", GV);
        par.store("wfc", wfc);
        par.store("K_gpu", K_gpu);
        par.store("V_gpu", V_gpu);
        par.store("wfc_gpu", wfc_gpu);
        par.store("pAy_gpu", pAy_gpu);
        par.store("pAx_gpu", pAx_gpu);

        // Special cases for 3d
        if (dimnum == 3){
            GpAz = par.cufftDoubleComplexval("GpAz");
            err=cudaMemcpy(pAz_gpu, GpAz, sizeof(cufftDoubleComplex)*gsize,
                           cudaMemcpyHostToDevice);

            if(err!=cudaSuccess){
                std::cout << "ERROR: Could not copy pAz_gpu to device" << '\n';
                exit(1);
            }
            par.store("pAz_gpu", pAz_gpu);

            evolve_3d(wave, opr, par_sum,
                      gsteps, 0, par, buffer);
        }
        if (dimnum == 2){
            evolve_2d(wave, opr, par_sum,
                      gsteps,  0, par, buffer);
        }
        wfc = par.cufftDoubleComplexval("wfc");
        wfc_gpu = par.cufftDoubleComplexval("wfc_gpu");
        cudaMemcpy(wfc, wfc_gpu, sizeof(cufftDoubleComplex)*gsize,
                   cudaMemcpyDeviceToHost);
    }

    std::cout << GV[0].x << '\t' << GK[0].x << '\t'
              << pAy[0] << '\t' << pAx[0] << '\n';

    std::cout << "evolution started..." << '\n';
    std::cout << "esteps: " << esteps << '\n';

    //************************************************************//
    /*
    * Evolution
    */
    //************************************************************//
    if(esteps > 0){
        err=cudaMemcpy(pAy_gpu, EpAy, sizeof(cufftDoubleComplex)*gsize,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy pAy_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(pAx_gpu, EpAx, sizeof(cufftDoubleComplex)*gsize,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy pAx_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(K_gpu, EK, sizeof(cufftDoubleComplex)*gsize,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy K_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(V_gpu, EV, sizeof(cufftDoubleComplex)*gsize,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy V_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(wfc_gpu, wfc, sizeof(cufftDoubleComplex)*gsize,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy wfc_gpu to device" << '\n';
            exit(1);
        }

        par.store("pAx", pAx);
        par.store("pAy", pAy);
        par.store("EK", EK);
        par.store("EV", EV);
        par.store("wfc", wfc);
        par.store("K_gpu", K_gpu);
        par.store("V_gpu", V_gpu);
        par.store("wfc_gpu", wfc_gpu);
        par.store("pAy_gpu", pAy_gpu);
        par.store("pAx_gpu", pAx_gpu);
        FileIO::writeOutDouble(buffer, data_dir + "V_opt",V_opt,gsize,0);
        // Special variables / instructions for 3d case
        if (dimnum == 3){
            pAz_gpu = par.cufftDoubleComplexval("pAz_gpu");
            EpAz = par.cufftDoubleComplexval("EpAz");
            err=cudaMemcpy(pAz_gpu, EpAz, sizeof(cufftDoubleComplex)*gsize,
                           cudaMemcpyHostToDevice);
            if(err!=cudaSuccess){
                std::cout << "ERROR: Could not copy pAz_gpu to device" << '\n';
                exit(1);
            }
            par.store("pAz_gpu", pAz_gpu);
            evolve_3d(wave, opr, par_sum,
                      esteps, 1, par, buffer);
        }
        if (dimnum == 2){
            evolve_2d(wave, opr, par_sum,
                      esteps, 1, par, buffer);
        }
        wfc = par.cufftDoubleComplexval("wfc");
        wfc_gpu = par.cufftDoubleComplexval("wfc_gpu");
    }
    std::cout << "done evolving" << '\n';
    free(EV); free(EK); free(EpAy); free(EpAx);
    free(x);free(y);
    cudaFree(wfc_gpu); cudaFree(K_gpu); cudaFree(V_gpu); cudaFree(pAx_gpu);
    cudaFree(pAy_gpu); cudaFree(par_sum);
    time(&fin);
    printf("Finish: %s\n", ctime(&fin));
    printf("Total time: %ld seconds\n ",(long)fin-start);
    std::cout << '\n';
    return 0;
}

