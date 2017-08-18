#include "../include/operators.h"
#include "../include/kernels.h"

double sign(double x){
    if (x < 0){
        return -1.0;
    }
    else if (x == 0){
        return 0.0;
    }
    else{
        return 1.0;
    }
}

// Function to take the curl of Ax and Ay in 2d
// note: This is on the cpu, there should be a GPU version too.
double *curl2d(Grid &par, double *Ax, double *Ay){
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");

    int size = sizeof(double) * xDim * yDim;
    double *curl;
    curl = (double *)malloc(size);

    int index;

    // Note: To take the curl, we need a change in x and y to create a dx or dy
    //       For this reason, we have added yDim to y and 1 to x
    for (int i = 0; i < xDim; i++){
        for (int j = 0; j < yDim-1; j++){
            index = j + yDim * i;
            curl[index] = (Ay[index] - Ay[index+xDim]) 
                          - (Ax[index] - Ax[index+1]);
        }
    }

    return curl;
}

// Function to take the curl of Ax and Ay in 2d
// note: This is on the cpu, there should be a GPU version too.
// Not complete yet!
double *curl3d_x(Grid &par, double *Ax, double *Ay, double *Az){
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int zDim = par.ival("zDim");

    int size = sizeof(double) * xDim * yDim * zDim;
    double *curl;
    curl = (double *)malloc(size);

    int index;

    // Note: To take the curl, we need a change in x and y to create a dx or dy
    //       For this reason, we have added yDim to y and 1 to x
    for (int i = 0; i < xDim; i++){
        for (int j = 0; j < yDim-1; j++){
            for (int k = 0; k < zDim - 1; k++){
                index = k + zDim * j + zDim * yDim * i;
                curl[index] = (Az[index] - Az[index + xDim])
                              -(Ay[index] - Ay[index + 1]);
            }
        }
    }

    return curl;
}

// Function to take the curl of Ax and Ay in 2d
// note: This is on the cpu, there should be a GPU version too.
// Not complete yet!
double *curl3d_y(Grid &par, double *Ax, double *Ay, double *Az){
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int zDim = par.ival("zDim");

    int size = sizeof(double) * xDim * yDim * zDim;
    double *curl;
    curl = (double *)malloc(size);

    int index;

    // Note: To take the curl, we need a change in x and y to create a dx or dy
    //       For this reason, we have added yDim to y and 1 to x
    for (int i = 0; i < xDim-1; i++){
        for (int j = 0; j < yDim; j++){
            for (int k = 0; k < zDim - 1; k++){
                index = k + zDim * j + zDim * yDim * i;
                curl[index] = -(Az[index] - Az[index + xDim*yDim])
                              +(Ax[index] - Ax[index + 1]);
            }
        }
    }

    return curl;
}

// Function to take the curl of Ax and Ay in 2d
// note: This is on the cpu, there should be a GPU version too.
// Not complete yet!
double *curl3d_z(Grid &par, double *Ax, double *Ay, double *Az){
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int zDim = par.ival("zDim");

    int size = sizeof(double) * xDim * yDim * zDim;
    double *curl;
    curl = (double *)malloc(size);

    int index;

    // Note: To take the curl, we need a change in x and y to create a dx or dy
    //       For this reason, we have added yDim to y and 1 to x
    for (int i = 0; i < xDim - 1; i++){
        for (int j = 0; j < yDim-1; j++){
            for (int k = 0; k < zDim; k++){
                index = k + zDim * j + zDim * yDim * i;
                curl[index] = (Ay[index] - Ay[index + xDim*yDim])
                              -(Ax[index] - Ax[index + xDim]);
            }
        }
    }

    return curl;
}

// Function for simple 2d rotation with i and j as the interators
double rotation_K(Grid &par, Op &opr, int i, int j, int k){
    double *xp = par.dsval("xp");
    double *yp = par.dsval("yp");
    double mass = par.dval("mass");
    return (HBAR*HBAR/(2*mass))*(xp[i]*xp[i] + yp[j]*yp[j]);
}

// A simple 3d rotation with i, j, and k as integers
double rotation_K3d(Grid &par, Op &opr, int i, int j, int k){
    double *xp = par.dsval("xp");
    double *yp = par.dsval("yp");
    double *zp = par.dsval("zp");
    double mass = par.dval("mass");
    return (HBAR*HBAR/(2*mass))*(xp[i]*xp[i] + yp[j]*yp[j] + zp[k]*zp[k]);
}

// Function for simple 2d rotation with i and j as the interators dimensionless
double rotation_K_dimensionless(Grid &par, Op &opr, int i, int j, int k){
    double *xp = par.dsval("xp");
    double *yp = par.dsval("yp");
    return (xp[i]*xp[i] + yp[j]*yp[j])*0.5;
}


// Function for simple 2d rotation with i and j as the interators
double rotation_gauge_K(Grid &par, Op &opr, int i, int j, int k){
    double *xp = par.dsval("xp");
    double *yp = par.dsval("yp");
    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double omega = par.dval("omega");
    double omegaX = par.dval("omegaX");
    double omega_0 = omega * omegaX;
    double mass = par.dval("mass");
    double p1 = HBAR*HBAR*(xp[i]*xp[i] + yp[j]*yp[j]);
    double p2 = mass*mass*omega_0*omega_0*(x[i]*x[i] + y[j]*y[j]);
    double p3 = 2*HBAR*mass*omega_0*(xp[i]*y[j] - yp[j]*x[i]);

    return (1/(2*mass))*(p1 + p2 + p3) *0.5;
}

// Function for simple 2d harmonic V with i and j as the iterators
double harmonic_V(Grid &par, Op &opr, int i , int j, int k){
    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double omegaX = par.dval("omegaX");
    double omegaY = par.dval("omegaY");
    double gammaY = par.dval("gammaY");
    double yOffset = 0.0;
    double xOffset = 0.0;
    double mass = par.dval("mass");
    double V_x = omegaX*(x[i]+xOffset); 
    double V_y = gammaY*omegaY*(y[j]+yOffset);
    if (par.Afn != "file"){
        return 0.5 * mass * (( V_x * V_x + V_y * V_y) + 
                             pow(opr.Ax_fn(par, opr, i, j, k),2) + 
                             pow(opr.Ay_fn(par, opr, i, j, k),2));
    }
    else{
        double *Ax = opr.dsval("Ax");
        double *Ay = opr.dsval("Ay");
        int yDim = par.ival("yDim");
        int count = i*yDim + j; 
        return 0.5 * mass * (( V_x * V_x + V_y * V_y) + 
                             pow(Ax[count],2) + 
                             pow(Ay[count],2));
    }

}

// Function for simple 3d torus trapping potential
double torus_V(Grid &par, Op &opr, int i, int j, int k){
    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double *z = par.dsval("z");

    double xMax = par.dval("xMax");

    double omegaX = par.dval("omegaX");
    double omegaY = par.dval("omegaY");
    double omegaZ = par.dval("omegaZ");
    double yOffset = par.dval("x0_shift");
    double xOffset = par.dval("y0_shift");
    double zOffset = par.dval("z0_shift");
    double mass = par.dval("mass");
    double fudge = par.dval("fudge");

    // Now we need to determine how we are representing V_xyz

    double rMax = xMax;
    double rad = sqrt((x[i] - xOffset) * (x[i] - xOffset)
                      + (y[j] - yOffset) * (y[j] - yOffset)) - 0.5*rMax*fudge;
    double omegaR = (omegaX*omegaX + omegaY*omegaY + omegaZ*omegaZ);
    double V_tot = omegaR*((z[k] - zOffset)*(z[k] - zOffset) + rad*rad);
    if (par.Afn != "file"){
        return 0.5 * mass * (( V_tot) + 
                             (pow(opr.Ax_fn(par, opr, i, j, k),2) + 
                              pow(opr.Az_fn(par, opr, i, j, k),2) + 
                              pow(opr.Ay_fn(par, opr, i, j, k),2)));
    }
    else{
        double *Ax = opr.dsval("Ax");
        double *Ay = opr.dsval("Ay");
        double *Az = opr.dsval("Az");
        int yDim = par.ival("yDim");
        int zDim = par.ival("zDim");
        int count = i*yDim*zDim + j*zDim + k; 
        return 0.5 * mass * (( V_tot) + 
                            (pow(Ax[count],2) + 
                             pow(Az[count],2) + 
                             pow(Ay[count],2)));
    }
    return V_tot;
}

// Function for simple 3d harmonic V with i and j as the iterators
double harmonic_V3d(Grid &par, Op &opr, int i , int j, int k){
    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double *z = par.dsval("z");
    double omegaX = par.dval("omegaX");
    double omegaY = par.dval("omegaY");
    double omegaZ = par.dval("omegaZ");
    double gammaY = par.dval("gammaY");
    double yOffset = 0.0;
    double xOffset = 0.0;
    double zOffset = 0.0;
    double mass = par.dval("mass");
    double V_x = omegaX*(x[i]+xOffset); 
    double V_y = gammaY*omegaY*(y[j]+yOffset);
    double V_z = gammaY*omegaZ*(z[k]+zOffset);
    if (par.Afn != "file"){
        return 0.5 * mass * (( V_x * V_x + V_y * V_y + V_z*V_z) + 
                            pow(opr.Ax_fn(par, opr, i, j, k),2) + 
                            pow(opr.Az_fn(par, opr, i, j, k),2) + 
                            pow(opr.Ay_fn(par, opr, i, j, k),2));
    }
    else{
        double *Ax = opr.dsval("Ax");
        double *Ay = opr.dsval("Ay");
        double *Az = opr.dsval("Az");
        int yDim = par.ival("yDim");
        int zDim = par.ival("zDim");
        int count = i*yDim*zDim + j*zDim + k; 
        return 0.5 * mass * (( V_x * V_x + V_y * V_y + V_z * V_z) + 
                            pow(Ax[count],2) + 
                            pow(Az[count],2) + 
                            pow(Ay[count],2));
    }

}

// Function for simple 2d harmonic V with i and j as iterators, dimensionless
double harmonic_V_dimensionless(Grid &par, Op &opr, int i , int j, int k){
    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double omegaX = par.dval("omegaX");
    double omegaY = par.dval("omegaY");
    double gammaY = par.dval("gammaY");
    double yOffset = 0.0;
    double xOffset = 0.0;
    double V_x = omegaX*(x[i]+xOffset); 
    double V_y = gammaY*omegaY*(y[j]+yOffset);
    return 0.5*( V_x * V_x + V_y * V_y) + 
           0.5 * pow(opr.Ax_fn(par, opr, i, j, k),2) + 
           0.5 * pow(opr.Ay_fn(par, opr, i, j, k),2);

}


// Function for simple 2d harmonic V with i and j as iterators, gauge
double harmonic_gauge_V(Grid &par, Op &opr, int i , int j, int k){
    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double omega = par.dval("omega");
    double omegaX = par.dval("omegaX");
    double omegaY = par.dval("omegaY");
    double gammaY = par.dval("gammaY");
    double omega_0 = omega * omegaX;
    double omega_1 = omega * omegaY;
    double yOffset = 0.0;
    double xOffset = 0.0;
    double mass = par.dval("mass");
    double ox = omegaX - omega_0;
    double oy = omegaY - omega_1;
    double v1 = ox * (x[i]+xOffset) * ox * (x[i]+xOffset) 
                + gammaY*oy*(y[j]+yOffset) * gammaY*oy*(y[j]+yOffset);
    return 0.5 * mass * (v1 );
    //return 0.5*mass*( pow(omegaX*(x[i]+xOffset) - omega_0,2) + 
    //       pow(gammaY*omegaY*(y[j]+yOffset) - omega_1,2) );

}

// Functions for pAx, y, z for rotation along the z axis
// note that pAx and pAy call upon the Ax and Ay functions
double pAx_fn(Grid &par, Op &opr, int i, int j, int k){
    double *xp = par.dsval("xp");
    if (par.Afn != "file"){
        return opr.Ax_fn(par, opr, i, j, k) * xp[i];
    }
    else{
        double *Ax = opr.dsval("Ax");
        int yDim = par.ival("yDim");
        int count = 0;
        if (par.ival("dimnum") == 2){
            count = i*yDim + j; 
        }
        else if (par.ival("dimnum") == 3){
            int zDim = par.ival("zDim");
            count = k + j*zDim + i*yDim*zDim;
        }
        return Ax[count] * xp[i];
    }
}

double pAy_fn(Grid &par, Op &opr, int i, int j, int k){
    double *yp = par.dsval("yp");
    if (par.Afn != "file"){
        return opr.Ay_fn(par, opr, i, j, k) * yp[j];
    }
    else{
        double *Ay = opr.dsval("Ay");
        int yDim = par.ival("yDim");
        int count = 0;
        if (par.ival("dimnum") == 2){
            count = i*yDim + j; 
        }
        else if (par.ival("dimnum") == 3){
            int zDim = par.ival("zDim");
            count = k + j*zDim + i*yDim*zDim;
        }
        return Ay[count] * yp[j];
    }
}

double pAz_fn(Grid &par, Op &opr, int i, int j, int k){
    double *zp = par.dsval("zp");
    if (par.Afn != "file"){
        return opr.Az_fn(par, opr, i, j, k) * zp[k];
    }
    else{
        double *Az = opr.dsval("Az");
        int yDim = par.ival("yDim");
        int count = 0;
        if (par.ival("dimnum") == 2){
            count = i*yDim + j; 
        }
        else if (par.ival("dimnum") == 3){
            int zDim = par.ival("zDim");
            count = k + j*zDim + i*yDim*zDim;
        }
        return Az[count] * zp[k];
    }
}

double rotation_Ax(Grid &par, Op &opr, int i, int j, int k){
    double *y = par.dsval("y");
    double omega = par.dval("omega");
    double omegaX = par.dval("omegaX");
    return -y[j] * omega * omegaX;
}

double rotation_Az(Grid &par, Op &opr, int i, int j, int k){
    return 0;
}

double rotation_Ay(Grid &par, Op &opr, int i, int j, int k){
    double *x = par.dsval("x");
    double omega = par.dval("omega");
    double omegaY = par.dval("omegaY");
    return x[i] * omega * omegaY;
}

// Functions for a simple vortex ring
double ring_Az(Grid &par, Op &opr, int i, int j, int k){
    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double xMax = par.dval("xMax");
    double omega = par.dval("omega");

    double rad = sqrt(x[i]*x[i] + y[j]*y[j]);

    return omega * exp(-rad*rad / (0.0001*xMax)) * 0.01;
}

// Function to return 0, this is for constant gauge field tests.
double constant_A(Grid &par, Op &opr, int i, int j, int k){
    return 0;
}

// Fuinctions for Ax, y, z for rotation along the z axis
double test_Ax(Grid &par, Op &opr, int i, int j, int k){
    double *y = par.dsval("y");
    //double *x = par.dsval("x");
    double omega = par.dval("omega");
    double omegaX = par.dval("omegaX");
    double yMax = par.dval("yMax");
    //double val = -y[j]*y[j];
    double val = (sin(y[j] * 50000)+1) * omegaX * yMax * omega;
    return val;
}

double test_Ay(Grid &par, Op &opr, int i, int j, int k){
    //double *x = par.dsval("x");
    //double omega = par.dval("omega");
    //double omegaY = par.dval("omegaY");
    //double xMax = par.dval("xMax");
    //double val = x[i]*x[i];
    double val = 0;
    return val;
}

// Functions for fiber -- BETA
// Note: because of the fiber axis we are working with here, we will be using
//       E_z and E_r and ignoring E_phi
//       E_r -> E_x
//       E_z -> E_y

// This is a Function to return Az, because there is no A_r or A_phi
double fiber2d_Ax(Grid &par, Op &opr, int i, int j, int k){
    double val = 0;
    val = HBAR ; // Plus everything else. How to implement detuning?

    return val;
}

double fiber2d_Ay(Grid &par, Op &opr, int i, int j, int k){
    double val = 0;

    return val;
}

/*
// Functions to determine Electric field at a provided point
// Note that we need to multiply this by the dipole moment, (d)^2
double LP01_E_squared(Grid &par, Op &opr, int i, int j, int k){
    double val = 0;

    double r = par.dsval("x")[i];

    std::unordered_map<std::string, double>
        matlab_map = read_matlab_data(14);

    double beta1 = matlab_map["beta1"];
    double q = matlab_map["q"];
    double h = matlab_map["h"];
    double a = matlab_map["a"];
    double n1 = matlab_map["n1"];
    double n2 = matlab_map["n2"];
    double spar = matlab_map["spar"];
    double N1 = (beta1*beta1/(4*h*h))
                *(pow((1-spar),2)*(pow(jn(0,h*a),2)+pow(jn(1,h*a),2))
                  +pow(1+spar,2)
                   *(pow(jn(2,h*a),2)-jn(1,h*a)*jn(3,h*a)))
                +((0.5)*(((pow(jn(1,h*a),2))-(jn(0,h*a)*jn(2,h*a)))));


    double N2=(0.5)*(jn(1,h*a)/pow(boost::math::cyl_bessel_k(1,q*a),2))
               *(((beta1*beta1/(2*q*q))
               *(pow(1-spar,2)*(pow(boost::math::cyl_bessel_k(1,q*a),2)
                                -pow(boost::math::cyl_bessel_k(0,q*a),2))
               -pow(1+spar,2)*(pow(boost::math::cyl_bessel_k(2,q*a),2)
               -boost::math::cyl_bessel_k(1,q*a)
               *boost::math::cyl_bessel_k(3,q*a))))
               -pow(boost::math::cyl_bessel_k(1,q*a),2) 
               +boost::math::cyl_bessel_k(0,q*a)
                *boost::math::cyl_bessel_k(2,q*a));
    

    double AA = (beta1 / (2 * q)) * 
                (jn(1,h*a)/boost::math::cyl_bessel_k(1,q*a))
                / (2 * M_PI * a * a * (n1*n1*N1 + n2*n2*N2));

    val = 2 * AA * AA *((1-spar)*(1-spar)
                        *pow(boost::math::cyl_bessel_k(0,q*r),2)
                        + (1+spar)*(1+spar)
                          *pow(boost::math::cyl_bessel_k(2,q*r),2)
                        + (2*q*q / (beta1*beta1))
                          *pow(boost::math::cyl_bessel_k(1,q*r),2));

    return val;
}
*/

// Now we need a function to read in the data from matlab
// Note that this has already been parsed into a bunch of different files
//     in data/data... This may need to be changed...
// Ideally, we would fix the parser so that it takes the fiber option into 
//     account and reads in the appropriate index. 
// For now (due to lack of dev time), we will simply read in ii = 14.
// BETA
std::unordered_map<std::string, double> read_matlab_data(int index){

    std::cout << "doing stuff" << '\n';

    // Note that we need a std::unordered_map for all the variables
    std::unordered_map<std::string, double> matlab_variables;

    // Now we need to read in the file 
    std::string filename = "data/data" + std::to_string(index) + ".dat";
    std::ifstream fileID(filename);
    std::string item1, item2;

    std::string line;
    while (fileID >> line){
        item1 = line.substr(0,line.find(","));
        item2 = line.substr(line.find(",")+1,line.size());
        matlab_variables[item1] = std::stod(item2);
    }

    return matlab_variables;
}

// Function to read Ax from file.
// Note that this comes with a special method in init...
void file_A(std::string filename, double *A, double omega){
    std::fstream infile(filename, std::ios_base::in);

    double inval;
    int count = 0;
    while (infile >> inval){
        A[count] = omega*inval;
        count++;
    }
}

// Function to check whether a file exists
std::string filecheck(std::string filename){

    struct stat buffer = {0};
    if (stat(filename.c_str(), &buffer) == -1){
        std::cout << "File " << filename << " does not exist!" << '\n';
        std::cout << "Please select a new file:" << '\n'; 
        std::cin >> filename; 
        filename = filecheck(filename);
    } 

    return filename;
}

// Function to compute the wfc for the standard 2d case
cufftDoubleComplex standard_wfc_2d(Grid &par, double Phi,
                                   int i, int j, int k){
    cufftDoubleComplex wfc;
    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double Rxy = par.dval("Rxy");
    double a0y = par.dval("a0y");
    double a0x = par.dval("a0x");
    wfc.x = exp(-( pow((x[i])/(Rxy*a0x),2) + 
                   pow((y[j])/(Rxy*a0y),2) ) ) * cos(Phi);
    wfc.y = -exp(-( pow((x[i])/(Rxy*a0x),2) + 
                    pow((y[j])/(Rxy*a0y),2) ) ) * sin(Phi);

    return wfc;
}

// Function to compute the initial wavefunction for the standard 3d caase
cufftDoubleComplex standard_wfc_3d(Grid &par, double Phi,
                                   int i, int j, int k){
    cufftDoubleComplex wfc;
    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double *z = par.dsval("z");
    double Rxy = par.dval("Rxy");
    double a0y = par.dval("a0y");
    double a0x = par.dval("a0x");
    double a0z = par.dval("a0z");
    wfc.x = exp(-( pow((x[i])/(Rxy*a0x),2) + 
                   pow((y[j])/(Rxy*a0y),2) +
                   pow((z[k])/(Rxy*a0z),2))) *
                   cos(Phi);
    wfc.y = -exp(-( pow((x[i])/(Rxy*a0x),2) + 
                    pow((y[j])/(Rxy*a0y),2) +
                    pow((z[k])/(Rxy*a0z),2))) *
                    sin(Phi);

    return wfc;
}

// Function to initialize a toroidal wfc
// note that we will need to specify the size of everything based on fiber
// size and such.
cufftDoubleComplex torus_wfc(Grid &par, double Phi,
                             int i, int j, int k){

    cufftDoubleComplex wfc;

    // Let's read in all the necessary parameters
    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double *z = par.dsval("z");
    double xOffset = par.dval("x0_shift");
    double yOffset = par.dval("y0_shift");
    double fudge = par.dval("fudge");

    double xMax = par.dval("xMax");

    double Rxy = par.dval("Rxy");

    double a0x = par.dval("a0x");
    //double a0y = par.dval("a0y");
    double a0z = par.dval("a0z");

    //double rMax = sqrt(xMax*xMax + yMax*yMax);
    double rMax = xMax;

    // We will now create a 2d projection and extend it in a torus shape
    double rad = sqrt((x[i] - xOffset) * (x[i] - xOffset) 
                      + (y[j] - yOffset) * (y[j] - yOffset)) - 0.5*rMax*fudge;
    //double a0r = sqrt(a0x*a0x + a0y*a0y);

    wfc.x = exp(-( pow((rad)/(Rxy*a0x*0.5),2) + 
                   pow((z[k])/(Rxy*a0z*0.5),2) ) ) * cos(Phi);
    wfc.y = -exp(-( pow((rad)/(Rxy*a0x*0.5),2) + 
                    pow((z[k])/(Rxy*a0z*0.5),2) ) ) * sin(Phi);

    return wfc;

}

/*----------------------------------------------------------------------------//
* GPU KERNELS
*-----------------------------------------------------------------------------*/

// Function to generate momentum grids
void generate_p_space(Grid &par){

    int dimnum = par.ival("dimnum");
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int zDim = par.ival("zDim");
    double xMax = par.dval("xMax");
    double yMax = par.dval("yMax");
    double zMax = 0;
    if (dimnum == 3){
        zMax = par.dval("zMax");
    }
    double pxMax = par.dval("pxMax");
    double pyMax = par.dval("pyMax");
    double pzMax = 0;
    if (dimnum == 3){
        pzMax = par.dval("pzMax");
    }
    double dx = par.dval("dx");
    double dy = par.dval("dy");
    double dz = 0;
    if (dimnum == 3){
        dz = par.dval("dz");
    }
    double dpx = par.dval("dpx");
    double dpy = par.dval("dpy");
    double dpz = 0;
    if (dimnum == 3){
        dpz = par.dval("dpz");
    }

    double *x, *y, *z, *px, *py, *pz,
           *x_gpu, *y_gpu, *z_gpu, 
           *px_gpu, *py_gpu, *pz_gpu;

    x = (double *) malloc(sizeof(double) * xDim);
    y = (double *) malloc(sizeof(double) * yDim);
    z = (double *) malloc(sizeof(double) * zDim);
    px = (double *) malloc(sizeof(double) * xDim);
    py = (double *) malloc(sizeof(double) * yDim);
    pz = (double *) malloc(sizeof(double) * zDim);

    if (dimnum == 2){

        for(int i=0; i<xDim/2; ++i){
            x[i] = -xMax + i*dx;
            x[i + (xDim/2)] = i*dx;

            px[i] = i*dpx;
            px[i + (xDim/2)] = -pxMax + i*dpx;

        }
        for(int i=0; i<yDim/2; ++i){
            y[i] = -yMax + i*dy;
            y[i + (yDim/2)] = i*dy;

            py[i] = i*dpy;
            py[i + (yDim/2)] = -pyMax + i*dpy;

        }

        for(int i = 0; i < zDim; ++i){
            z[i] = 0;
            pz[i] = 0;
        }

    }
    else if(dimnum == 3){
        for(int i=0; i<xDim/2; ++i){
            x[i] = -xMax + i*dx;
            x[i + (xDim/2)] = i*dx;

            px[i] = i*dpx;
            px[i + (xDim/2)] = -pxMax + i*dpx;

        }
        for(int i=0; i<yDim/2; ++i){
            y[i] = -yMax + i*dy;
            y[i + (yDim/2)] = i*dy;

            py[i] = i*dpy;
            py[i + (yDim/2)] = -pyMax + i*dpy;

        }
        for(int i=0; i<zDim/2; ++i){
            z[i] = -zMax + i*dz;
            z[i + (zDim/2)] = i*dz;

            pz[i] = i*dpz;
            pz[i + (zDim/2)] = -pzMax + i*dpz;

        }

    }
    par.store("x",x);
    par.store("y",y);
    par.store("z",z);
    par.store("px",px);
    par.store("py",py);
    par.store("pz",pz);

    // Now move these items to the gpu
    cudaMalloc((void**) &x_gpu, sizeof(double) * xDim);
    cudaMalloc((void**) &y_gpu, sizeof(double) * yDim);
    cudaMalloc((void**) &z_gpu, sizeof(double) * zDim);
    cudaMalloc((void**) &px_gpu, sizeof(double) * xDim);
    cudaMalloc((void**) &py_gpu, sizeof(double) * yDim);
    cudaMalloc((void**) &pz_gpu, sizeof(double) * zDim);

    cudaMemcpy(x_gpu, x, sizeof(double)*xDim, cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, y, sizeof(double)*yDim, cudaMemcpyHostToDevice);
    cudaMemcpy(z_gpu, z, sizeof(double)*zDim, cudaMemcpyHostToDevice);
    cudaMemcpy(px_gpu, px, sizeof(double)*xDim, cudaMemcpyHostToDevice);
    cudaMemcpy(py_gpu, py, sizeof(double)*yDim, cudaMemcpyHostToDevice);
    cudaMemcpy(pz_gpu, pz, sizeof(double)*zDim, cudaMemcpyHostToDevice);

    par.store("x_gpu",x_gpu);
    par.store("y_gpu",y_gpu);
    par.store("z_gpu",z_gpu);
    par.store("px_gpu",px_gpu);
    par.store("py_gpu",py_gpu);
    par.store("pz_gpu",pz_gpu);
}

// This function is basically a wrapper to call the appropriate K kernel
void generate_K(Grid &par){

    // For k, we need xp, yp, and zp. These will also be used in generating 
    // pAxyz parameters, so it should already be stored in par.
    double *px_gpu = par.dsval("px_gpu");
    double *py_gpu = par.dsval("py_gpu");
    double *pz_gpu = par.dsval("pz_gpu");
    double gSize = par.ival("gSize");
    double mass = par.dval("mass");

    // Creating K to work with
    double *K, *K_gpu;
    K = (double*)malloc(sizeof(double)*gSize);
    cudaMalloc((void**) &K_gpu, sizeof(double)*gSize);

    simple_K<<<par.grid, par.threads>>>(px_gpu, py_gpu, pz_gpu, mass, K_gpu);

    cudaMemcpy(K, K_gpu, sizeof(double)*gSize, cudaMemcpyDeviceToHost);
    par.store("K",K);
    par.store("K_gpu",K_gpu);
    
}

// Simple kernel for generating K
__global__ void simple_K(double *xp, double *yp, double *zp, double mass,
                         double *K){

    unsigned int gid = getGid3d3d();
    unsigned int xid = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int yid = blockDim.y*blockIdx.y + threadIdx.y;
    unsigned int zid = blockDim.z*blockIdx.z + threadIdx.z;
    K[gid] = (HBAR*HBAR/(2*mass))*(xp[xid]*xp[xid] + yp[yid]*yp[yid]
                                  + zp[zid]*zp[zid]);
}

// Function to generate game fields
void generate_gauge(Grid &par){

    int gSize = par.ival("gSize");
    int dimnum = par.ival("dimnum");

    double *Ax, *Ay, *Az, *Ax_gpu, *Ay_gpu, *Az_gpu;
    double *x_gpu = par.dsval("x_gpu");
    double *y_gpu = par.dsval("y_gpu");
    double *z_gpu;
    if (dimnum == 3){
        double *z_gpu = par.dsval("z_gpu");
    }

    double xMax = par.dval("xMax");
    double yMax = par.dval("yMax");
    double zMax;
    if (dimnum == 3){
        double zMax = par.dval("zMax");
    }
    double omega = par.dval("omega");
    double fudge = par.dval("fudge");

    Ax = (double *)malloc(sizeof(double)*gSize);
    Ay = (double *)malloc(sizeof(double)*gSize);
    Az = (double *)malloc(sizeof(double)*gSize);

    cudaMalloc((void**) &Ax_gpu, sizeof(double)*gSize);
    cudaMalloc((void**) &Ay_gpu, sizeof(double)*gSize);
    cudaMalloc((void**) &Az_gpu, sizeof(double)*gSize);

    if (par.Afn == "file"){
        file_A(par.Axfile, Ax, omega);
        cudaMemcpy(Ax_gpu, Ax, sizeof(double)*gSize, cudaMemcpyHostToDevice);

        file_A(par.Ayfile, Ay, omega);
        cudaMemcpy(Ay_gpu, Ay, sizeof(double)*gSize, cudaMemcpyHostToDevice);

        if (dimnum == 3){
            file_A(par.Azfile, Az, omega);
            cudaMemcpy(Az_gpu,Az,sizeof(double)*gSize,cudaMemcpyHostToDevice);
        }

        std::cout << "finished reading Ax / Ay from file" << '\n';
    }
    else{
        par.Ax_fn<<<par.grid, par.threads>>>(x_gpu, y_gpu, z_gpu, 
                                             xMax, yMax, zMax, 
                                             omega, fudge, Ax_gpu);
        par.Ay_fn<<<par.grid, par.threads>>>(x_gpu, y_gpu, z_gpu, 
                                             xMax, yMax, zMax, 
                                             omega, fudge, Ay_gpu);
        if (dimnum == 3){
            par.Az_fn<<<par.grid, par.threads>>>(x_gpu, y_gpu, z_gpu, 
                                                 xMax, yMax, zMax, 
                                                 omega, fudge, Az_gpu);
        }
        else{
            kconstant_A<<<par.grid, par.threads>>>(x_gpu, y_gpu, z_gpu, 
                                                   xMax, yMax, zMax, 
                                                   omega, fudge, Az_gpu);
        }
    }
    cudaMemcpy(Ax, Ax_gpu, sizeof(double)*gSize,cudaMemcpyDeviceToHost);
    cudaMemcpy(Ay, Ay_gpu, sizeof(double)*gSize,cudaMemcpyDeviceToHost);
    cudaMemcpy(Az, Az_gpu, sizeof(double)*gSize,cudaMemcpyDeviceToHost);

    par.store("Ax", Ax);
    par.store("Ay", Ay);
    par.store("Az", Az);

    par.store("Ax_gpu", Ax_gpu);
    par.store("Ay_gpu", Ay_gpu);
    par.store("Az_gpu", Az_gpu);

}

// constant Kernel A
__global__ void kconstant_A(double *x, double *y, double *z,
                            double xMax, double yMax, double zMax,
                            double omega, double fudge, double *A){
    int gid = getGid3d3d();
    A[gid] = 0;        
}

// Kernel for simple rotational case, Ax
__global__ void krotation_Ax(double *x, double *y, double *z,
                             double xMax, double yMax, double zMax,
                             double omega, double fudge, double *A){
    int gid = getGid3d3d();
    int yid = blockDim.y*blockIdx.y + threadIdx.y;
    A[gid] = -y[yid] * omega;
}

// Kernel for simple rotational case, Ay
__global__ void krotation_Ay(double *x, double *y, double *z,
                             double xMax, double yMax, double zMax,
                             double omega, double fudge, double *A){
    int gid = getGid3d3d();
    int xid = blockDim.x*blockIdx.x + threadIdx.x;
    A[gid] = x[xid] * omega;
}

// kernel for a simple vortex ring
__global__ void kring_Az(double *x, double *y, double *z,
                         double xMax, double yMax, double zMax,
                         double omega, double fudge, double *A){
    int gid = getGid3d3d();
    int xid = blockDim.x*blockIdx.x + threadIdx.x;
    int yid = blockDim.y*blockIdx.y + threadIdx.y;

    double rad = sqrt(x[xid]*x[xid] + y[yid]*y[yid]);

    A[gid] = omega * exp(-rad*rad / (0.0001*xMax)) * 0.01;
}

// testing kernel Ax
__global__ void ktest_Ax(double *x, double *y, double *z,
                         double xMax, double yMax, double zMax,
                         double omega, double fudge, double *A){
    int gid = getGid3d3d();
    int yid = blockDim.y*blockIdx.x + threadIdx.x;
    A[gid] = (sin(y[yid] * 50000)+1) * yMax * omega;
}

// testing kernel Ay
__global__ void ktest_Ay(double *x, double *y, double *z,
                         double xMax, double yMax, double zMax,
                         double omega, double fudge, double *A){
    int gid = getGid3d3d();
    A[gid] = 0;
}

// function to generate V
void generate_fields(Grid &par){

    generate_p_space(par);
    generate_K(par);
    generate_gauge(par);

    int gSize = par.ival("gSize");
    int dimnum = par.ival("dimnum");
    int winding = par.dval("winding");

    double dt = par.dval("dt");
    double gdt = par.dval("gdt");
    double *x_gpu = par.dsval("x_gpu");
    double *y_gpu = par.dsval("y_gpu");
    double *z_gpu = par.dsval("z_gpu");
    double *px_gpu = par.dsval("px_gpu");
    double *py_gpu = par.dsval("py_gpu");
    double *pz_gpu = par.dsval("pz_gpu");
    double *Ax_gpu = par.dsval("Ax_gpu");
    double *Ay_gpu = par.dsval("Ay_gpu");
    double *Az_gpu = par.dsval("Az_gpu");
    double *K_gpu = par.dsval("K_gpu");

    // Creating items list for kernels

    double *items, *items_gpu;
    int item_size = 18;
    items = (double*)malloc(sizeof(double)*item_size);
    cudaMalloc((void**) &items_gpu, sizeof(double)*item_size);

    for (int i = 0; i < item_size; ++i){
        items[0] = 0;
    }
    items[0] = par.dval("xMax");
    items[1] = par.dval("yMax");
    if (dimnum == 3){
        items[2] = par.dval("zMax");
    }

    items[3] = par.dval("omegaX");
    items[4] = par.dval("omegaY");
    if (dimnum == 3){
        items[5] = par.dval("omegaZ");
    }

    items[6] = par.dval("x0_shift");
    items[7] = par.dval("y0_shift");
    if (dimnum == 3){
        items[8] = par.dval("z0_shift");
    }
    else{
        items[8] = 0.0;
    }

    items[9] = par.dval("mass");
    items[10] = par.dval("gammaY");
    items[11] = 1.0; // For gammaZ
    items[12] = par.dval("fudge");
    items[13] = 0.0; // For time

    items[14] = par.dval("Rxy");

    items[15] = par.dval("a0x");
    items[16] = par.dval("a0y");
    if (dimnum == 3){
        items[17] = par.dval("a0z");
    }
    else{
        items[17] = 1.0;
    }

    cudaMemcpy(items_gpu, items, sizeof(double)*item_size,
               cudaMemcpyHostToDevice);

    double fudge = par.dval("fudge");

    // Generating V

    double *V, *V_gpu;

    V = (double *)malloc(sizeof(double)*gSize);

    cudaMalloc((void **) &V_gpu, sizeof(double)*gSize);

    par.V_fn<<<par.grid, par.threads>>>(x_gpu, y_gpu, z_gpu, items_gpu,
                                        Ax_gpu, Ay_gpu, Az_gpu, V_gpu);

    cudaMemcpy(V, V_gpu, sizeof(double)*gSize, cudaMemcpyDeviceToHost);

    // Generating wfc

    double2 *wfc, *wfc_gpu;
    double *phi, *phi_gpu;

    wfc = (double2 *)malloc(sizeof(double2)*gSize);
    phi = (double *)malloc(sizeof(double)*gSize);

    cudaMalloc((void**) &wfc_gpu, sizeof(double2)*gSize);
    cudaMalloc((void**) &phi_gpu, sizeof(double)*gSize);

    if (par.Wfcfn == "file"){
        wfc = par.cufftDoubleComplexval("wfc");
        cudaMemcpy(wfc_gpu, wfc, sizeof(double2)*gSize, cudaMemcpyHostToDevice);
    }
    else{
        par.wfc_fn<<<par.grid, par.threads>>>(x_gpu, y_gpu, z_gpu, items_gpu,
                                              winding, phi_gpu, wfc_gpu);
        cudaMemcpy(wfc, wfc_gpu, sizeof(double2)*gSize, cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(phi, phi_gpu, sizeof(double)*gSize, cudaMemcpyDeviceToHost);

    // generating aux fields.
    double2 *GV, *EV, *GK, *EK;
    double2 *GV_gpu, *EV_gpu, *GK_gpu, *EK_gpu;
    double2 *GpAx, *GpAy, *GpAz, *EpAx, *EpAy, *EpAz;
    double2 *GpAx_gpu, *GpAy_gpu, *GpAz_gpu, *EpAx_gpu, *EpAy_gpu, *EpAz_gpu;
    double *pAx, *pAy, *pAz;
    double *pAx_gpu, *pAy_gpu, *pAz_gpu;

    GV = (double2 *)malloc(sizeof(double2)*gSize);
    EV = (double2 *)malloc(sizeof(double2)*gSize);
    GK = (double2 *)malloc(sizeof(double2)*gSize);
    EK = (double2 *)malloc(sizeof(double2)*gSize);

    GpAx = (double2 *)malloc(sizeof(double2)*gSize);
    EpAx = (double2 *)malloc(sizeof(double2)*gSize);
    GpAy = (double2 *)malloc(sizeof(double2)*gSize);
    EpAy = (double2 *)malloc(sizeof(double2)*gSize);
    GpAz = (double2 *)malloc(sizeof(double2)*gSize);
    EpAz = (double2 *)malloc(sizeof(double2)*gSize);

    pAx = (double *)malloc(sizeof(double)*gSize);
    pAy = (double *)malloc(sizeof(double)*gSize);
    pAz = (double *)malloc(sizeof(double)*gSize);

    cudaMalloc((void**) &GV_gpu, sizeof(double2)*gSize);
    cudaMalloc((void**) &EV_gpu, sizeof(double2)*gSize);
    cudaMalloc((void**) &GK_gpu, sizeof(double2)*gSize);
    cudaMalloc((void**) &EK_gpu, sizeof(double2)*gSize);

    cudaMalloc((void**) &GpAx_gpu, sizeof(double2)*gSize);
    cudaMalloc((void**) &EpAx_gpu, sizeof(double2)*gSize);
    cudaMalloc((void**) &GpAy_gpu, sizeof(double2)*gSize);
    cudaMalloc((void**) &EpAy_gpu, sizeof(double2)*gSize);
    cudaMalloc((void**) &GpAz_gpu, sizeof(double2)*gSize);
    cudaMalloc((void**) &EpAz_gpu, sizeof(double2)*gSize);

    cudaMalloc((void**) &pAx_gpu, sizeof(double)*gSize);
    cudaMalloc((void**) &pAy_gpu, sizeof(double)*gSize);
    cudaMalloc((void**) &pAz_gpu, sizeof(double)*gSize);

    aux_fields<<<par.grid, par.threads>>>(V_gpu, K_gpu, gdt, dt,
                                          Ax_gpu, Ay_gpu, Az_gpu,
                                          px_gpu, py_gpu, pz_gpu,
                                          pAx_gpu, pAy_gpu, pAz_gpu,
                                          GV_gpu, EV_gpu, GK_gpu, EK_gpu,
                                          GpAx_gpu, GpAy_gpu, GpAz_gpu,
                                          EpAx_gpu, EpAy_gpu, EpAz_gpu);

    cudaMemcpy(GV, GV_gpu, sizeof(double2)*gSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(EV, EV_gpu, sizeof(double2)*gSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(GK, GK_gpu, sizeof(double2)*gSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(EK, EK_gpu, sizeof(double2)*gSize, cudaMemcpyDeviceToHost);

    cudaMemcpy(GpAx, GpAx_gpu, sizeof(double2)*gSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(EpAx, EpAx_gpu, sizeof(double2)*gSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(GpAy, GpAy_gpu, sizeof(double2)*gSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(EpAy, EpAy_gpu, sizeof(double2)*gSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(GpAz, GpAz_gpu, sizeof(double2)*gSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(EpAz, EpAz_gpu, sizeof(double2)*gSize, cudaMemcpyDeviceToHost);

    cudaMemcpy(pAx, pAx_gpu, sizeof(double)*gSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(pAy, pAy_gpu, sizeof(double)*gSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(pAz, pAz_gpu, sizeof(double)*gSize, cudaMemcpyDeviceToHost);

    // Storing variables

    par.store("V",V);
    par.store("V_gpu",V_gpu);
    par.store("items", items);
    par.store("items_gpu", items_gpu);
    par.store("wfc", wfc);
    par.store("wfc_gpu", wfc_gpu);
    par.store("Phi", phi);
    par.store("Phi_gpu", phi_gpu);

    par.store("GV",GV);
    par.store("EV",EV);
    par.store("GK",GK);
    par.store("EK",EK);
    par.store("GV_gpu",GV_gpu);
    par.store("EV_gpu",EV_gpu);
    par.store("GK_gpu",GK_gpu);
    par.store("EK_gpu",EK_gpu);

    par.store("GpAx",GpAx);
    par.store("EpAx",EpAx);
    par.store("GpAy",GpAy);
    par.store("EpAy",EpAy);
    par.store("GpAz",GpAz);
    par.store("EpAz",EpAz);

    par.store("pAx",pAx);
    par.store("pAy",pAy);
    par.store("pAz",pAz);
    par.store("pAx_gpu",pAx_gpu);
    par.store("pAy_gpu",pAy_gpu);
    par.store("pAz_gpu",pAz_gpu);
    
}

__global__ void kharmonic_V(double *x, double *y, double *z, double* items,
                            double *Ax, double *Ay, double *Az, double *V){

    int gid = getGid3d3d();
    int xid = blockDim.x*blockIdx.x + threadIdx.x;
    int yid = blockDim.y*blockIdx.y + threadIdx.y;
    int zid = blockDim.z*blockIdx.z + threadIdx.z;

    double xOffset = items[6];
    double V_x = items[3]*(x[xid]+items[6]);
    double V_y = items[10]*items[4]*(y[yid]+items[7]);
    double V_z = items[11]*items[5]*(z[zid]+items[8]);

    V[gid] = 0.5*items[9]*((V_x*V_x + V_y*V_y + V_z*V_z)
             + Ax[gid]*Ax[gid] + Ay[gid]*Ay[gid] + Az[gid]*Az[gid]);
}

// kernel for simple 3d torus trapping potential
__global__ void ktorus_V(double *x, double *y, double *z, double* items,
                         double *Ax, double *Ay, double *Az, double *V){

    int gid = getGid3d3d();
    int xid = blockDim.x*blockIdx.x + threadIdx.x;
    int yid = blockDim.y*blockIdx.y + threadIdx.y;
    int zid = blockDim.z*blockIdx.z + threadIdx.z;

    double rad = sqrt((x[xid] - items[6]) * (x[xid] - items[6])
                      + (y[yid] - items[7]) * (y[yid] - items[7])) 
                      - 0.5*items[0]*items[12];
    double omegaR = (items[3]*items[3] + items[4]*items[4] + items[5]*items[5]);
    double V_tot = omegaR*((z[zid] - items[8])*(z[zid] - items[8]) + rad*rad);
    V[gid] = 0.5*items[9]*(V_tot
                           + Ax[gid]*Ax[gid]
                           + Ay[gid]*Ay[gid]
                           + Az[gid]*Az[gid]);
}

__global__ void kstd_wfc(double *x, double *y, double *z, double *items,
                         double winding, double *phi, double2 *wfc){

    int gid = getGid3d3d();
    int xid = blockDim.x*blockIdx.x + threadIdx.x;
    int yid = blockDim.y*blockIdx.y + threadIdx.y;
    int zid = blockDim.z*blockIdx.z + threadIdx.z;

    phi[gid] = fmod(winding*atan2(y[yid], x[xid]),2*PI);

    wfc[gid].x = exp(-(x[xid]*x[xid]/(items[14]*items[14]*items[15]*items[15]) 
                     + y[yid]*y[yid]/(items[14]*items[14]*items[16]*items[16]) 
                     + z[zid]*z[zid]/(items[14]*items[14]*items[17]*items[17])))
                     * cos(phi[gid]);
    wfc[gid].y = -exp(-(x[xid]*x[xid]/(items[14]*items[14]*items[15]*items[15]) 
                     + y[yid]*y[yid]/(items[14]*items[14]*items[16]*items[16]) 
                     + z[zid]*z[zid]/(items[14]*items[14]*items[17]*items[17])))
                     * sin(phi[gid]);

}

__global__ void ktorus_wfc(double *x, double *y, double *z, double *items,
                           double winding, double *phi, double2 *wfc){

    int gid = getGid3d3d();
    int xid = blockDim.x*blockIdx.x + threadIdx.x;
    int yid = blockDim.y*blockIdx.y + threadIdx.y;
    int zid = blockDim.z*blockIdx.z + threadIdx.z;

    double rad = sqrt((x[xid] - items[6]) * (x[xid] - items[6])
                      + (y[yid] - items[7]) * (y[yid] - items[7])) 
                      - 0.5*items[0]*items[12];

    wfc[gid].x = exp(-( pow((rad)/(items[14]*items[15]*0.5),2) +
                   pow((z[zid])/(items[14]*items[17]*0.5),2) ) );
    wfc[gid].y = 0.0;
}

__global__ void aux_fields(double *V, double *K, double gdt, double dt,
                           double* Ax, double *Ay, double* Az, 
                           double *px, double *py, double *pz,
                           double* pAx, double* pAy, double* pAz,
                           double2* GV, double2* EV, double2* GK, double2* EK,
                           double2* GpAx, double2* GpAy, double2* GpAz,
                           double2* EpAx, double2* EpAy, double2* EpAz){
    int gid = getGid3d3d();
    int xid = blockDim.x*blockIdx.x + threadIdx.x;
    int yid = blockDim.y*blockIdx.y + threadIdx.y;
    int zid = blockDim.z*blockIdx.z + threadIdx.z;

    GV[gid].x = exp( -V[gid]*(gdt/(2*HBAR)));
    GK[gid].x = exp( -K[gid]*(gdt/HBAR));
    GV[gid].y = 0.0;
    GK[gid].y = 0.0;

    // Ax and Ay will be calculated here but are used only for
    // debugging. They may be needed later for magnetic field calc

    pAy[gid] = Ax[gid] * px[xid];
    pAx[gid] = Ay[gid] * py[yid];
    pAz[gid] = Az[gid] * pz[zid];

    GpAx[gid].x = exp(-pAx[gid]*gdt);
    GpAx[gid].y = 0;
    GpAy[gid].x = exp(-pAy[gid]*gdt);
    GpAy[gid].y = 0;
    GpAz[gid].x = exp(-pAz[gid]*gdt);
    GpAz[gid].y = 0;

    EV[gid].x=cos( -V[gid]*(dt/(2*HBAR)));
    EV[gid].y=sin( -V[gid]*(dt/(2*HBAR)));
    EK[gid].x=cos( -K[gid]*(dt/HBAR));
    EK[gid].y=sin( -K[gid]*(dt/HBAR));

    EpAy[gid].x=cos(-pAy[gid]*dt);
    EpAy[gid].y=sin(-pAy[gid]*dt);
    EpAx[gid].x=cos(-pAx[gid]*dt);
    EpAx[gid].y=sin(-pAx[gid]*dt);
}

// Function to generate grids and treads for 2d and 3d cases
void generate_grid(Grid& par){

    int max_threads = 128;
    int dimnum = par.ival("dimnum");
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int zDim = par.ival("zDim");
    int xD = 1, yD = 1, zD = 1;

    if (dimnum == 2){
        if (xDim <= max_threads){
            par.threads.x = xDim;
            par.threads.y = 1;
            par.threads.z = 1;
    
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
    
            par.threads.x = dim_tmp;
            par.threads.y = 1;
            par.threads.z = 1;
            xD = pow(2,count);
            yD = yDim;
            zD = 1;
        }

    }
    else if (dimnum == 3){

        if (xDim <= max_threads){
            par.threads.x = xDim;
            par.threads.y = 1;
            par.threads.z = 1;
    
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
    
            par.threads.x = dim_tmp;
            par.threads.y = 1;
            par.threads.z = 1;
            xD = pow(2,count);
            yD = yDim;
            zD = zDim;
        }
    
    }
    par.grid.x=xD;
    par.grid.y=yD;
    par.grid.z=zD;

    std::cout << "threads in x are: " << par.threads.x << '\n';
    std::cout << "dimensions are: " << par.grid.x << '\t' 
                                    << par.grid.y << '\t' 
                                    << par.grid.z << '\n';

}
