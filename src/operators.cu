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
    if (par.ival("dimun") == 3){
        zMax = par.dval("zMax");
    }
    double pxMax = par.dval("pxMax");
    double pyMax = par.dval("pyMax");
    double pzMax = 0;
    if (par.ival("dimun") == 3){
        pzMax = par.dval("pzMax");
    }
    double dx = par.dval("dx");
    double dy = par.dval("dy");
    double dz = 0;
    if (par.ival("dimun") == 3){
        pzMax = par.dval("dz");
    }
    double dpx = par.dval("dpx");
    double dpy = par.dval("dpy");
    double dpz = 0;
    if (par.ival("dimun") == 3){
        pzMax = par.dval("dpz");
    }

    double *x, *y, *z, *px, *py, *pz,
           *x_gpu, *y_gpu, *z_gpu, 
           *px_gpu, *py_gpu, *pz_gpu;

    if (dimnum == 2){
        x = (double *) malloc(sizeof(double) * xDim);
        y = (double *) malloc(sizeof(double) * yDim);
        z = (double *) malloc(sizeof(double) * zDim);
        px = (double *) malloc(sizeof(double) * xDim);
        py = (double *) malloc(sizeof(double) * yDim);
        pz = (double *) malloc(sizeof(double) * zDim);

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

        z[0] = 0;
        pz[0] = 0;

    }
    else if(dimnum == 3){
        x = (double *) malloc(sizeof(double) * xDim);
        y = (double *) malloc(sizeof(double) * yDim);
        z = (double *) malloc(sizeof(double) * zDim);
        px = (double *) malloc(sizeof(double) * xDim);
        py = (double *) malloc(sizeof(double) * yDim);
        pz = (double *) malloc(sizeof(double) * zDim);
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
    if (dimnum == 3){
        Az = (double *)malloc(sizeof(double)*gSize);
    }

    cudaMalloc((void**) &Ax_gpu, sizeof(double)*gSize);
    cudaMalloc((void**) &Ay_gpu, sizeof(double)*gSize);
    if (dimnum == 3){
        cudaMalloc((void**) &Az_gpu, sizeof(double)*gSize);
    }

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
    }
    cudaMemcpy(Ax, Ax_gpu, sizeof(double)*gSize,cudaMemcpyDeviceToHost);
    cudaMemcpy(Ay, Ay_gpu, sizeof(double)*gSize,cudaMemcpyDeviceToHost);
    par.store("Ax", Ax);
    par.store("Ay", Ay);
    par.store("Ax_gpu", Ax_gpu);
    par.store("Ay_gpu", Ay_gpu);

    if(dimnum == 3){
        cudaMemcpy(Az, Az_gpu, sizeof(double)*gSize,cudaMemcpyDeviceToHost);
        par.store("Az", Az);
        par.store("Az_gpu", Az_gpu);
    }
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
