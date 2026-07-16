#include <iostream>
using namespace std;
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <thread>
#include <string.h>
// 2 inclusion below are for matrix inversions
#include <array> 
#include <Eigen/Dense>

#include "cuba.h"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
//#include "quadmath.h"

/* ---- Cuhre arguments ---- */
#define NDIM 4
#define NCOMP 1
#define NVEC 1
#define EPSREL 1e-3 //5e-2 is the default one
#define EPSRELGSL 1e-3 //5e-2
#define EPSABS 0
#define VERBOSE 0
#define LAST 4
#define MINEVAL 0
#define MAXEVAL 1e8
//#define LIMIT 1e8 // not sure needed here



#define KEY 0

#define STATEFILE NULL
#define SPIN NULL


/* ----- Vegas arguments ----- */

#define NSTART 1000
#define NINCREASE 500
#define NBATCH 1000


/* ---- Constants of nature   ---- */
//Aligned on what Luca has

#define GF 1.1663788*1e-11 
#define xw 0.23863        
#define e2 ((4*M_PI)/137.035999180)
#define m_e 0.51099895000      

/* -- Plasma 4-velocity -- */
//const double unit4vec[4]; // initialize later, not sure works to initialize vector out of scope...
           
/* ---- QAWS table parameters ---- */
#define alpha (-0.5)
#define beta (-0.5)
#define mu  0
#define nu  0

/* ---- power voids ----- */
#define pow2(x) ((x)*(x))
#define pow3(x) ((x)*(x)*(x))
#define pow4(x) ((x)*(x)*(x)*(x))
#define pow5(x) ((x)*(x)*(x)*(x)*(x))

/* Metric tensor  */

const double gmunu[4][4] = {
    { 1.0,  0.0,  0.0,  0.0},
    { 0.0, -1.0,  0.0,  0.0},
    { 0.0,  0.0, -1.0,  0.0},
    { 0.0,  0.0,  0.0, -1.0}
};


/* ---- Useful functions ---- */


/* Heaviside step function */
double Heaviside(double x){
    if(x > 0.) return 1.;
    return 0.;
}

double sign(double x){
    if(x > 0.) return 1.;
    else if (x < 0.) return -1.;
    else return 0.;
}

void printvector(double p[4]){
    cout << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << endl;
    return;
}

void print3vector(double p[3]){
    cout << p[0] << " " << p[1] << " " << p[2] << endl;
    return;
}

void printtensor(double I[4][4]){
    cout << I[0][0] << " " << I[0][1] << " " << I[0][2] << " " << I[0][3] << endl;
    cout << I[1][0] << " " << I[1][1] << " " << I[1][2] << " " << I[1][3] << endl;
    cout << I[2][0] << " " << I[2][1] << " " << I[2][2] << " " << I[2][3] << endl;
    cout << I[3][0] << " " << I[3][1] << " " << I[3][2] << " " << I[3][3] << endl;
    return;
}



void vec4(double E, double p1, double p2, double p3, double outputvector[4]){
    outputvector[0] = E;
    outputvector[1] = p1;
    outputvector[2] = p2;
    outputvector[3] = p3;
    return;
}

void vec4Ep(double E, double p[3], double outputvector[4]){
    outputvector[0] = E;
    outputvector[1] = p[0];
    outputvector[2] = p[1];
    outputvector[3] = p[2];
    return;
}

void vec3(double p[4], double outputvector[3]){
    outputvector[0] = p[1];
    outputvector[1] = p[2];
    outputvector[2] = p[3];
    return;
}

double norm3comp(double p1, double p2, double p3){
    return sqrt(pow2(p1)+pow2(p2)+pow2(p3));
}

double norm(double p[3]){
    return sqrt(pow2(p[0])+pow2(p[1])+pow2(p[2]));
}

//Any linear combination of 2 4-vectors
void combili4vector(double p[4], double q[4], double a, double b, double result[4]){ 
    result[0] = a*p[0]+b*q[0];
    result[1] = a*p[1]+b*q[1];
    result[2] = a*p[2]+b*q[2];
    result[3] = a*p[3]+b*q[3];
    return ;
}

double dottensor(double T[4][4], double p1[4], double p2[4]){
    double gmunu[4], dotproduct;
    dotproduct = 0.0;
    gmunu[0] = 1.0;
    for (int i=1;i<4;i++){
        gmunu[i] = -1.0;}
    for (int i=0;i<4;i++){
        for (int j=0;j<4;j++){
            dotproduct+= p1[i]*p2[j]*T[i][j]*gmunu[i]*gmunu[j];
        }
    }
    return dotproduct;
}


double dotvector(double p1[4], double p2[4]){
    double dotproduct;
    dotproduct = p1[0]*p2[0];
    for (int i=1;i<4;i++){
        dotproduct+= -1.0*p1[i]*p2[i];
    }
    return dotproduct;
}

double dot3vector(double p1[3], double p2[3]){
    double dotproduct;
    dotproduct = 0;
    for (int i=0;i<3;i++){
        dotproduct+= p1[i]*p2[i];
    }
    return dotproduct;
}


double trTensor(double T[4][4]){
    return T[0][0]-T[1][1]-T[2][2]-T[3][3];
}





void normalize(double p[3], double p_hat[3]) {
    double norm_p = norm(p);
    p_hat[0] = p[0]/norm_p;
    p_hat[1] = p[1]/norm_p;
    p_hat[2] = p[2]/norm_p;
}

// Cross product of 2 vectors
void cross(double p[3], double q[3], double k[3]) {
    k[0] = p[1]*q[2] - p[2]*q[1];
    k[1] = p[2]*q[0] - p[0]*q[2];
    k[2] = p[0]*q[1] - p[1]*q[0];
}

// Build an orthonormal basis from a vector p
void buildOrthonormalBasis(double p[3], double p_hat[3], double e1_hat[3], double e2_hat[3]) {
    double ref[3], cross_vec[3];
    normalize(p,p_hat);

    // Choose a reference vector that is not parallel to p_hat
    if (abs(p_hat[0]) < 0.9) {
      ref[0] = 1.0; ref[1] = 0.0; ref[2] = 0.0;
      } 
    else {
    ref[0] = 0.0; ref[1] = 1.0; ref[2] = 0.0;
    }

    // e1 = normalized cross product of p_hat and ref
    cross(p_hat, ref, cross_vec);
    normalize(cross_vec,e1_hat);

    // e2 = p_hat × e1
    cross(p_hat, e1_hat, e2_hat);
}






void extractAngles(double p[3],double q[3], double& theta, double& phi){ // p is the vector starting from which you do the basis construction
    double p_hat[3], e1_hat[3], e2_hat[3], q_hat[3];
    buildOrthonormalBasis(p, p_hat, e1_hat, e2_hat);
    normalize(q, q_hat);
    theta = acos(dotvector(p_hat,q_hat));
    phi = acos(dotvector(e1_hat,q_hat)/sin(theta));
}





double rho(double M, double m1, double m2){
    if (pow4(M)-2.0*pow2(M)*(pow2(m1)+pow2(m2)) + pow2(pow2(m1)-pow2(m2))>0.0){
        return 1.0/M/2.0*sqrt(pow4(M)-2.0*pow2(M)*(pow2(m1)+pow2(m2)) + pow2(pow2(m1)-pow2(m2)));}
    else{ return 0.0;}
}


double energy(double M, double p){
    return sqrt(pow2(M) + pow2(p));
}


double phi_hat_integration(double (*func)(double)){
    return M_PI*(func(M_PI/4.0) + func(3.0*M_PI/4.0) );
}


double nFD(double u[4], double p[4], double T){
    return 1.0/(exp(dotvector(u,p) / T) + 1.0 );
}

double nBE(double u[4], double p[4], double T){
    return 1.0 / (exp(dotvector(u,p) / T) - 1.0 );
}

double nFD_abs(double u[4], double p[4], double T){
    return 1.0 / (exp(fabs(dotvector(u,p)) / T) + 1.0 );
}

double nBE_abs(double u[4], double p[4], double T){
    return 1.0 / (exp(fabs(dotvector(u,p) / T)) - 1.0 );
}

double nB(double u[4], double p[4], double T){
    return 1.0 / exp(dotvector(u,p) / T);
}

double Fermi1D(double E, double T){
    return 1.0/(exp(E/T)+1.0);
}

double Bose1D(double E,double T){
    return 1.0/(exp(E/T)-1.0);
}
















/*----- Integrand for the vertex part where electrons on-shell ----*/ 


/* ------ Scalar part ----- */

 // integrate first k0 then |k|
struct Integrand_param_elec {double T; double p10; double p11; double p12; double p13; double p20; double p21; double p22; double p23; double sign; double lambda;}; 


//result of the integral of 1/(a*x^2+b*x+c)
double Integral_abc(double a, double b, double c){
  double Delta = pow2(b)-4*a*c;
  if (Delta>0.0){
  return 1/sqrt(Delta)*log(abs((2*a+b-sqrt(Delta))*(b+sqrt(Delta))/(2*a+b+sqrt(Delta))/(b-sqrt(Delta))));
  }
  else if (Delta<0.0){
  return 2/sqrt(-Delta)*(atan((2*a+b)/(sqrt(-Delta)))-atan(b/(sqrt(-Delta))));
  }
  else{
  return 4*a/(2*a+b)/b;
  }
}

double Integrand_elec(double kabs, void *pars){
    cubareal *q = (cubareal *) pars, T, p10, p11, p12, p13, p20, p21, p22, p23, sign, lambda;
    T = q[0]; 
    p10 = q[1];
    p11 = q[2]; 
    p12 = q[3]; 
    p13 = q[4]; 
    p20 = q[5];
    p21 = q[6]; 
    p22 = q[7]; 
    p23 = q[8]; 
    sign = q[9];
    lambda = q[10];
    
    double k = kabs/(1-kabs);
    double E = sqrt(pow2(k)+pow2(m_e));
    
    double p1[4], p2[4], p1mp2[4], p1vec[3], p2vec[3];
    vec4(p10, p11, p12, p13, p1);
    vec4(p20, p21, p22, p23, p2);
    vec3(p1, p1vec);
    vec3(p2, p2vec);
    combili4vector(p1, p2, 1.0, -1.0, p1mp2);
    double normp1sq = dot3vector(p1vec,p1vec);
    double normp2sq = dot3vector(p2vec,p2vec);
    double p1p2vecprod = dot3vector(p1vec,p2vec);
    
    double p1mp2sq = dotvector(p1mp2, p1mp2); 
    double Cxsq_x2coeff = pow2(p1mp2sq-2*pow2(m_e)+pow2(lambda));
    double Cxsq_x1coeff = 2*(2*pow2(m_e)-pow2(lambda))*(p1mp2sq-2*pow2(m_e)+pow2(lambda));
    double Cxsq_x0coeff = pow2(2*pow2(m_e)-pow2(lambda));
    double EQ0_x2coeff = 4*pow2(E)*pow2(p10);
    double EQ0_x1coeff = -8*pow2(E)*p10*p20;
    double EQ0_x0coeff = 4*pow2(E)*pow2(p20);
    double kQabs_x2coeff = -4*pow2(k)*normp1sq;
    double kQabs_x1coeff = 8*pow2(k)*p1p2vecprod;
    double kQabs_x0coeff = -4*pow2(k)*normp2sq;
    double ECx_x2coeff = 4*sign*E*p10*(p1mp2sq-2*pow2(m_e)+pow2(lambda));
    double ECx_x1coeff = -4*sign*E*p20*(p1mp2sq-2*pow2(m_e)+pow2(lambda))+4*sign*E*p10*(2*pow2(m_e)-pow2(lambda));
    double ECx_x0coeff = -4*sign*E*p20*(2*pow2(m_e)-pow2(lambda));
    return 1/pow2(2*M_PI)/pow2(1-kabs)*pow2(k)*Fermi1D(E,T)/E*Integral_abc(Cxsq_x2coeff+EQ0_x2coeff+kQabs_x2coeff+ECx_x2coeff, Cxsq_x1coeff+EQ0_x1coeff+kQabs_x1coeff+ECx_x1coeff, Cxsq_x0coeff+EQ0_x0coeff+kQabs_x0coeff+ECx_x0coeff);
}

double I_elec(double T, double p1[4], double p2[4], double lambda){
    double result1, result2, abserr; 
    size_t limit = 10000000,nevals; // size_t stores the maximum size of an array
    gsl_integration_workspace * w = gsl_integration_workspace_alloc(limit);
    gsl_function F; 
    // compute first part of kinematic space
    struct Integrand_param_elec  parameter1 = {T, p1[0], p1[1], p1[2], p1[3], p2[0], p2[1], p2[2], p2[3], 1.0, lambda};
    F.function =  &Integrand_elec;
    F.params = &parameter1; 
    gsl_integration_qag(&F, 0., 1., 0.,EPSRELGSL, limit, 2, w, &result1, &abserr);
    // compute second part of kinematic space
    struct Integrand_param_elec  parameter2 = {T, p1[0], p1[1], p1[2], p1[3], p2[0], p2[1], p2[2], p2[3], -1.0, lambda};
    F.function =  &Integrand_elec;
    F.params = &parameter2; 
    gsl_integration_qag(&F, 0., 1., 0.,EPSRELGSL, limit, 2, w, &result2, &abserr);
    // deallocate GSL workspace
    gsl_integration_workspace_free(w);
    return result1+result2;
}





/* ------ Vector part ----- */


/////// Ju

 // integrate first k0 then |k|
struct Integrand_param_elec_Ju {double T; double p10; double p11; double p12; double p13; double p20; double p21; double p22; double p23; double sign; double lambda;}; 

double Integrand_elec_Ju(double kabs, void *pars){
    cubareal *q = (cubareal *) pars, T, p10, p11, p12, p13, p20, p21, p22, p23, sign, lambda;
    T = q[0]; 
    p10 = q[1];
    p11 = q[2]; 
    p12 = q[3]; 
    p13 = q[4]; 
    p20 = q[5];
    p21 = q[6]; 
    p22 = q[7]; 
    p23 = q[8]; 
    sign = q[9];
    lambda = q[10];
    
    double k = kabs/(1-kabs);
    double E = sqrt(pow2(k)+pow2(m_e));
    
    double p1[4], p2[4], p1mp2[4], p1vec[3], p2vec[3];
    vec4(p10, p11, p12, p13, p1);
    vec4(p20, p21, p22, p23, p2);
    vec3(p1, p1vec);
    vec3(p2, p2vec);
    combili4vector(p1, p2, 1.0, -1.0, p1mp2);
    double normp1sq = dot3vector(p1vec,p1vec);
    double normp2sq = dot3vector(p2vec,p2vec);
    double p1p2vecprod = dot3vector(p1vec,p2vec);
    
    double p1mp2sq = dotvector(p1mp2, p1mp2); 
    double Cxsq_x2coeff = pow2(p1mp2sq-2*pow2(m_e)+pow2(lambda));
    double Cxsq_x1coeff = 2*(2*pow2(m_e)-pow2(lambda))*(p1mp2sq-2*pow2(m_e)+pow2(lambda));
    double Cxsq_x0coeff = pow2(2*pow2(m_e)-pow2(lambda));
    double EQ0_x2coeff = 4*pow2(E)*pow2(p10);
    double EQ0_x1coeff = -8*pow2(E)*p10*p20;
    double EQ0_x0coeff = 4*pow2(E)*pow2(p20);
    double kQabs_x2coeff = -4*pow2(k)*normp1sq;
    double kQabs_x1coeff = 8*pow2(k)*p1p2vecprod;
    double kQabs_x0coeff = -4*pow2(k)*normp2sq;
    double ECx_x2coeff = 4*sign*E*p10*(p1mp2sq-2*pow2(m_e)+pow2(lambda));
    double ECx_x1coeff = -4*sign*E*p20*(p1mp2sq-2*pow2(m_e)+pow2(lambda))+4*sign*E*p10*(2*pow2(m_e)-pow2(lambda));
    double ECx_x0coeff = -4*sign*E*p20*(2*pow2(m_e)-pow2(lambda));
    return 1/pow2(2*M_PI)/pow2(1-kabs)*pow2(k)*Fermi1D(E,T)*sign*Integral_abc(Cxsq_x2coeff+EQ0_x2coeff+kQabs_x2coeff+ECx_x2coeff, Cxsq_x1coeff+EQ0_x1coeff+kQabs_x1coeff+ECx_x1coeff, Cxsq_x0coeff+EQ0_x0coeff+kQabs_x0coeff+ECx_x0coeff);
}

double Ju_fun(double T, double p1[4], double p2[4], double Is, double lambda){
    double result1, result2, abserr; 
    size_t limit = 10000000,nevals; // size_t stores the maximum size of an array
    gsl_integration_workspace * w = gsl_integration_workspace_alloc(limit);
    gsl_function F; 
    // compute first part of kinematic space
    struct Integrand_param_elec_Ju  parameter1 = {T, p1[0], p1[1], p1[2], p1[3], p2[0], p2[1], p2[2], p2[3], 1.0, lambda};
    F.function =  &Integrand_elec_Ju;
    F.params = &parameter1; 
    gsl_integration_qag(&F, 0., 1., 0.,EPSRELGSL, limit, 2, w, &result1, &abserr);
    // compute second part of kinematic space
    struct Integrand_param_elec_Ju  parameter2 = {T, p1[0], p1[1], p1[2], p1[3], p2[0], p2[1], p2[2], p2[3], -1.0, lambda};
    F.function =  &Integrand_elec_Ju;
    F.params = &parameter2; 
    gsl_integration_qag(&F, 0., 1., 0.,EPSRELGSL, limit, 2, w, &result2, &abserr);
    // deallocate GSL workspace
    gsl_integration_workspace_free(w);
    return result1+result2 - p2[0]*Is;
}



////////////// J12

struct Integrand_param_elec_J12 {double T; double p0; double p1; double p2; double p3; double m;}; 

double Integrand_elec_J12(double kabs, void *pars){
    cubareal *q = (cubareal *) pars, T, p0, p1, p2, p3, m;
    T = q[0]; 
    p0 = q[1];
    p1 = q[2]; 
    p2 = q[3]; 
    p3 = q[4]; 
    m = q[5];
    
    double k = kabs/(1-kabs);
    double E = sqrt(pow2(k)+pow2(m_e)	);
    double pabs = norm3comp(p1,p2,p3);
    double psquared = pow2(p0)-pow2(pabs);

    return 1/(16*pow2(M_PI)*pabs)/pow2(1-kabs)*k*Fermi1D(E,T)/(2*E)*log(abs((pow2(psquared+pow2(m_e)-pow2(m)+2*k*pabs)-4*pow2(E)*pow2(p0))/(pow2(psquared+pow2(m_e)-pow2(m)-2*k*pabs)-4*pow2(E)*pow2(p0))));
}

double J12_fun(double T, double p[4], double m){
    double result, abserr; 
    size_t limit = 10000000,nevals; // size_t stores the maximum size of an array
    gsl_integration_workspace * w = gsl_integration_workspace_alloc(limit);
    gsl_function F; 
    // compute first part of kinematic space
    struct Integrand_param_elec_J12  parameter = {T, p[0], p[1], p[2], p[3], m};
    F.function =  &Integrand_elec_J12;
    F.params = &parameter; 
    gsl_integration_qag(&F, 0., 1., 0.,EPSRELGSL, limit, 2, w, &result, &abserr);
    // deallocate GSL workspace
    gsl_integration_workspace_free(w);
    return result;
}




void I_elec_Vec(double T, double p1[4], double p2[4], double Is, double lambda, double IV[4]){
    double p2mp1[4];
    combili4vector(p2, p1, 1.0, -1.0, p2mp1);

    double Ju = Ju_fun(T, p1, p2, Is, lambda);
    double J1 = J12_fun(T, p2, 0.0)-J12_fun(T, p2mp1, m_e);
    double J2 = -J12_fun(T, p2mp1, m_e);
    double E1 = p1[0];//both can be negative, just shortcut notation
    double E2 = p2[0];
    double p1p2 = dotvector(p1, p2);
    
    
    double Delta =-pow2(E1)*pow2(m_e)-pow2(E2)*pow2(m_e)+pow2(pow2(m_e))+2*E1*E2*p1p2-pow2(p1p2);
    
    double IV1 = ((pow2(m_e)-pow2(E2))*J1+(E1*E2-p1p2)*J2+(-E1*pow2(m_e)+E2*p1p2)*Ju)/Delta;
    double IV2 = ((E1*E2 - p1p2)*J1+(pow2(m_e) - pow2(E1))*J2+(-E2*pow2(m_e) + E1*p1p2)*Ju)/Delta;
    double IV3 = ((-E1*pow2(m_e) + E2*p1p2)*J1+(-E2*pow2(m_e) + E1*p1p2)*J2+(pow2(pow2(m_e)) - pow2(p1p2))*Ju)/Delta;
    
    double u_plasma[4];
    vec4(1.0, 0.0, 0.0, 0.0, u_plasma);
    
        // Reconstruct vector
    for(int i=0; i<4; i++){
        IV[i] = IV1*p1[i]+ IV2*p2[i]+ IV3*u_plasma[i];
    }
    return;
}




/*
// Try to invert matrix by hand to see if different, seems to give the same result

using Matrix3d = Eigen::Matrix<double,3,3>;
using Vector3d = array<double,3>;

Matrix3d inverse_GT_vector(double p1[4], double p2[4])
{
    double E1 = p1[0];
    double E2 = p2[0];
    double s  = dotvector(p1,p2);

    double me2 = m_e*m_e;

    Matrix3d G;

    G <<
    me2,      s,              E1,
    s,    me2,                E2,
    E1,      E2,              1.0;

    return G.inverse();
}



void I_elec_Vec(double T, double p1[4], double p2[4], double lambda, double IV[4]){
    double p2mp1[4];
    combili4vector(p2, p1, 1.0, -1.0, p2mp1);

    double Ju = Ju_fun(T, p1, p2, lambda);
    double J1 = J12_fun(T, p2, 0.0)-J12_fun(T, p2mp1, m_e);
    double J2 = -J12_fun(T, p2mp1, m_e);
    double E1 = p1[0];//both can be negative, just shortcut notation
    double E2 = p2[0];
    double p1p2 = dotvector(p1, p2);
    
    
    double Delta =-pow2(E1)*pow2(m_e)-pow2(E2)*pow2(m_e)+pow2(pow2(m_e))+2*E1*E2*p1p2-pow2(p1p2);
    
    double IV1 = ((pow2(m_e)-pow2(E2))*J1+(E1*E2-p1p2)*J2+(-E1*pow2(m_e)+E2*p1p2)*Ju)/Delta;
    double IV2 = ((E1*E2 - p1p2)*J1+(pow2(m_e) - pow2(E1))*J2+(-E2*pow2(m_e) + E1*p1p2)*Ju)/Delta;
    double IV3 = ((-E1*pow2(m_e) + E2*p1p2)*J1+(-E2*pow2(m_e) + E1*p1p2)*J2+(pow2(pow2(m_e)) - pow2(p1p2))*Ju)/Delta;
    
    double u_plasma[4];
    vec4(1.0, 0.0, 0.0, 0.0, u_plasma);
    
        // Reconstruct vector
    for(int i=0; i<4; i++){
        IV[i] = IV1*p1[i]+ IV2*p2[i]+ IV3*u_plasma[i];
    }
    
    
    // Define matrix necessary for inversion + do the product
    Matrix3d Ginv = inverse_GT_vector(p1,p2);
    Eigen::Matrix<double,3,1> Vi;
    Vi << J1, J2, Ju;
    Eigen::Matrix<double,3,1> IVcoeff = Ginv*Vi;
    
        // Reconstruct tensor
    for(int i=0; i<4; i++){
        IV[i] = IVcoeff(0)*p1[i]+ IVcoeff(1)*p2[i]+ IVcoeff(2)*u_plasma[i];
    }
    return;
}

*/






/* -------- Tensor part --------- */


// First provide the inverted matrix that links the integrals we will compute to the true tensor

using Matrix7d = Eigen::Matrix<double,7,7>;
using Vector4d = array<double,4>;


Matrix7d inverse_GT(double p1[4], double p2[4])
{
    double E1 = p1[0];
    double E2 = p2[0];
    double s  = dotvector(p1,p2);

    double me2 = m_e*m_e;
    double me4 = me2*me2;

    Matrix7d G;

    G <<
    me4,      2.0*me2*s,              2.0*me2*E1,            s*s,                    2.0*E1*s,               E1*E1,      me2,
    me2*s,    me4+s*s,                me2*E2+E1*s,          me2*s,                  E2*s+me2*E1,           E1*E2,      s,
    s*s,      2.0*me2*s,              2.0*E2*s,             me4,                    2.0*me2*E2,            E2*E2,      me2,
    me2*E1,   me2*E2+E1*s,            me2+E1*E1,            E2*s,                   s+E1*E2,               E1,         E1,
    E1*s,     E2*s+me2*E1,            s+E1*E2,              me2*E2,                 me2+E2*E2,             E2,         E2,
    E1*E1,    2.0*E1*E2,              2.0*E1,               E2*E2,                  2.0*E2,                1.0,        1.0,
    me2,      2.0*s,                  2.0*E1,               me2,                    2.0*E2,                1.0,        4.0;

    return G.inverse();
}


// Compute the basic functions necessary to extract the T_{ij}, I used ChatGPT for this code so to be reviewed...

struct Integrand_param_T{double T,p0,p1,p2,p3,m;};

double Integrand_T(double kabs, void *pars){

    Integrand_param_T *par = (Integrand_param_T *) pars;

    double T  = par->T;
    double p0 = par->p0;
    double p1 = par->p1;
    double p2 = par->p2;
    double p3 = par->p3;
    double m  = par->m;

    double k = kabs/(1.0-kabs);
    double eps = sqrt(pow2(k)+pow2(m_e));

    double pabs = norm3comp(p1,p2,p3);

    double A = pow2(pow2(p0)-pow2(pabs)+pow2(m_e)-pow2(m));

    double num = A - 4.0*pow2(eps*p0-k*pabs);
    double den = A - 4.0*pow2(eps*p0+k*pabs);

    double LogTerm = log(fabs(num/den));

    return 1.0/(16.0*pow2(M_PI)*pabs)*0.5*k*Fermi1D(eps,T)*LogTerm/pow2(1.0-kabs);
}

double T_fun(double T, double p[4], double m){

    double result, abserr;
    size_t limit = 10000000, nevals;

    gsl_integration_workspace *w = gsl_integration_workspace_alloc(limit);

    gsl_function F;

    Integrand_param_T parameter = {T,p[0],p[1],p[2],p[3],m};

    F.function = &Integrand_T;
    F.params = &parameter;

    gsl_integration_qag(&F,0.0,1.0,0.0,EPSRELGSL,limit,2,w,&result,&abserr);

    gsl_integration_workspace_free(w);

    return result;
}


struct Integrand_param_K{double T;};

double Integrand_K(double kabs, void *pars){

    Integrand_param_K *par = (Integrand_param_K *) pars;

    double T = par->T;

    double k = kabs/(1.0-kabs);
    double eps = sqrt(pow2(k)+pow2(m_e));

    return 1.0/(2.0*pow2(M_PI))*k*k/eps*Fermi1D(eps,T)/pow2(1.0-kabs);
}

double K_fun(double T){

    double result, abserr;
    size_t limit = 10000000, nevals;

    gsl_integration_workspace *w = gsl_integration_workspace_alloc(limit);

    gsl_function F;

    Integrand_param_K parameter = {T};

    F.function = &Integrand_K;
    F.params = &parameter;

    gsl_integration_qag(&F,0.0,1.0,0.0,EPSRELGSL,limit,2,w,&result,&abserr);

    gsl_integration_workspace_free(w);

    return result;
}


struct Integrand_param_L{double T,p0,p1,p2,p3,q0,q1,q2,q3,mp,mq;};

double Integrand_L(double kabs, void *pars){
    Integrand_param_L *par = (Integrand_param_L *) pars;

    double T  = par->T;
    double p0 = par->p0;
    double p1 = par->p1;
    double p2 = par->p2;
    double p3 = par->p3;
    double q0 = par->q0;
    double q1 = par->q1;
    double q2 = par->q2;
    double q3 = par->q3;
    double mp = par->mp;
    double mq = par->mq;

    double k = kabs/(1.0-kabs);
    double eps = sqrt(pow2(k)+pow2(m_e));
    double pabs = norm3comp(p1,p2,p3);
    double qabs = norm3comp(q1,q2,q3);
    double pdotq = p1*q1 + p2*q2 + p3*q3;
    double costhetap = pdotq/(pabs*qabs);
    double p2inv = pow2(p0)-pow2(pabs);
    double q2inv = pow2(q0)-pow2(qabs);

    double sumh = 0.0;
    for(int h=-1; h<=1; h+=2){
        
        double C1 = (p2inv + pow2(m_e) - pow2(mp)) - 2.0*h*eps*p0;
        double C2 = (q2inv + pow2(m_e) - pow2(mq)) - 2.0*h*eps*q0;

        double denom = C2 - 2.0*k*qabs;
        //if(fabs(denom) < 1e-14) denom = copysign(1e-14,denom);

        double LogTerm = log(fabs((C2 + 2.0*k*qabs)/denom));

        double Ih = 2.0*pabs*costhetap/qabs + (C1/(2.0*k*qabs) - pabs*costhetap*C2/(2.0*k*pow2(qabs)))*LogTerm;
        sumh += Ih;
    }

    return 1.0/(4.0*pow2(M_PI))*k*k/(2.0*eps)*Fermi1D(eps,T)*sumh/pow2(1.0-kabs);
}

double L_fun(double T, double p[4], double q[4], double mp, double mq){

    double result, abserr;
    size_t limit = 10000000, nevals;

    gsl_integration_workspace *w = gsl_integration_workspace_alloc(limit);

    gsl_function F;

    Integrand_param_L parameter = {T,p[0],p[1],p[2],p[3],q[0],q[1],q[2],q[3],mp,mq};

    F.function = &Integrand_L;
    F.params = &parameter;

    gsl_integration_qag(&F,0.0,1.0,0.0,EPSRELGSL,limit,2,w,&result,&abserr);

    gsl_integration_workspace_free(w);

    return result;
}


// Last integral, necessary in T_uu
 // integrate first k0 then |k|
struct Integrand_param_elec_tensor {double T; double p10; double p11; double p12; double p13; double p20; double p21; double p22; double p23; double sign; double lambda;}; 

double Integrand_elec_tensor(double kabs, void *pars){
    cubareal *q = (cubareal *) pars, T, p10, p11, p12, p13, p20, p21, p22, p23, sign, lambda;
    T = q[0]; 
    p10 = q[1];
    p11 = q[2]; 
    p12 = q[3]; 
    p13 = q[4]; 
    p20 = q[5];
    p21 = q[6]; 
    p22 = q[7]; 
    p23 = q[8]; 
    sign = q[9];
    lambda = q[10];
    
    double k = kabs/(1-kabs);
    double E = sqrt(pow2(k)+pow2(m_e));
    
    double p1[4], p2[4], p1mp2[4], p1vec[3], p2vec[3];
    vec4(p10, p11, p12, p13, p1);
    vec4(p20, p21, p22, p23, p2);
    vec3(p1, p1vec);
    vec3(p2, p2vec);
    combili4vector(p1, p2, 1.0, -1.0, p1mp2);
    double normp1sq = dot3vector(p1vec,p1vec);
    double normp2sq = dot3vector(p2vec,p2vec);
    double p1p2vecprod = dot3vector(p1vec,p2vec);
    
    double p1mp2sq = dotvector(p1mp2, p1mp2); 
    double Cxsq_x2coeff = pow2(p1mp2sq-2*pow2(m_e)+pow2(lambda));
    double Cxsq_x1coeff = 2*(2*pow2(m_e)-pow2(lambda))*(p1mp2sq-2*pow2(m_e)+pow2(lambda));
    double Cxsq_x0coeff = pow2(2*pow2(m_e)-pow2(lambda));
    double EQ0_x2coeff = 4*pow2(E)*pow2(p10);
    double EQ0_x1coeff = -8*pow2(E)*p10*p20;
    double EQ0_x0coeff = 4*pow2(E)*pow2(p20);
    double kQabs_x2coeff = -4*pow2(k)*normp1sq;
    double kQabs_x1coeff = 8*pow2(k)*p1p2vecprod;
    double kQabs_x0coeff = -4*pow2(k)*normp2sq;
    double ECx_x2coeff = 4*sign*E*p10*(p1mp2sq-2*pow2(m_e)+pow2(lambda));
    double ECx_x1coeff = -4*sign*E*p20*(p1mp2sq-2*pow2(m_e)+pow2(lambda))+4*sign*E*p10*(2*pow2(m_e)-pow2(lambda));
    double ECx_x0coeff = -4*sign*E*p20*(2*pow2(m_e)-pow2(lambda));
    return 1/pow2(2*M_PI)/pow2(1-kabs)*pow2(k)*Fermi1D(E,T)*E*Integral_abc(Cxsq_x2coeff+EQ0_x2coeff+kQabs_x2coeff+ECx_x2coeff, Cxsq_x1coeff+EQ0_x1coeff+kQabs_x1coeff+ECx_x1coeff, Cxsq_x0coeff+EQ0_x0coeff+kQabs_x0coeff+ECx_x0coeff);
}

double I_elec_tensor_uupart(double T, double p1[4], double p2[4], double lambda){
    double result1, result2, abserr; 
    size_t limit = 10000000,nevals; // size_t stores the maximum size of an array
    gsl_integration_workspace * w = gsl_integration_workspace_alloc(limit);
    gsl_function F; 
    // compute first part of kinematic space
    struct Integrand_param_elec_tensor  parameter1 = {T, p1[0], p1[1], p1[2], p1[3], p2[0], p2[1], p2[2], p2[3], 1.0, lambda};
    F.function =  &Integrand_elec_tensor;
    F.params = &parameter1; 
    gsl_integration_qag(&F, 0., 1., 0.,EPSRELGSL, limit, 2, w, &result1, &abserr);
    // compute second part of kinematic space
    struct Integrand_param_elec_tensor  parameter2 = {T, p1[0], p1[1], p1[2], p1[3], p2[0], p2[1], p2[2], p2[3], -1.0, lambda};
    F.function =  &Integrand_elec_tensor;
    F.params = &parameter2; 
    gsl_integration_qag(&F, 0., 1., 0.,EPSRELGSL, limit, 2, w, &result2, &abserr);
    // deallocate GSL workspace
    gsl_integration_workspace_free(w);
    return result1+result2;
}








// Combine everything into the final tensor


void I_elec_Tensor(double T, double p1[4], double p2[4], double Is, double IV[4], double lambda, double IT[4][4]){
    double u_plasma[4];
    vec4(1.0, 0.0, 0.0, 0.0, u_plasma);
    double p2mp1[4];
    combili4vector(p2, p1, 1.0, -1.0, p2mp1);
    double E1 = p1[0];//both can be negative, just shortcut notation
    double E2 = p2[0];
    double p1p2 = dotvector(p1, p2);

    
    // Doing Tuu first
    double Ju = Ju_fun(T, p1, p2, Is, lambda);
    double Tuu_newcontrib = I_elec_tensor_uupart( T, p1, p2, lambda);
    double Tuu = Tuu_newcontrib - 2*E2*Ju - pow2(E2)*Is;
    
    // Then Tg
    double Tg = -2*dotvector(p2,IV);
    
    // Then T1u and T2u
    double Tcalp2 = T_fun(T, p2, 0); //caligraphic T
    double Tcalp2mp1 = T_fun(T, p2mp1, m_e);
    double Jp2 = J12_fun(T, p2, 0.0);
    double Jp2mp1 = J12_fun(T, p2mp1, m_e);

    double T1u = Tcalp2-Tcalp2mp1-E2*(Jp2-Jp2mp1);
    double T2u = -Tcalp2mp1+E2*Jp2mp1;
    
    
    // Finally, T_11, T_22 and T_12
    double Lp2p2mp1 = L_fun(T, p2, p2mp1, 0.0, m_e);
    double Lp2mp1p2 = L_fun(T, p2mp1, p2, m_e, 0.0);
    double Kp2 = K_fun(T);
    
    double T22 = Lp2p2mp1/4;
    double T11 = (Lp2p2mp1+Lp2mp1p2)/4-Kp2/2;
    double T12 = Lp2p2mp1/4-Kp2/4;


    // Define matrix necessary for inversion + do the product
    Matrix7d Ginv = inverse_GT(p1,p2);
    Eigen::Matrix<double,7,1> Tij;
    Tij << T11, T12, T22, T1u, T2u, Tuu, Tg;
    Eigen::Matrix<double,7,1> ITcoeff = Ginv*Tij;
    
        // Reconstruct tensor
    for(int i=0; i<4; i++){
      for(int j=0; j<4; j++){
        IT[i][j] = ITcoeff(0)*p1[i]*p1[j]+ITcoeff(1)*(p1[i]*p2[j]+p1[j]*p2[i])+ITcoeff(2)*(p1[i]*u_plasma[j]+p1[j]*u_plasma[i])+ITcoeff(3)*p2[i]*p2[j]+ITcoeff(4)*(p2[i]*u_plasma[j]+p2[j]*u_plasma[i])+ITcoeff(5)*u_plasma[i]*u_plasma[j]+ITcoeff(6)*gmunu[i][j];
        }
    }
    return;
}
















/* ----- Test that Hadamard regularisation is taken care of by GSL ------- */

struct IntegrandTest_param {double a; double b;}; 

double IntegrandTest(double x, void *pars){
    cubareal *q = (cubareal *) pars, a, b;
    a = q[0]; 
    b = q[1];
    return 1/(a+b*x);
}

double I_test(double a, double b){
    double result, abserr; 
    size_t limit = 10000000,nevals; // size_t stores the maximum size of an array
    gsl_integration_workspace * w = gsl_integration_workspace_alloc(limit);
    gsl_function F; 
    struct IntegrandTest_param  parameter = {a, b};
    F.function =  &IntegrandTest;
    F.params = &parameter; 
    gsl_integration_qag(&F, 0., 1., 0.,EPSRELGSL, limit, 2, w, &result, &abserr);
    // deallocate GSL workspace
    gsl_integration_workspace_free(w);
    return result;
}




/* ------- Main function ------ */




// /*
int main(int argc, char** argv){
    FILE *fptr;
    fptr = fopen("/home/ygeoris/c++code/Dataset/VertexThermalIntegrals_newparam_withJ1_zz.csv","w+"); 
    fprintf(fptr,"EPSRELGSL = %e\n", EPSRELGSL);
    fprintf(fptr,"T p10 p11 p12 p13 p20 p21 p22 p23 lambda Is IV0 IV1 IV2 IV3 IT00 IT01 IT02 IT03 IT10 IT11 IT12 IT13 IT20 IT21 IT22 IT23 IT30 IT31 IT32 IT33 J1 \n");
    double size = 14.0;
    double T, p1[4], p2[4];
    for(int i=0;i<size;i++){
        T = 1.0;
        double p11 = 0.8;
        double p12 = 0.1;
        double p13 = 0.05 + 3.0/size*i;
        double E1 = energy(m_e, norm3comp(p11, p12, p13));
        double p21 = 0.25;
        double p22 = 0.4;
        double p23 = 0.5;
        double E2 = energy(m_e, norm3comp(p21, p22, p23));
        vec4(E1,p11,p12,p13,p1);
        vec4(E2,p21,p22,p23,p2);
        double lambda = 0.000003;
        double Is = I_elec(T, p1, p2, lambda);
        cout << Is << endl;
        
        double IV[4], IT[4][4]; //, IVp[4] // IV[0]+IVp[0], IV[1]+IVp[1], IV[2]+IVp[2], IV[3]+IVp[3],
        I_elec_Vec(T, p1, p2, Is, lambda, IV);
        //I_elec_Vec(T, p2, p1, Is, lambda*10, IVp);
        I_elec_Tensor(T, p1, p2, Is, IV, lambda, IT);
        
        cout << i << endl;
    	fprintf(fptr,"%e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e \n",T, p1[0], p1[1], p1[2], p1[3], p2[0], p2[1], p2[2], p2[3], lambda, Is, IV[0], IV[1], IV[2], IV[3], IT[0][0], IT[0][1], IT[0][2], IT[0][3], IT[1][0], IT[1][1], IT[1][2], IT[1][3], IT[2][0], IT[2][1], IT[2][2], IT[2][3], IT[3][0], IT[3][1], IT[3][2], IT[3][3], 2*J12_fun(T, p2, 0.0));}
    fclose(fptr);
    return 0;
}
// */   




/*
int main()
{
    double p1[4] = {10.0,1.0,2.0,3.0};
    double p2[4] = {12.0,-1.0,0.5,4.0};

    Matrix7d Ginv = inverse_GT(p1,p2);

    cout << "G^{-1} =\n" << Ginv << std::endl;
    cout << Ginv(1,1) << " " << Ginv(0,0) << endl;
    
    cout << L_fun(1.0, p1, p2, 0.0, m_e) << " " << T_fun(1.0, p1, m_e) << " " << K_fun(1.0) << endl;
    double lambda = 0.000003;
    double T=1.0;
    double Is = I_elec(T, p1, p2, lambda);
    cout << Is << endl;
    
    double IV[4], IT[4][4];
    I_elec_Vec(T, p1, p2, lambda, IV);
    I_elec_Tensor(T, p1, p2, lambda, Is, IV, IT);
    
    printtensor(IT);

    return 0;
}
*/


    
    
    
    /*
int main(int argc, char** argv){
    cout << I_test(1.0, -4.0) << endl;
    return 0;
}
*/
