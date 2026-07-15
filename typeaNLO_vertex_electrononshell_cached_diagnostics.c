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
#define EPSREL 1e-1 //5e-2 is the default one
#define EPSRELGSL 1e-3 // inner quadrature must be substantially tighter than Cuhre
// The raw Cuhre integral is normally O(1). This absolute floor prevents a
// cancellation-dominated channel from demanding an unattainable relative error.
#define EPSABS 1e-6
#define VERBOSE 0
#define LAST 4
#define MINEVAL 0
#define DEFAULT_MAXEVAL 200000
#define GSL_LIMIT 5000 // convergence-test this in the range 1000--10000
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

//p is the initial vector, boosted by pIn. Final result is boosted[4]
void boost(double pIn[4], double p[4], double boosted[4]){
    double betaX, betaY, betaZ, gamma, prod1, prod2;
    betaX = -pIn[1] / pIn[0];
    betaY = -pIn[2] / pIn[0];
    betaZ = -pIn[3] / pIn[0];
    if (1.0 - pow2(betaX) - pow2(betaY) - pow2(betaZ)>0.0){
        gamma = 1.0 / sqrt(1.0 - pow2(betaX) - pow2(betaY) - pow2(betaZ));
        prod1 = betaX * p[1] + betaY * p[2] + betaZ * p[3];
        prod2 = gamma * (gamma * prod1 / (1.0 + gamma) + p[0]);
        boosted[1] = p[1] + prod2 * betaX;
        boosted[2] = p[2] + prod2 * betaY;
        boosted[3] = p[3] + prod2 * betaZ;
        boosted[0] =  gamma * (p[0] + prod1);}
    return;
}

void boost_inv(double pIn[4], double p[4], double inv_boosted[4]){
    double betaX, betaY, betaZ, gamma, prod1, prod2;
    betaX = pIn[1] / pIn[0];
    betaY = pIn[2] / pIn[0];
    betaZ = pIn[3] / pIn[0];
    if (1.0 - pow2(betaX) - pow2(betaY) - pow2(betaZ)>0.0){
        gamma = 1.0 / sqrt(1.0 - pow2(betaX) - pow2(betaY) - pow2(betaZ));
        prod1 = betaX * p[1] + betaY * p[2] + betaZ * p[3];
        prod2 = gamma * (gamma * prod1 / (1. + gamma) + p[0]);
        inv_boosted[1] = p[1] + prod2 * betaX;
        inv_boosted[2] = p[2] + prod2 * betaY;
        inv_boosted[3] = p[3] + prod2 * betaZ;
        inv_boosted[0] =  gamma * (p[0] + prod1);}
    return;
}

void boost_matrix(double pIn[4], double Lorentz_transfo[4][4]){
    double betaX, betaY, betaZ, gamma;
    betaX = -pIn[1] / pIn[0];
    betaY = -pIn[2] / pIn[0];
    betaZ = -pIn[3] / pIn[0];
    if (1.0 - pow2(betaX) - pow2(betaY) - pow2(betaZ)>0.0){
        gamma = 1.0 / sqrt(1.0 - pow2(betaX) - pow2(betaY) - pow2(betaZ));
        Lorentz_transfo[0][0] = gamma;
        Lorentz_transfo[0][1] = gamma*betaX;
        Lorentz_transfo[0][2] = gamma*betaY;
        Lorentz_transfo[0][3] = gamma*betaZ;
        Lorentz_transfo[1][0] = gamma*betaX;
        Lorentz_transfo[1][1] = 1.0+pow2(gamma)/(1.0+gamma)*pow2(betaX);
        Lorentz_transfo[1][2] = pow2(gamma)/(1.0+gamma)*betaX*betaY;
        Lorentz_transfo[1][3] = pow2(gamma)/(1.0+gamma)*betaX*betaZ;
        Lorentz_transfo[2][0] = gamma*betaY;
        Lorentz_transfo[2][1] = pow2(gamma)/(1.0+gamma)*betaX*betaY;
        Lorentz_transfo[2][2] = 1.0+pow2(gamma)/(1.0+gamma)*pow2(betaY);
        Lorentz_transfo[2][3] = pow2(gamma)/(1.0+gamma)*betaY*betaZ;
        Lorentz_transfo[3][0] = gamma*betaZ;
        Lorentz_transfo[3][1] = pow2(gamma)/(1.0+gamma)*betaX*betaZ;
        Lorentz_transfo[3][2] = pow2(gamma)/(1.0+gamma)*betaY*betaZ;
        Lorentz_transfo[3][3] = 1.0+pow2(gamma)/(1.0+gamma)*pow2(betaZ);
            }
    return;
}


void boost_tensor(double pIn[4], double T[4][4], double boosted_tensor[4][4]){
    double Lorentz_transfo[4][4];
    boost_matrix(pIn,Lorentz_transfo);
    for (int i=0;i<4;i++){
        for (int j=0;j<4;j++){
            boosted_tensor[i][j] = 0;
            for (int k=0;k<4;k++){
                for (int l=0;l<4;l++){
                    boosted_tensor[i][j] += Lorentz_transfo[i][k]*Lorentz_transfo[j][l]*T[k][l];}
            }
        }
    }
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


//Any linear combination of 2 tensors
void combilitensor(double I[4][4], double J[4][4], double a, double b, double result[4][4]){ 
    for (int i=0;i<4;i++){
        for (int j=0;j<4;j++){
            result[i][j] = a*I[i][j]+b*J[i][j];
        }
    }
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
    //cout << "q: " << q[0] << " " << q[1] << " " << q[2] << endl;
    buildOrthonormalBasis(p, p_hat, e1_hat, e2_hat);
    normalize(q, q_hat);
    //cout << "q_hat: " << q_hat[0] << " " << q_hat[1] << " " << q_hat[2] << endl;
    //cout << "p_hat: " << p_hat[0] << " " << p_hat[1] << " " << p_hat[2] << endl;
    //cout << "extractangle " << dotvector(p_hat,q_hat) << endl;
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










/* ----- Reusable, diagnosed GSL integration ----- */

struct QagResult {
    double value;
    double abserr;
    int status;
    size_t subdivisions;
};

struct InnerQuadratureStats {
    unsigned long calls;
    unsigned long failures;
    size_t max_subdivisions;
    double max_abserr;
};

static thread_local InnerQuadratureStats inner_stats = {0, 0, 0, 0.0};

class WorkerWorkspace {
public:
    WorkerWorkspace() : workspace(gsl_integration_workspace_alloc(GSL_LIMIT)) {
        if (workspace == NULL) {
            fprintf(stderr, "fatal: unable to allocate GSL workspace (limit=%d)\n", GSL_LIMIT);
            abort();
        }
    }
    ~WorkerWorkspace() { gsl_integration_workspace_free(workspace); }
    gsl_integration_workspace *get() { return workspace; }
private:
    gsl_integration_workspace *workspace;
};

static gsl_integration_workspace *worker_workspace(){
    static thread_local WorkerWorkspace holder;
    return holder.get();
}

static QagResult qag_checked(gsl_function *F, const char *label){
    QagResult out = {NAN, NAN, GSL_EFAILED, 0};
    gsl_integration_workspace *w = worker_workspace();
    out.status = gsl_integration_qag(F, 0.0, 1.0, 0.0, EPSRELGSL,
                                     GSL_LIMIT, 2, w, &out.value, &out.abserr);
    out.subdivisions = w->size;
    inner_stats.calls++;
    if (out.subdivisions > inner_stats.max_subdivisions)
        inner_stats.max_subdivisions = out.subdivisions;
    if (isfinite(out.abserr) && out.abserr > inner_stats.max_abserr)
        inner_stats.max_abserr = out.abserr;
    if (out.status != GSL_SUCCESS || !isfinite(out.value) || !isfinite(out.abserr)) {
        inner_stats.failures++;
        fprintf(stderr,
                "GSL qag failure: integral=%s status=%d (%s) value=% .9e "
                "abserr=%.3e subdivisions=%zu/%d\n",
                label, out.status, gsl_strerror(out.status), out.value,
                out.abserr, out.subdivisions, GSL_LIMIT);
        out.value = NAN;
    }
    return out;
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
    gsl_function F;
    struct Integrand_param_elec parameter1 = {T, p1[0], p1[1], p1[2], p1[3], p2[0], p2[1], p2[2], p2[3], 1.0, lambda};
    F.function = &Integrand_elec;
    F.params = &parameter1;
    QagResult first = qag_checked(&F, "Is(+)");
    struct Integrand_param_elec parameter2 = {T, p1[0], p1[1], p1[2], p1[3], p2[0], p2[1], p2[2], p2[3], -1.0, lambda};
    F.params = &parameter2;
    QagResult second = qag_checked(&F, "Is(-)");
    return first.value + second.value;
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
    gsl_function F;
    struct Integrand_param_elec_Ju parameter1 = {T, p1[0], p1[1], p1[2], p1[3], p2[0], p2[1], p2[2], p2[3], 1.0, lambda};
    F.function = &Integrand_elec_Ju;
    F.params = &parameter1;
    QagResult first = qag_checked(&F, "Ju(+)");
    struct Integrand_param_elec_Ju parameter2 = {T, p1[0], p1[1], p1[2], p1[3], p2[0], p2[1], p2[2], p2[3], -1.0, lambda};
    F.params = &parameter2;
    QagResult second = qag_checked(&F, "Ju(-)");
    return first.value + second.value - p2[0]*Is;
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
    double E = sqrt(pow2(k)+pow2(m_e));
    double pabs = norm3comp(p1,p2,p3);
    double psquared = pow2(p0)-pow2(pabs);

    return 1/(16*pow2(M_PI)*pabs)/pow2(1-kabs)*k*Fermi1D(E,T)/(2*E)*log(abs((pow2(psquared+pow2(m_e)-pow2(m)+2*k*pabs)-4*pow2(E)*pow2(p0))/(pow2(psquared+pow2(m_e)-pow2(m)-2*k*pabs)-4*pow2(E)*pow2(p0))));
}

double J12_fun(double T, double p[4], double m){
    gsl_function F;
    struct Integrand_param_elec_J12 parameter = {T, p[0], p[1], p[2], p[3], m};
    F.function = &Integrand_elec_J12;
    F.params = &parameter;
    return qag_checked(&F, "J12").value;
}




struct VertexBasis {
    double Is, Ju;
    double Jgamma, Je;
    double Tgamma, Te;
    double Lgammae, Legamma;
    double K, Tuu_direct;
    bool valid;
};

void I_elec_Vec(double p1[4], double p2[4], const VertexBasis &basis, double IV[4]){
    double J1 = basis.Jgamma-basis.Je;
    double J2 = -basis.Je;
    double E1 = p1[0];
    double E2 = p2[0];
    double p1p2 = dotvector(p1, p2);
    double Delta =-pow2(E1)*pow2(m_e)-pow2(E2)*pow2(m_e)+pow2(pow2(m_e))+2*E1*E2*p1p2-pow2(p1p2);
    double IV1 = ((pow2(m_e)-pow2(E2))*J1+(E1*E2-p1p2)*J2+(-E1*pow2(m_e)+E2*p1p2)*basis.Ju)/Delta;
    double IV2 = ((E1*E2 - p1p2)*J1+(pow2(m_e) - pow2(E1))*J2+(-E2*pow2(m_e) + E1*p1p2)*basis.Ju)/Delta;
    double IV3 = ((-E1*pow2(m_e) + E2*p1p2)*J1+(-E2*pow2(m_e) + E1*p1p2)*J2+(pow2(pow2(m_e)) - pow2(p1p2))*basis.Ju)/Delta;
    double u_plasma[4];
    vec4(1.0, 0.0, 0.0, 0.0, u_plasma);
    for(int i=0; i<4; i++)
        IV[i] = IV1*p1[i]+ IV2*p2[i]+ IV3*u_plasma[i];
}





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
    gsl_function F;
    Integrand_param_T parameter = {T,p[0],p[1],p[2],p[3],m};
    F.function = &Integrand_T;
    F.params = &parameter;
    return qag_checked(&F, "T").value;
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
    gsl_function F;
    Integrand_param_K parameter = {T};
    F.function = &Integrand_K;
    F.params = &parameter;
    return qag_checked(&F, "K").value;
}

static double cached_K_temperature = NAN;
static double cached_K_value = NAN;

static double K_cached(double T){
    if (!isfinite(cached_K_value) || T != cached_K_temperature) {
        cached_K_value = K_fun(T);
        cached_K_temperature = T;
    }
    return cached_K_value;
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
    gsl_function F;
    Integrand_param_L parameter = {T,p[0],p[1],p[2],p[3],q[0],q[1],q[2],q[3],mp,mq};
    F.function = &Integrand_L;
    F.params = &parameter;
    return qag_checked(&F, "L").value;
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
    gsl_function F;
    struct Integrand_param_elec_tensor parameter1 = {T, p1[0], p1[1], p1[2], p1[3], p2[0], p2[1], p2[2], p2[3], 1.0, lambda};
    F.function = &Integrand_elec_tensor;
    F.params = &parameter1;
    QagResult first = qag_checked(&F, "Tuu(+)");
    struct Integrand_param_elec_tensor parameter2 = {T, p1[0], p1[1], p1[2], p1[3], p2[0], p2[1], p2[2], p2[3], -1.0, lambda};
    F.params = &parameter2;
    QagResult second = qag_checked(&F, "Tuu(-)");
    return first.value + second.value;
}








static VertexBasis build_vertex_basis(double T, double p1[4], double p2[4],
                                      double lambda, double cached_K){
    VertexBasis basis;
    double p2mp1[4];
    combili4vector(p2, p1, 1.0, -1.0, p2mp1);
    basis.Is = I_elec(T, p1, p2, lambda);
    basis.Ju = Ju_fun(T, p1, p2, basis.Is, lambda);
    basis.Jgamma = J12_fun(T, p2, lambda);
    basis.Je = J12_fun(T, p2mp1, m_e);
    basis.Tgamma = T_fun(T, p2, lambda);
    basis.Te = T_fun(T, p2mp1, m_e);
    basis.Lgammae = L_fun(T, p2, p2mp1, lambda, m_e);
    basis.Legamma = L_fun(T, p2mp1, p2, m_e, lambda);
    basis.Tuu_direct = I_elec_tensor_uupart(T, p1, p2, lambda);
    basis.K = cached_K;
    basis.valid = isfinite(basis.Is) && isfinite(basis.Ju) &&
                  isfinite(basis.Jgamma) && isfinite(basis.Je) &&
                  isfinite(basis.Tgamma) && isfinite(basis.Te) &&
                  isfinite(basis.Lgammae) && isfinite(basis.Legamma) &&
                  isfinite(basis.Tuu_direct) && isfinite(basis.K);
    return basis;
}

// Combine everything into the final tensor


void I_elec_Tensor(double p1[4], double p2[4], const VertexBasis &basis,
                   double IV[4], double IT[4][4]){
    double u_plasma[4];
    vec4(1.0, 0.0, 0.0, 0.0, u_plasma);
    double E2 = p2[0];

    double Tuu = basis.Tuu_direct - 2*E2*basis.Ju - pow2(E2)*basis.Is;
    double Tg = -2*dotvector(p2,IV);
    double T1u = basis.Tgamma-basis.Te-E2*(basis.Jgamma-basis.Je);
    double T2u = -basis.Te+E2*basis.Je;
    double T22 = basis.Lgammae/4;
    double T11 = (basis.Lgammae+basis.Legamma)/4-basis.K/2;
    double T12 = basis.Lgammae/4-basis.K/4;

    Matrix7d Ginv = inverse_GT(p1,p2);
    Eigen::Matrix<double,7,1> Tij;
    Tij << T11, T12, T22, T1u, T2u, Tuu, Tg;
    Eigen::Matrix<double,7,1> ITcoeff = Ginv*Tij;
    for(int i=0; i<4; i++){
      for(int j=0; j<4; j++){
        IT[i][j] = ITcoeff(0)*p1[i]*p1[j]+ITcoeff(1)*(p1[i]*p2[j]+p1[j]*p2[i])+ITcoeff(2)*(p1[i]*u_plasma[j]+p1[j]*u_plasma[i])+ITcoeff(3)*p2[i]*p2[j]+ITcoeff(4)*(p2[i]*u_plasma[j]+p2[j]*u_plasma[i])+ITcoeff(5)*u_plasma[i]*u_plasma[j]+ITcoeff(6)*gmunu[i][j];
      }
    }
}































/*------------ Computing the traces ------------*/



//Very partial at the moment
double tau1(double P1[4], double P2[4], double P3[4], double Q[4], double Is, double IV[4], double It[4][4], double m1, double m2, double m3, double gL, double gR, double T){
    double P1Q, P2Q, P3Q;
    double P1IV, P2IV, P3IV, QIV;
    double P1P2, P1P3, P2P3, res;
    double IP3Q, IP2Q, IP1P2, IP1Q, IP1P3, IP2P3;
    P1Q = dotvector(P1,Q); 
    P2Q = dotvector(P2,Q);
    P3Q = dotvector(P3,Q); 
    P1P2 = dotvector(P1,P2); 
    P1P3 = dotvector(P1,P3); 
    P2P3 = dotvector(P2,P3); 
    
    P1IV = dotvector(P1,IV); 
    P2IV = dotvector(P2,IV);
    P3IV = dotvector(P3,IV); 
    QIV = dotvector(Q,IV); 

    IP1Q = dottensor(It,P1,Q);
    IP2Q = dottensor(It,P2,Q);
    IP3Q = dottensor(It,P3,Q);
    IP1P2 = dottensor(It,P1,P2);
    IP2P3 = dottensor(It,P2,P3);
    IP1P3 = dottensor(It,P1,P3);
    double Itrace = trTensor(It);

    
    res = sign(m1*m2)*Is*64*P1P2*(pow2(gL)*P1P3*P2Q-gL*gR*m1*m2*P3Q+pow2(gR)*P1Q*P2P3); // Scalar part
    
    
    res += 32*sign(m2)*(-(pow2(gL)+pow2(gR))*P1IV*P2P3*P2Q+ gL*((P2IV)*(gL*P1P3*P2Q - gR*m1*m2*P3Q)+ gL*P1P2*P2Q*P3IV)+ pow2(gR)*P1Q*P2P3*P2IV+ pow2(gR)*P1P2*P2P3*QIV); // First vector contribution
    res += 32*sign(m1)*(P1Q*(-(pow2(gL)+pow2(gR))*P1P3*P2IV+ pow2(gR)*P1IV*P2P3+ pow2(gR)*P1P2*P3IV)+ gL*( gL*P1P3*(P1IV*P2Q + P1P2*QIV)- gR*m1*m2*P1IV*P3Q )); // Second vector contribution
    
    
    //res += 32*(IP3Q*((pow2(gL)+pow2(gR))*P1P2-2*gL*gR*m1*m2)-pow2(gR)*P1P3*IP2Q+(pow2(gL)+pow2(gR))*IP1P2*P3Q-pow2(gL)*P2P3*IP1Q-pow2(gR)*P2Q*IP1P3-pow2(gL)*P1Q*IP2P3)+ 16*Itrace*(pow2(gL)+pow2(gR))*(P1Q*P2P3+P1P3*P2Q-P1P2*P3Q); //Correct is below, here I just test stuff...
    res += 32*(IP3Q*((pow2(gL)+pow2(gR))*P1P2- 2*gL*gR*m1*m2)-pow2(gR)*P1P3*IP2Q+(pow2(gL)+pow2(gR))*IP1P2*P3Q-pow2(gL)*P2P3*IP1Q-pow2(gR)*P2Q*IP1P3-pow2(gL)*P1Q*IP2P3)+ 16*Itrace*(pow2(gL)+pow2(gR))*(P1Q*P2P3+P1P3*P2Q-P1P2*P3Q);
 
    return res;
}




/*------------ Adding the Boltzmann distribution ------------*/


double prefacttau12to2p3(double u[4], double P1[4], double P2[4], double P3[4], double Q[4], double Is, double IV[4], double It[4][4], double T, double m1, double m2, double m3, double gL, double gR){
    return (1-nFD(u, P1, T))*(1-nFD(u, P2, T))*nFD(u, P3, T)*tau1(P1, P2, P3, Q, Is, IV, It, m1, m2, m3, gL, gR, T);
}

double prefacttau12to2p1(double u[4], double P1[4], double P2[4], double P3[4], double Q[4], double Is, double IV[4], double It[4][4], double T, double m1, double m2, double m3, double gL, double gR){
    return (1-nFD(u, P3, T))*(1-nFD(u, P2, T))*nFD(u, P1, T)*tau1(P1, P2, P3, Q, Is, IV, It, m1, m2, m3, gL, gR, T);
}

double prefacttau12to2p2(double u[4], double P1[4], double P2[4], double P3[4], double Q[4], double Is, double IV[4], double It[4][4], double T, double m1, double m2, double m3, double gL, double gR){
    return (1-nFD(u, P1, T))*(1-nFD(u, P3, T))*nFD(u, P2, T)*tau1(P1, P2, P3, Q, Is, IV, It, m1, m2, m3, gL, gR, T);
}




























/* ----- Integrand of the full rates (boosted to the CM frame) ---- */


/* -------- p3 -------- */

double integrand2to2p3(double p3mod, double theta, double theta_hat, double phi_hat, double m1, double m2, double m3, double q, double T, double gL, double gR, double lambda){
    double qvec[3], p3vec[3], normp3q, Q[4], P3[4], P3Q[4], e[3], Q_hat[4], P3_hat[4], unit4vec[4], u_hat[4], Is, IV[4], IV_lab[4], It[4][4], It_lab[4][4], E3_hat, q0_hat, p1_hatmod, Wp1_hat[3], P1[4], P2[4], P1_lab[4], P2_lab[4], P1P2_lab[4], multSZ, mult0, mult1, mult2, mult3, mult4;//ex, ey, ez, not needed for now.
    // define unit4vec
    unit4vec[0] = 1.0;
    unit4vec[1] = 0.0;
    unit4vec[2] = 0.0;
    unit4vec[3] = 0.0;
    // define qvec
    qvec[0] = 0.0;
    qvec[1] = 0.0;
    qvec[2] = q;
    // define p3vec
    p3vec[0] = p3mod*sin(theta);
    p3vec[1] = 0.0;
    p3vec[2] = p3mod*cos(theta);
    //
    vec4Ep(energy(0, q), qvec, Q);
    vec4Ep(energy(m3, p3mod), p3vec, P3);
    // define P3+Q
    combili4vector(P3,Q,1,1,P3Q);
    // define e
    normp3q = sqrt(pow2(qvec[0] + p3vec[0])+pow2(qvec[1] + p3vec[1])+pow2(qvec[2] + p3vec[2]));
    e[0] = (qvec[0] + p3vec[0])/normp3q; //norm(qvec + p3vec)
    e[1] = (qvec[1] + p3vec[1])/normp3q;
    e[2] = (qvec[2] + p3vec[2])/normp3q;
    boost(P3Q, Q, Q_hat);
    boost(P3Q, P3, P3_hat);
    boost(P3Q, unit4vec, u_hat);
    E3_hat = P3_hat[0];
    q0_hat = Q_hat[0];
    if(E3_hat + q0_hat - m1 - m2 > 0.0){
        p1_hatmod = rho( E3_hat + q0_hat, m1, m2);
        // define Wp1_hat*P1_hatmod if use convention of Julia notebook
        Wp1_hat[0] = p1_hatmod*(e[2]*sin(theta_hat)*cos(phi_hat) + e[0]*cos(theta_hat));
        Wp1_hat[1] = p1_hatmod*(sin(theta_hat)*sin(phi_hat));
        Wp1_hat[2] = p1_hatmod*(e[2]*cos(theta_hat) - e[0]*sin(theta_hat)*cos(phi_hat)); 
        vec4Ep(energy(m1, p1_hatmod),  Wp1_hat, P1);
        // add minus sign of difference
        Wp1_hat[0] *= -1.0; 
        Wp1_hat[1] *= -1.0; 
        Wp1_hat[2] *= -1.0;
        vec4Ep(energy(m2, p1_hatmod), Wp1_hat, P2);

        boost_inv(P3Q,P1,P1_lab);
        boost_inv(P3Q,P2,P2_lab); // Find P1, P2 in the lab frame in order to compute the integral...
        
        
        double P2_labm[4];
        combili4vector(P2_lab,Q,-1.0,0.0,P2_labm);
        
        double cached_K = K_cached(T);
        VertexBasis basis1 = build_vertex_basis(T, P1_lab, P2_labm, lambda, cached_K);
        VertexBasis basis2 = build_vertex_basis(T, P2_labm, P1_lab, lambda, cached_K);
        if (!basis1.valid || !basis2.valid) {
            fprintf(stderr, "reject p3 point: p=%.9e theta=%.9e theta_hat=%.9e phi_hat=%.9e\n", p3mod, theta, theta_hat, phi_hat);
            return 0.0;
        }
        Is = basis1.Is + basis2.Is;
        double IV_lab1[4], IV_lab2[4];
        I_elec_Vec(P1_lab, P2_labm, basis1, IV_lab1);
        I_elec_Vec(P2_labm, P1_lab, basis2, IV_lab2);
        combili4vector(IV_lab1,IV_lab2,1.0,1.0,IV_lab);
        boost(P3Q,IV_lab, IV);
        double It_lab1[4][4], It_lab2[4][4];
        I_elec_Tensor(P1_lab, P2_labm, basis1, IV_lab1, It_lab1);
        I_elec_Tensor(P2_labm, P1_lab, basis2, IV_lab2, It_lab2);
        combilitensor(It_lab1,It_lab2,1.0,1.0,It_lab);
        boost_tensor(P3Q,It_lab, It);
        
        
        /* OLD momenta, wrong, forgot sign change
        Is = I_elec(T, P1_lab, P2_lab, lambda);
        I_elec_Vec(T, P1_lab, P2_lab, Is, lambda, IV_lab);
        boost(P3Q,IV_lab, IV);
        I_elec_Tensor(T, P1_lab, P2_lab, Is, IV_lab, lambda, It_lab);
        boost_tensor(P3Q,It_lab, It); // boost tensor in correct frame...
        */

        multSZ = 1.0/pow2(4*M_PI);
        mult0 = 1.0/pow2(2*M_PI);
        mult1 = pow2(p3mod) / (2* energy(m3, p3mod));
        mult2 = sin(theta)*sin(theta_hat);
        mult3 = rho( E3_hat + q0_hat, m1, m2) / (E3_hat + q0_hat);//P1+P2 = P3+Q 
        mult4 = prefacttau12to2p3(u_hat, P1, P2, P3_hat, Q_hat, Is, IV, It, T, m1, -m2, -m3, gL, gR); // no need for a general f here since only one type of traces, include also directly sign = Vec3(1,-1,-1), always for p3 !
        return multSZ*mult0*mult1*mult2*mult3*mult4;} 
    else{return 0.0;}
}


static int integrand2to2p3_changevariable(const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata){
    // set additional arguments
    double *indata,m1,m2,m3,q,T,gL,gR,lambda;
    double x,y,z,phi;
    indata = (double *) userdata;
    m1 = indata[0];
    m2 = indata[1];
    m3 = indata[2];
    q = indata[3];
    T = indata[4];
    gL = indata[5];
    gR = indata[6];
    lambda = indata[7];
    x = xx[0]/(1-xx[0]);
    y = xx[1]*M_PI;
    z = xx[2]*M_PI;
    phi = xx[3]*2*M_PI;
    ff[0] = 2*pow3(M_PI)/pow2(1-xx[0])*integrand2to2p3(x,y,z,phi,m1,m2,m3,q,T,gL,gR,lambda);    //test_fun_old(xx[0],xx[1]);
    return 0;
}


double GammaPp3(double m1, double m2, double m3, double q, double T, double gL, double gR, double lambda, int maxeval){
    int nregions, neval, fail;
    double integral[NCOMP], error[NCOMP], prob[NCOMP];
    double args[8] = {m1,m2,m3,q,T,gL,gR,lambda};
    K_cached(T);
    auto start = chrono::steady_clock::now();
    Cuhre(NDIM, NCOMP, integrand2to2p3_changevariable, args, NVEC, EPSREL, EPSABS, VERBOSE | LAST, MINEVAL, maxeval, KEY, STATEFILE, SPIN, &nregions, &neval, &fail, integral, error, prob);
    double seconds = chrono::duration<double>(chrono::steady_clock::now()-start).count();
    printf("CUHRE p3: nregions=%d neval=%d fail=%d value=% .9e error=%.3e prob=%.3e elapsed=%.3fs eval/s=%.1f epsrel=%.1e epsrel_gsl=%.1e epsabs=%.1e maxeval=%d\n", nregions, neval, fail, integral[0], error[0], prob[0], seconds, seconds > 0.0 ? neval/seconds : 0.0, EPSREL, EPSRELGSL, EPSABS, maxeval);
    return -8.0*pow2(GF)*e2*(1.0+exp(-q/T))/q*integral[0]/pow3(2*M_PI);
}



/* -------- p2 -------- */

double integrand2to2p2(double p2mod, double theta, double theta_hat, double phi_hat, double m1, double m2, double m3, double q, double T, double gL, double gR, double lambda){
    double qvec[3], p2vec[3], normp2q, Q[4], P2[4], P2Q[4], e[3], Q_hat[4], P2_hat[4], unit4vec[4], u_hat[4], Is, IV[4], IV_lab[4], It[4][4], It_lab[4][4], E2_hat, q0_hat, p1_hatmod, Wp1_hat[3], P1[4], P3[4], P1_lab[4], P1P2_lab[4], multSZ, mult0, mult1, mult2, mult3, mult4;
    // define unit4vec
    unit4vec[0] = 1.0;
    unit4vec[1] = 0.0;
    unit4vec[2] = 0.0;
    unit4vec[3] = 0.0;
    // define qvec
    qvec[0] = 0.0;
    qvec[1] = 0.0;
    qvec[2] = q;
    // define p1vec
    p2vec[0] = p2mod*sin(theta);
    p2vec[1] = 0.0;
    p2vec[2] = p2mod*cos(theta);
    //
    vec4Ep(energy(0, q), qvec, Q);
    vec4Ep(energy(m2, p2mod), p2vec, P2);
    // define P2+Q, sum incoming momentum in the lab frame
    combili4vector(P2,Q,1,1,P2Q);
    // define e
    normp2q = sqrt(pow2(qvec[0] + p2vec[0])+pow2(qvec[1] + p2vec[1])+pow2(qvec[2] + p2vec[2]));
    e[0] = (qvec[0] + p2vec[0])/normp2q;
    e[1] = (qvec[1] + p2vec[1])/normp2q;
    e[2] = (qvec[2] + p2vec[2])/normp2q;
    boost(P2Q, Q, Q_hat);
    boost(P2Q, P2, P2_hat);
    boost(P2Q, unit4vec, u_hat);
    E2_hat = P2_hat[0];
    q0_hat = Q_hat[0];
    if(E2_hat + q0_hat - m1 - m3 > 0.0){
        p1_hatmod = rho( E2_hat + q0_hat, m1, m3);
        // define Wp1_hat*P1_hatmod if use convention of Julia notebook
        Wp1_hat[0] = p1_hatmod*(e[2]*sin(theta_hat)*cos(phi_hat) + e[0]*cos(theta_hat));
        Wp1_hat[1] = p1_hatmod*(sin(theta_hat)*sin(phi_hat));
        Wp1_hat[2] = p1_hatmod*(e[2]*cos(theta_hat) - e[0]*sin(theta_hat)*cos(phi_hat)); 
        vec4Ep(energy(m1, p1_hatmod),  Wp1_hat, P1);
        // add minus sign of difference
        Wp1_hat[0] *= -1.0; 
        Wp1_hat[1] *= -1.0; 
        Wp1_hat[2] *= -1.0;
        vec4Ep(energy(m3, p1_hatmod), Wp1_hat, P3);
        
        
        boost_inv(P2Q,P1,P1_lab);
        double cached_K = K_cached(T);
        VertexBasis basis1 = build_vertex_basis(T, P1_lab, P2, lambda, cached_K);
        VertexBasis basis2 = build_vertex_basis(T, P2, P1_lab, lambda, cached_K);
        if (!basis1.valid || !basis2.valid) {
            fprintf(stderr, "reject p2 point: p=%.9e theta=%.9e theta_hat=%.9e phi_hat=%.9e\n", p2mod, theta, theta_hat, phi_hat);
            return 0.0;
        }
        Is = basis1.Is + basis2.Is;
        double IV_lab1[4], IV_lab2[4];
        I_elec_Vec(P1_lab, P2, basis1, IV_lab1);
        I_elec_Vec(P2, P1_lab, basis2, IV_lab2);
        combili4vector(IV_lab1,IV_lab2,1.0,1.0,IV_lab);
        boost(P2Q,IV_lab, IV);
        double It_lab1[4][4], It_lab2[4][4];
        I_elec_Tensor(P1_lab, P2, basis1, IV_lab1, It_lab1);
        I_elec_Tensor(P2, P1_lab, basis2, IV_lab2, It_lab2);
        combilitensor(It_lab1,It_lab2,1.0,1.0,It_lab);
        boost_tensor(P2Q,It_lab, It);

        multSZ = 1.0/pow2(4*M_PI);
        mult0 = 1.0/pow2(2*M_PI);
        mult1 = pow2(p2mod) / (2* energy(m2, p2mod));
        mult2 = sin(theta)*sin(theta_hat);
        mult3 = rho( E2_hat + q0_hat, m1, m3) / (E2_hat + q0_hat);
        mult4 = prefacttau12to2p2(u_hat, P1, P2_hat, P3, Q_hat, Is, IV, It, T, m1, m2, m3, gL,gR); // no need for a general f here since only one type of traces, include also directly sign = Vec3(1,1,1), always for p2 !
        return multSZ*mult0*mult1*mult2*mult3*mult4;} 
    else{return 0.0;}
}


static int integrand2to2p2_changevariable(const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata){
    // set additional arguments 
    double *indata,m1,m2,m3,q,T,gL,gR,lambda;
    double x,y,z,phi;
    indata = (double *) userdata;
    m1 = indata[0];
    m2 = indata[1];
    m3 = indata[2];
    q = indata[3];
    T = indata[4];
    gL = indata[5];
    gR = indata[6];
    lambda = indata[7];
    x = xx[0]/(1-xx[0]);
    y = xx[1]*M_PI;
    z = xx[2]*M_PI;
    phi = xx[3]*2*M_PI;
    ff[0] = 2*pow3(M_PI)/pow2(1-xx[0])*integrand2to2p2(x,y,z,phi,m1,m2,m3,q,T,gL,gR,lambda);    
    return 0;
}

double GammaPp2(double m1, double m2, double m3, double q, double T, double gL, double gR, double lambda, int maxeval){
    int nregions, neval, fail;
    double integral[NCOMP], error[NCOMP], prob[NCOMP];
    double args[8] = {m1,m2,m3,q,T,gL,gR,lambda};
    K_cached(T);
    auto start = chrono::steady_clock::now();
    Cuhre(NDIM, NCOMP, integrand2to2p2_changevariable, args, NVEC, EPSREL, EPSABS, VERBOSE | LAST, MINEVAL, maxeval, KEY, STATEFILE, SPIN, &nregions, &neval, &fail, integral, error, prob);
    double seconds = chrono::duration<double>(chrono::steady_clock::now()-start).count();
    printf("CUHRE p2: nregions=%d neval=%d fail=%d value=% .9e error=%.3e prob=%.3e elapsed=%.3fs eval/s=%.1f epsrel=%.1e epsrel_gsl=%.1e epsabs=%.1e maxeval=%d\n", nregions, neval, fail, integral[0], error[0], prob[0], seconds, seconds > 0.0 ? neval/seconds : 0.0, EPSREL, EPSRELGSL, EPSABS, maxeval);
    return -8.0*pow2(GF)*e2*(1.0+exp(-q/T))/q*integral[0]/pow3(2*M_PI);
}

/* -------- p1 -------- */



double integrand2to2p1(double p1mod, double theta, double theta_hat, double phi_hat, double m1, double m2, double m3, double q, double T, double gL, double gR, double lambda){
    double qvec[3], p1vec[3], normp1q, Q[4], P1[4], P1Q[4], e[3], Q_hat[4], P1_hat[4], unit4vec[4], u_hat[4], Is, IV[4], IV_lab[4], It[4][4], It_lab[4][4], E1_hat, q0_hat, p3_hatmod, Wp3_hat[3], P3[4], P2[4], P2_lab[4], P1P2_lab[4], multSZ, mult0, mult1, mult2, mult3, mult4;
    // define unit4vec
    unit4vec[0] = 1.0;
    unit4vec[1] = 0.0;
    unit4vec[2] = 0.0;
    unit4vec[3] = 0.0;
    // define qvec
    qvec[0] = 0.0;
    qvec[1] = 0.0;
    qvec[2] = q;
    // define p1vec
    p1vec[0] = p1mod*sin(theta);
    p1vec[1] = 0.0;
    p1vec[2] = p1mod*cos(theta);
    //
    vec4Ep(energy(0, q), qvec, Q);
    vec4Ep(energy(m1, p1mod), p1vec, P1);
    // define P1+Q, sum incoming momentum in the lab frame
    combili4vector(P1,Q,1,1,P1Q);
    // define e
    normp1q = sqrt(pow2(qvec[0] + p1vec[0])+pow2(qvec[1] + p1vec[1])+pow2(qvec[2] + p1vec[2]));
    e[0] = (qvec[0] + p1vec[0])/normp1q;
    e[1] = (qvec[1] + p1vec[1])/normp1q;
    e[2] = (qvec[2] + p1vec[2])/normp1q;
    boost(P1Q, Q, Q_hat);
    boost(P1Q, P1, P1_hat);
    boost(P1Q, unit4vec, u_hat);
    E1_hat = P1_hat[0];
    q0_hat = Q_hat[0];
    if(E1_hat + q0_hat - m2 - m3 > 0.0){
        p3_hatmod = rho( E1_hat + q0_hat, m2, m3);
        // define Wp1_hat*P1_hatmod if use convention of Julia notebook
        Wp3_hat[0] = p3_hatmod*(e[2]*sin(theta_hat)*cos(phi_hat) + e[0]*cos(theta_hat));
        Wp3_hat[1] = p3_hatmod*(sin(theta_hat)*sin(phi_hat));
        Wp3_hat[2] = p3_hatmod*(e[2]*cos(theta_hat) - e[0]*sin(theta_hat)*cos(phi_hat)); 
        vec4Ep(energy(m3, p3_hatmod),  Wp3_hat, P3);
        // add minus sign of difference
        Wp3_hat[0] *= -1.0; 
        Wp3_hat[1] *= -1.0; 
        Wp3_hat[2] *= -1.0;
        vec4Ep(energy(m2, p3_hatmod), Wp3_hat, P2);
        

        boost_inv(P1Q,P2,P2_lab);
        double P1m[4], P2_labm[4];
        combili4vector(P1,Q,-1.0,0.0,P1m);
        combili4vector(P2_lab,Q,-1.0,0.0,P2_labm);
        double cached_K = K_cached(T);
        VertexBasis basis1 = build_vertex_basis(T, P1m, P2_labm, lambda, cached_K);
        VertexBasis basis2 = build_vertex_basis(T, P2_labm, P1m, lambda, cached_K);
        if (!basis1.valid || !basis2.valid) {
            fprintf(stderr, "reject p1 point: p=%.9e theta=%.9e theta_hat=%.9e phi_hat=%.9e\n", p1mod, theta, theta_hat, phi_hat);
            return 0.0;
        }
        Is = basis1.Is + basis2.Is;
        double IV_lab1[4], IV_lab2[4];
        I_elec_Vec(P1m, P2_labm, basis1, IV_lab1);
        I_elec_Vec(P2_labm, P1m, basis2, IV_lab2);
        combili4vector(IV_lab1,IV_lab2,1.0,1.0,IV_lab);
        boost(P1Q,IV_lab, IV);
        double It_lab1[4][4], It_lab2[4][4];
        I_elec_Tensor(P1m, P2_labm, basis1, IV_lab1, It_lab1);
        I_elec_Tensor(P2_labm, P1m, basis2, IV_lab2, It_lab2);
        combilitensor(It_lab1,It_lab2,1.0,1.0,It_lab);
        boost_tensor(P1Q,It_lab, It);

        /* OLD: WRONG SIGNS
        Is = I_elec(T, P1, P2_lab, lambda);
        I_elec_Vec(T, P1, P2_lab, Is, lambda, IV_lab);
        boost(P1Q,IV_lab, IV);
        I_elec_Tensor(T, P1, P2_lab, Is, IV_lab, lambda, It_lab);
        boost_tensor(P1Q,It_lab, It); // boost tensor in correct frame...
        */
      
        
        multSZ = 1.0/pow2(4*M_PI);
        mult0 = 1.0/pow2(2*M_PI);
        mult1 = pow2(p1mod) / (2* energy(m1, p1mod));
        mult2 = sin(theta)*sin(theta_hat);
        mult3 = rho( E1_hat + q0_hat, m2, m3) / (E1_hat + q0_hat);
        mult4 = prefacttau12to2p1(u_hat, P1_hat, P2, P3, Q_hat, Is, IV, It, T, -m1, -m2, m3, gL,gR); // no need for a general f here since only one type of traces, include also directly sign = Vec3(-1,-1,1), always for p1 !
        return multSZ*mult0*mult1*mult2*mult3*mult4;} 
    else{return 0.0;}
}


static int integrand2to2p1_changevariable(const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata){
    // set additional arguments 
    double *indata,m1,m2,m3,q,T,gL,gR,lambda;
    double x,y,z,phi;
    indata = (double *) userdata;
    m1 = indata[0];
    m2 = indata[1];
    m3 = indata[2];
    q = indata[3];
    T = indata[4];
    gL = indata[5];
    gR = indata[6];
    lambda = indata[7];
    x = xx[0]/(1-xx[0]);
    y = xx[1]*M_PI;
    z = xx[2]*M_PI;
    phi = xx[3]*2*M_PI;
    ff[0] = 2*pow3(M_PI)/pow2(1-xx[0])*integrand2to2p1(x,y,z,phi,m1,m2,m3,q,T,gL,gR,lambda);    
    return 0;
}

double GammaPp1(double m1, double m2, double m3, double q, double T, double gL, double gR, double lambda, int maxeval){
    int nregions, neval, fail;
    double integral[NCOMP], error[NCOMP], prob[NCOMP];
    double args[8] = {m1,m2,m3,q,T,gL,gR,lambda};
    K_cached(T);
    auto start = chrono::steady_clock::now();
    Cuhre(NDIM, NCOMP, integrand2to2p1_changevariable, args, NVEC, EPSREL, EPSABS, VERBOSE | LAST, MINEVAL, maxeval, KEY, STATEFILE, SPIN, &nregions, &neval, &fail, integral, error, prob);
    double seconds = chrono::duration<double>(chrono::steady_clock::now()-start).count();
    printf("CUHRE p1: nregions=%d neval=%d fail=%d value=% .9e error=%.3e prob=%.3e elapsed=%.3fs eval/s=%.1f epsrel=%.1e epsrel_gsl=%.1e epsabs=%.1e maxeval=%d\n", nregions, neval, fail, integral[0], error[0], prob[0], seconds, seconds > 0.0 ? neval/seconds : 0.0, EPSREL, EPSRELGSL, EPSABS, maxeval);
    return -8.0*pow2(GF)*e2*(1.0+exp(-q/T))/q*integral[0]/pow3(2*M_PI);
}






typedef double (*FixedPointIntegrand)(double, double, double, double, double, double,
                                     double, double, double, double, double, double);

static void validate_fixed_point(const char *label, FixedPointIntegrand integrand,
                                 double q, double T, double gL, double gR, double lambda){
    inner_stats = {0, 0, 0, 0.0};
    auto start = chrono::steady_clock::now();
    double value = integrand(5.0, 1.0, 1.0, 1.0, m_e, m_e, 0.0,
                             q, T, gL, gR, lambda);
    double seconds = chrono::duration<double>(chrono::steady_clock::now()-start).count();
    printf("FIXED %s: value=% .17e elapsed=%.6fs qag_calls=%lu failures=%lu max_subdivisions=%zu max_abserr=%.3e\n",
           label, value, seconds, inner_stats.calls, inner_stats.failures,
           inner_stats.max_subdivisions, inner_stats.max_abserr);
}

static void validate_fixed_points(double q, double T, double gL, double gR, double lambda){
    K_cached(T); // evaluated once, before any outer cubature or per-orientation work
    validate_fixed_point("p3", integrand2to2p3, q, T, gL, gR, lambda);
    validate_fixed_point("p2", integrand2to2p2, q, T, gL, gR, lambda);
    validate_fixed_point("p1", integrand2to2p1, q, T, gL, gR, lambda);
}

static void run_channels(double q, double T, double gL, double gR,
                         double lambda, int maxeval){
    printf("\n===== CONVERGENCE RUN maxeval=%d =====\n", maxeval);
    double Gamma3 = GammaPp3(m_e, m_e, 0.0, q, T, gL, gR, lambda, maxeval);
    double Gamma2 = GammaPp2(m_e, m_e, 0.0, q, T, gL, gR, lambda, maxeval);
    double Gamma1 = GammaPp1(m_e, m_e, 0.0, q, T, gL, gR, lambda, maxeval);
    printf("RATES maxeval=%d p3=% .9e p2=% .9e p1=% .9e total=% .9e\n",
           maxeval, Gamma3, Gamma2, Gamma1, Gamma1 + Gamma2 + Gamma3);
}

/* ------- Main function ------ */


// ATTENTION FOR NOW I TURN OFF ONE COMPONENT OF TENSOR TO BE ABLE TO RUN !!!!!!!!!

int main(int argc, char** argv){
    double T, Q[4], U[4], P[4], P2, lambda, I_s, It_result[4][4], gL, gR, q;
    gL = 1.0/2.0+xw;
    gR = xw;
    T = 3.0;
    lambda = 0.005; 
    q = 3.15*T;
    gsl_set_error_handler_off();
    validate_fixed_points(q, T, gL, gR, lambda);
    if (argc == 2 && strcmp(argv[1], "--validate-only") == 0)
        return 0;
    if (argc == 2 && strcmp(argv[1], "--convergence") == 0) {
        const int caps[] = {200000, 500000, 1000000};
        for (size_t i = 0; i < sizeof(caps)/sizeof(caps[0]); ++i)
            run_channels(q, T, gL, gR, lambda, caps[i]);
        return 0;
    }
    if (argc != 1) {
        fprintf(stderr, "usage: %s [--validate-only|--convergence]\n", argv[0]);
        return 2;
    }
    run_channels(q, T, gL, gR, lambda, DEFAULT_MAXEVAL);
    cout << "END" << endl;
    
     /*
    FILE *fptr;
    fptr = fopen("/home/ygeoris/c++code/Dataset/Typeavertex_5e-2_scalar_decomposed_electrononshell_withFeynmantrick.csv","w+"); //Typeavertex //Typeaphotoncut
    fprintf(fptr,"EPSRELGSL = %e\n", EPSRELGSL);
    fprintf(fptr,"T q lambda Gamma Gamma1 Gamma2 Gamma3\n");
    double size = 25.0;
    double Gamma;
    for(int i=0;i<size;i++){
        T = 0.5+0.1*i;
        q = 3.15*T;
        lambda = 0.005;//5; //T*pow(10,i/size*5-4);
        double Gamma3 = GammaPp3(m_e, m_e, 0.0, q, T, gL, gR, lambda);
        double Gamma2 = GammaPp2(m_e, m_e, 0.0, q, T, gL, gR, lambda);
        double Gamma1 = GammaPp1(m_e, m_e, 0.0, q, T, gL, gR, lambda);
        Gamma = Gamma1+Gamma2+Gamma3;
    	cout << i << " " << Gamma << endl;
    	fprintf(fptr,"%e %e %e %e %e %e %e \n",T, q, lambda, Gamma, Gamma1, Gamma2, Gamma3);}
    fclose(fptr);
    // */
    return 0;
}


