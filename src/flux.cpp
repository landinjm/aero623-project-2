#include "flux.hpp"
#include <cmath>
#include "parameters.hpp"
#include <utility>

std::pair<Tensor<1,4,double>, double>
Flux_HLLE(Tensor<1,4,double> UL,
          Tensor<1,4,double> UR,
          Tensor<1,2,double> n) {
    double gamma = Parameters<double>::gamma;
    
    //unpack the state
    double rhoL = UL[0];
    double uL = UL[1] / rhoL;
    double vL = UL[2] / rhoL;

    double rhoR = UR[0];
    double uR = UR[1] / rhoR;
    double vR = UR[2] / rhoR;

    //calculate wave speeds
    double sLmin = min(0, uL - cL);
    double sRmin = min(0, uR - cR);
    double sLmax = max(0, uL + cL);
    double sRmax = max(0, uR + cR);

    double sLmag = abs(unL) + cL;
    double sRmag = abs(unR) + cR;

    double smag = max(sLmag, sRmag);
    double sLRmin = min(sLmin, sRmin);
    double sLRmax = max(sLmax, sRmax);

    //calculate the left and right fluxes
    Tensor<1,4,double> FL = Cell_Flux(UL, n, gamma);
    Tensor<1,4,double> FR = Cell_Flux(UR, n, gamma);;
    Tensor<1,4,double> F;


    //TODO include tensor operators
    F = 0.5 * (FL + FR) - 0.5 * (sLRmax + sLRmin)/(slRmax - sLRmin)*(FR - FL) + sLRmax*sLRmin/(sLRmax - sLRmin)*(UR-UL);

    //return the numerical flux and wave speed
    std::pair<Tensor<1,4,double>, double> exports = {F, smag};
    return exports;
    
}//end Flux_HLLE

Tensor<1,4,double> Cell_Flux(Tensor<1,4,double> U, Tensor<1,2,double> n) {
    Tensor<1,4,double> F;
    double gamma = Parameters<double>::gamma;

    //unwrap the state
    double rho = U[0];
    double u = U[1] / rho;
    double v = U[2] / rho;

    //define un, q, p, rH, c
    double un = u * n[0] + v*n[1];
    double q = sqrt(std::pow(U[1],2) + std::pow(U[2],2)) / rho;
    double p = (gamma - 1) * (U[3] - 0.5*rho*std::pow(q,2));
    double rH = U[3] + p;
    double c = sqrt(gamma*p/rho);

    //compute the flux
    F[0] = rho * un;
    F[1] = U[1] * un + p * n[0];
    F[2] = U[2] * un + p * n[1];
    F[3] = rH*un;

    return F;
}//end Cell_Flux
