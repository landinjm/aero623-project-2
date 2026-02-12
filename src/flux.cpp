#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <utility>
#include "flux.hpp"
#include "parameters.hpp"

std::pair<Tensor<1,4,double>, double>
Flux_HLLE(Tensor<1,4,double> UL,
          Tensor<1,4,double> UR,
          Tensor<1,2,double> n) {
    double gamma = Parameters<double>::gamma;

    //unpack the state
    double rhoL = UL[0];
    double uL = UL[1] / rhoL;
    double vL = UL[2] / rhoL;
    double unL = uL * n[0] + vL*n[1];
    double qL = std::sqrt(std::pow(UL[1],2) + std::pow(UL[2],2)) / rhoL;
    double pL = (gamma - 1) * (UL[3] - 0.5*rhoL*std::pow(qL,2));
    assert(pL >= 0 && rhoL >= 0);
    double cL = std::sqrt(gamma*pL/rhoL);

    double rhoR = UR[0];
    double uR = UR[1] / rhoR;
    double vR = UR[2] / rhoR;
    double unR = uR * n[0] + vR*n[1];
    double qR = std::sqrt(std::pow(UR[1],2) + std::pow(UR[2],2)) / rhoR;
    double pR = (gamma - 1) * (UR[3] - 0.5*rhoR*std::pow(qR,2));
    assert(pR >= 0 && rhoR >= 0);
    double cR = std::sqrt(gamma*pR/rhoR);

    //calculate wave speeds
    double sLmin = std::min(0.0, unL - cL);
    double sRmin = std::min(0.0, unR - cR);
    double sLmax = std::max(0.0, unL + cL);
    double sRmax = std::max(0.0, unR + cR);

    double sLmag = std::abs(unL) + cL;
    double sRmag = std::abs(unR) + cR;

    double smag = std::max(sLmag, sRmag);
    double sLRmin = std::min(sLmin, sRmin);
    double sLRmax = std::max(sLmax, sRmax);

    //calculate he left and right fluxes
    Tensor<1,4,double> FL = Cell_Flux(UL, n);
    Tensor<1,4,double> FR = Cell_Flux(UR, n);;
    Tensor<1,4,double> F;

    F = 0.5 * (FL + FR) - 0.5 * (sLRmax + sLRmin)/(sLRmax - sLRmin)*(FR - FL) + sLRmax*sLRmin/(sLRmax - sLRmin)*(UR-UL);

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
    double q = std::sqrt(std::pow(U[1],2) + std::pow(U[2],2)) / rho;
    double p = (gamma - 1) * (U[3] - 0.5*rho*std::pow(q,2));
    double rH = U[3] + p;
    double c = std::sqrt(gamma*p/rho);

    assert(p >= 0 && rho >= 0);

    //compute the flux
    F[0] = rho * un;
    F[1] = U[1] * un + p * n[0];
    F[2] = U[2] * un + p * n[1];
    F[3] = rH*un;

    return F;

}//end Cell_Flux
