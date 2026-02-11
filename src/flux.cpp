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

    //define un, q, p, rH, c
    double unL = uL * n[0] + vL*n[1];
    double qL = sqrt(std::pow(UL[1],2) + std::pow(UL[2],2)) / rhoL;
    double pL = (gamma - 1) * (UL[3] - 0.5*rhoL*std::pow(qL,2));
    double rHL = UL[3] + pL;
    double cL = sqrt(gamma*pL/rhoL);

    double unR = uR * n[0] + vR*n[1];
    double qR = sqrt(std::pow(UR[1],2) + std::pow(UR[2],2)) / rhoR;
    double pR = (gamma - 1) * (UR[3] - 0.5*rhoR*std::pow(qR,2));
    double rHR = UR[3] + pR;
    double cR = sqrt(gamma*pR/rhoR);

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

    //calculate he left and right fluxes
    Tensor<1,4,double> FL;
    Tensor<1,4,double> FR;
    Tensor<1,4,double> F;

    FL[0] = rhoL * unL;
    FL[1] = UL[1] * unL + pL * n[0];
    FL[2] = UL[2] * unL + pL * n[1];
    FL[3] = rHL*unL;

    FR[0] = rhoR * unR;
    FR[1] = UR[1] * unR + pR * n[0];
    FR[2] = UR[2] * unR + pR * n[1];
    FR[3] = rHR*unR;

    //TODO include tensor operators
    F = 0.5 * (FL + FR) - 0.5 * (sLRmax + sLRmin)/(slRmax - sLRmin)*(FR - FL) + sLRmax*sLRmin/(sLRmax - sLRmin)*(UR-UL);

    //return the numerical flux and wave speed 
    return {F, smag};

}