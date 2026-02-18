#pragma once

#include <cmath>
#include <vector>

class CalcForceCoeffs
{
public:
    double calcCp(double density, double momentum_x, double momentum_y, double energy){
        // Define useful variables
        double gamma = 1.4;

        // Calculate useful variables
        double p0 = 1 / gamma; // rho0*a0^2/gamma but rho0 and a0 are 1 b/c nondimensional
        double pout = 0.7 * p0;
        double Mout2 = 2.0 / (gamma - 1.0) * (std::pow(p0 / pout, (gamma-1.0) / gamma) - 1);
        double qout = 0.5 * gamma * pout * Mout2;
        double q = std::sqrt(momentum_x * momentum_x + momentum_y * momentum_y) / density;
        double p = (gamma - 1.0) * (energy - 0.5 * density * q * q);

        // Calculate pressure coefficient
        double cp = (p - pout) / qout;

        // Return pressure coefficient
        return cp;
    }
    std::vector<double> calcForceCoeffs(double density, double momentum_x, double momentum_y, double energy,
        double n_x, double n_y){
        // Define useful variables
        double c = 18.804; // should be in mm, but I think x & y are also in mm

        // Get pressure coefficient
        double cp = calcCp(density, momentum_x, momentum_y, energy);

        // Calculate force coefficients
        double cx = cp*n_x/c; // normal x component
        double cy = cp*n_y/c; // normal y component

        // Return force coefficients
        return {cx, cy};
    }
};