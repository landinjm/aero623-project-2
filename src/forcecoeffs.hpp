#pragma once

#include <cmath>
#include <vector>
//#include <tensor.hpp>

class CalcForceCoeffs
{
public:
    /* calcCpTensor(const Tensor<1, 2 + dim, RealType>& interior_state,
                const Tensor<1, dim, RealType>& normal)
        static_assert(dim == 2, "Only 2D is supported");

        // Grab values
        const auto rho = interior_state[0];
        const auto momentum_x = interior_state[1];
        const auto momentum_y = interior_state[2];
        const auto energy = interior_state[3];

        const auto n_x = normal[0];
        const auto n_y = normal[1];) */

    double calcCp(double rho, double momentum_x, double momentum_y, double energy){
        // Define useful variables
        double gamma = 1.4;

        // Calculate useful variables
        double p0 = 1 / gamma; // rho0*a0^2/gamma but rho0 and a0 are 1 b/c nondimensional
        double pout = 0.7 * p0;
        double Mout2 = 2.0 / (gamma - 1.0) * (std::pow(p0 / pout, (gamma-1.0) / gamma) - 1);
        double qout = 0.5 * gamma * pout * Mout2;
        double q = std::sqrt(momentum_x * momentum_x + momentum_y * momentum_y) / rho;
        double p = (gamma - 1.0) * (energy - 0.5 * rho * q * q);

        // Calculate pressure coefficient
        double cp = (p - pout) / qout;

        // Return pressure coefficient
        return cp;
    }

    // also make this work with tensors
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