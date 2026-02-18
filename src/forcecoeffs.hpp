#pragma once

#include <cmath>
#include <vector>
#include <tensor.hpp>

template<unsigned int dim, typename RealType>
class CalcForceCoeffs
{
public:
    double calcCp(const Tensor<1, 2 + dim, RealType>& interior_state,
                const Tensor<1, dim, RealType>& normal){
        // Define useful variables
        double gamma = 1.4;

        // Grab values
        const auto rho = interior_state[0];
        const auto momentum_x = interior_state[1];
        const auto momentum_y = interior_state[2];
        const auto energy = interior_state[3];

        const auto n_x = normal[0];
        const auto n_y = normal[1];

        // Calculate useful variables
        double p0 = 1 / gamma; // rho0*a0^2/gamma but rho0 and a0 are 1 b/c nondimensional
        double pout = 0.7 * p0;
        double Mout2 = 2.0 / (gamma - 1.0) * (std::pow(p0 / pout, (gamma-1.0) / gamma) - 1);
        double qout = 0.5 * gamma * pout * Mout2;
        const auto q2 = (momentum_x * momentum_x + momentum_y * momentum_y) / rho;
        const auto p = (Parameters<RealType>::gamma - 1.0) * (energy - 0.5 * rho * q2);

        // Calculate pressure coefficient
        const auto cp = (p - Parameters<RealType>::pout) / Parameters<RealType>::qout;

        // Return pressure coefficient
        return cp;
    }

    // also make this work with tensors
    std::vector<double> calcForceCoeffs(const Tensor<1, 2 + dim, RealType>& interior_state,
                const Tensor<1, dim, RealType>& normal){
        // Define useful variables
        double c = 18.804; // should be in mm, but I think x & y are also in mm
        double c_x = 0.0;
        double c_y = 0.0;

        // Get pressure coefficient
        const auto cp = calcCp(interior_state, normal);

        // Get normal components
        const auto n_x = normal[0];
        const auto n_y = normal[1];

        // Calculate force coefficients
        for (unsigned int i = 0; i < dim; ++i) {
            c_x += cp * n_x / Parameters<RealType>::c;
            c_y += cp * n_y / Parameters<RealType>::c;
        }

        // Return force coefficients
        return {c_x, c_y};
    }
};