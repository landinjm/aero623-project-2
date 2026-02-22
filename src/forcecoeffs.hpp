#pragma once

#include <cmath>
#include <tensor.hpp>
#include <triangulation.hpp>

template<unsigned int dim, typename RealType>
class CalcForceCoeffs
{
public:
    double calcCp(const Tensor<1, 2 + dim, RealType>& interior_state,
                double n_x, double n_y){
        // Define useful variables
        double gamma = 1.4;

        // Grab values
        const auto rho = interior_state[0];
        const auto momentum_x = interior_state[1];
        const auto momentum_y = interior_state[2];
        const auto energy = interior_state[3];

        // Calculate useful variables
        double p0 = 1 / gamma; // rho0*a0^2/gamma but rho0 and a0 are 1 b/c nondimensional
        double pout = 0.7 * p0;
        double Mout2 = 2.0 / (gamma - 1.0) * (std::pow(p0 / pout, (gamma-1.0) / gamma) - 1);
        double qout = 0.5 * gamma * pout * Mout2;
        const auto dot = momentum_x * n_x + momentum_y * n_y;
        const auto u_b = (momentum_x - dot * n_x) / rho;
        const auto v_b = (momentum_y - dot * n_y) / rho;
        const auto q2 = u_b * u_b + v_b * v_b;
        const auto p = (Parameters<RealType>::gamma - 1.0) * (energy - 0.5 * rho * q2);

        // Calculate pressure coefficient
        const auto cp = (p - pout) / qout;

        // Return pressure coefficient
        return cp;
    }

    // also make this work with tensors
    std::vector<double> calcForceCoeffs(const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
                                        const ElementData<dim, RealType>& element_scratch){
        // Define useful variables
        double c = 18.804; // should be in mm, but I think x & y are also in mm
        double c_x = 0.0;
        double c_y = 0.0;
        double cp = 0.0;

        // Loop over boundary faces and calculate force coefficients
        for (unsigned int i = 0; i < boundary_face_scratch.size(); ++i){
            const auto e = boundary_face_scratch.elem[i];
            const auto n_x = boundary_face_scratch.normal_x[i];
            const auto n_y = boundary_face_scratch.normal_y[i];
            const auto area = boundary_face_scratch.face_area[i];
            const auto boundary_id = boundary_face_scratch.boundary_id[i];

            const Tensor<1, 4, RealType> u = { element_scratch.density[e],
                                         element_scratch.momentum_x[e],
                                         element_scratch.momentum_y[e],
                                         element_scratch.energy[e] };
            
            switch (boundary_id){
                case 6: // BladeTop
                    cp = calcCp(u, n_x, n_y);
                    c_x += cp * n_x * area / c;
                    c_y += cp * n_y * area / c;
                    break;
                case 7: // BladeBottom
                    cp = calcCp(u, n_x, n_y);
                    c_x += cp * n_x * area / c;
                    c_y += cp * n_y * area / c;
                    break;
                default:
                    break;
            }
        }

        // Return force coefficients
        return {c_x, c_y};
    }
};
