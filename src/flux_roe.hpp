#pragma once

#include <tensor.hpp>
#include <utility>
#include <vector>

namespace SWE {

/**
 * @brief Computes physical flux F*nx + G*ny
 */
Tensor<1, 4, double>
Cell_Flux(Tensor<1, 4, double> U, Tensor<1, 2, double> n);

/**
 * @brief Roe Flux for 2D Euler Equations
 * Matches the HLLE structure: returns {Flux, MaxWaveSpeed}
 */
std::pair<Tensor<1, 4, double>, double>
flux_roe(Tensor<1, 4, double> UL,
         Tensor<1, 4, double> UR,
         Tensor<1, 2, double> n);

/**
 * @brief HLLE Flux for 2D Euler Equations
 */
std::pair<Tensor<1, 4, double>, double>
Flux_HLLE(Tensor<1, 4, double> UL,
          Tensor<1, 4, double> UR,
          Tensor<1, 2, double> n);
}
