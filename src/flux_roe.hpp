#pragma once

#ifndef FLUX_ROE_HPP
#define FLUX_ROE_HPP

#include "tensor.hpp"
#include <utility> // Required for std::pair

namespace SWE {

/**
 * @brief Computes physical flux F*nx + G*ny
 */
Tensor<1, 4, double> Cell_Flux(Tensor<1, 4, double> U, Tensor<1, 2, double> n);

/**
 * @brief Roe Flux for 2D Euler Equations
 * Returns {Numerical Flux, Max Wave Speed}
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

} // namespace SWE

#endif // FLUX_ROE_HPP