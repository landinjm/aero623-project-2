#pragma once

#include <libassert/assert.hpp>
#include <parameters.hpp>
#include <tensor.hpp>
#include <utility>

/**
 * @brief Compute the pressure.
 *
 * @param[in] state State vector.
 * @param[out] Pressure.
 */
template<typename RealType>
inline RealType
pressure(Tensor<1, 4, RealType> state)
{
  DEBUG_ASSERT(state[0] > 0, "Density must be positive");

  const auto u = state[1] / state[0];
  const auto v = state[2] / state[0];

  return (Parameters<RealType>::gamma - 1.0) *
         (state[3] - 0.5 * state[0] * (u * u + v * v));
}

/**
 * @brief Compute the speed of sound.
 *
 * @param[in] p Pressure.
 * @param[in] rho Density.
 * @param[out] Speed of sound.
 */
template<typename RealType>
inline RealType
speed_of_sound(RealType p, RealType rho)
{
  DEBUG_ASSERT(rho > 0, "Density must be positive");
  DEBUG_ASSERT(p >= 0, "Pressure must be positive");
  return std::sqrt(Parameters<RealType>::gamma * p / rho);
}

/**
 * @brief Compute the speed of sound.
 *
 * @param[in] state State vector.
 * @param[out] Speed of sound.
 */
template<typename RealType>
inline RealType
speed_of_sound(Tensor<1, 4, RealType> state)
{
  DEBUG_ASSERT(state[0] > 0, "Density must be positive");
  const auto p = pressure(state);
  DEBUG_ASSERT(p >= 0, "Pressure must be positive");
  return std::sqrt(Parameters<RealType>::gamma * p / state[0]);
}

/**
 * @brief Compute the max wave speed.
 *
 * @param[in] u Horizontal velocity.
 * @param[in] v Vertical velocity.
 * @param[in] n_x Normal in x direction.
 * @param[in] n_y Normal in y direction.
 * @param[in] p Pressure.
 * @param[in] rho Density.
 * @param[out] Max wave speed.
 *
 * The max wavespeed is just |u| + c, where |u| is the magnitude of the normal
 * velocity and c is the speed of sound.
 */
template<typename RealType>
inline RealType
max_wavespeed(RealType u,
              RealType v,
              RealType n_x,
              RealType n_y,
              RealType p,
              RealType rho)
{
  DEBUG_ASSERT(rho > 0, "Density must be positive");
  DEBUG_ASSERT(p >= 0, "Pressure must be positive");
  return std::abs(u * n_x + v * n_y) + speed_of_sound(p, rho);
}

/**
 * @brief Compute the enthalpy.
 *
 * @param[in] E Energy density.
 * @param[in] p Pressure.
 * @param[in] rho Density.
 * @param[out] Enthalpy
 */
template<typename RealType>
inline RealType
enthalpy(RealType E, RealType p, RealType rho)
{
  DEBUG_ASSERT(rho > 0, "Density must be positive");
  DEBUG_ASSERT(p >= 0, "Pressure must be positive");
  return (E + p) / rho;
}

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
 * Returns {Numerical Flux, Max Wave Speed}
 */
std::pair<Tensor<1, 4, double>, double>
flux_hlle(Tensor<1, 4, double> UL,
          Tensor<1, 4, double> UR,
          Tensor<1, 2, double> n);

/**
 * @brief Compute the flux vector given the full state and normal vectors.
 */
template<unsigned int dim, typename RealType>
inline Tensor<1, 2 + dim, RealType>
euler_flux(const Tensor<1, 2 + dim, RealType>& state,
           const Tensor<1, dim, RealType>& normal)
{
  static_assert(dim == 2, "Only 2D is supported");

  // Grab values
  const auto rho = state[0];
  DEBUG_ASSERT(rho > 0, "Density must be positive");
  const auto u = state[1] / rho;
  const auto v = state[2] / rho;
  const auto p = pressure(state);
  DEBUG_ASSERT(p >= 0, "Pressure must be positive");

  // Compute the normal velocity
  const auto v_n = u * normal[0] + v * normal[1];

  // Compute the enthalpy per volume
  const auto H_V = state[3] + p;

  return Tensor<1, 2 + dim, RealType>{ rho * v_n,
                                       state[1] * v_n + p * normal[0],
                                       state[2] * v_n + p * normal[1],
                                       H_V * v_n };
}
