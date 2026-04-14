#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <tensor.hpp>

/**
 * @brief Simple structure for the parameters in the simulation.
 */
template<typename RealType>
struct Parameters
{
  /**
   * @brief Ratio of specific heats.
   */
  static constexpr RealType gamma = RealType{ 1.4 };

  /**
   * @brief Inlet stagnation density.
   */
  static constexpr RealType rho_0 = RealType{ 1.0 };

  /**
   * @brief Inlet stagnation speed of sound.
   */
  static constexpr RealType a_0 = RealType{ 1.0 };

  /**
   * @brief Inlet stagnation pressure.
   */
  static constexpr RealType p_0 = rho_0 * a_0 * a_0 / gamma;

  /**
   * @brief Inlet stagnation temperature multiplied by the gas constant.
   */
  static constexpr RealType T_0_and_R = p_0 / rho_0;

  /**
   * @brief Inlet angle of attack (degrees).
   */
  static constexpr RealType alpha = RealType{ 50.0 };

  /**
   * @brief x-component of inlet flow normal.
   */
  KOKKOS_INLINE_FUNCTION
  static RealType n_x_0() { return Kokkos::cos(alpha / 180.0 * M_PI); }

  /**
   * @brief y-component of inlet flow normal.
   */
  KOKKOS_INLINE_FUNCTION
  static RealType n_y_0() { return Kokkos::sin(alpha / 180.0 * M_PI); }

  /**
   * @brief Inlet flow normal.
   */
  template<unsigned int dim>
  KOKKOS_INLINE_FUNCTION static Tensor<1, dim, RealType> n_0()
  {
    Tensor<1, dim, RealType> n;
    if constexpr (dim == 2) {
      n(0) = n_x_0();
      n(1) = n_y_0();
    }

    return n;
  }

  /**
   * @brief Outflow static pressure.
   */
  static constexpr RealType p_out = RealType{ 0.7 } * p_0;

  /**
   * @brief Maximum CFL number to use when calculating optimal timestep.
   */
  static constexpr RealType cfl_max = RealType{ 0.5 };

  /**
   * @brief Convergence tolerance.
   */
  static constexpr RealType convergence_tol = 1e-5;

  /**
   * @brief Unsteady flow parameters.
   */
  static constexpr RealType f_wake = RealType{ 0.1 };
  static constexpr RealType delta = RealType{ 0.1 };
  static constexpr RealType Delta_y = RealType{ 18.0 };
};
