#pragma once

#include <cmath>

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
  static constexpr RealType alpha = RealType{ 5.0 };

  /**
   * @brief x-component of inlet flow normal.
   */
  static constexpr RealType n_x_0 = std::cos(alpha / 180.0 * M_PI);

  /**
   * @brief y-component of inlet flow normal.
   */
  static constexpr RealType n_y_0 = std::sin(alpha / 180.0 * M_PI);

  /**
   * @brief Outflow static pressure.
   */
  static constexpr RealType p_out = RealType{ 0.7 } * p_0;

  /**
   * @brief Maximum CFL number to use when calculating optimal timestep.
   */
  static constexpr RealType cfl_max = RealType{ 0.9 };

  /**
   * @brief Unsteady flow parameters.
   */
  static constexpr RealType f_wake = RealType{ 0.1 };
  static constexpr RealType delta = RealType{ 0.1 };
  static constexpr RealType Delta_y = RealType{ 18.0 };
};
