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
   * @brief Inlet angle of attack (degrees).
   */
  static constexpr RealType alpha = RealType{ 50.0 };

  /**
   * @brief Outflow static pressure.
   * TODO: What is the value for this?
   */
  RealType p_out = RealType{ 0.0 };
};