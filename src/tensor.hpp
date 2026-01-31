#include <concepts>
#include <limits>
#include <ostream>
#include <type_traits>

/**
 * @brief Forward declaration of tensor class
 */
template<unsigned int rank, unsigned int dim, typename RealType>
class Tensor;

/**
 * @brief 0th-rank specialization of tensor class
 */
template<unsigned int dim, typename RealType>
  requires std::floating_point<RealType>
class Tensor<0, dim, RealType>
{
public:
  static_assert(dim > 0, "Tensors must have a dimension greater than one.");
  static constexpr unsigned int dimension = dim;
  static constexpr unsigned int rank = 0;

  constexpr Tensor()
  {
#ifdef DEBUG
    value = std::numeric_limits<RealType>::quiet_NaN();
#else
    value = RealType{ 0 };
#endif
  };

  constexpr explicit Tensor(RealType val) noexcept
    : value(val) {};

  constexpr void clear() { value = RealType{ 0 }; };

private:
  RealType value;
};
