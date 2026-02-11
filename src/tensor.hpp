#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <initializer_list>

static constexpr unsigned int
ipow(unsigned int base, unsigned int exp)
{
  unsigned int result = 1;
  for (unsigned int i = 0; i < exp; ++i)
    result *= base;
  return result;
}

template<unsigned int rank, unsigned int dim, typename RealType>
  requires std::floating_point<RealType>
class Tensor
{
public:
  static constexpr unsigned int size = ipow(dim, rank);

  constexpr Tensor() = default;

  // Variadic constructor (works for ANY rank)
  template<typename... Args>
  constexpr Tensor(Args... args)
    requires(sizeof...(Args) == size)
    : data{ static_cast<RealType>(args)... }
  {
  }

  Tensor(std::initializer_list<RealType> init)
  {
    assert(init.size() == size);
    std::copy(init.begin(), init.end(), data.begin());
  }

  // Element access
  constexpr RealType& operator[](unsigned int i)
  {
    assert(i < size);
    return data[i];
  }

  constexpr const RealType& operator[](unsigned int i) const
  {
    assert(i < size);
    return data[i];
  }

  // Addition
  constexpr Tensor operator+(const Tensor& other) const
  {
    Tensor result;

    for (unsigned int i = 0; i < size; ++i)
      result.data[i] = data[i] + other.data[i];

    return result;
  }

  // Subtraction
  constexpr Tensor operator-(const Tensor& other) const
  {
    Tensor result;

    for (unsigned int i = 0; i < size; ++i)
      result.data[i] = data[i] - other.data[i];

    return result;
  }

  // Scalar multiplication
  constexpr Tensor operator*(RealType scalar) const
  {
    Tensor result;

    for (unsigned int i = 0; i < size; ++i)
      result.data[i] = data[i] * scalar;

    return result;
  }

  friend constexpr Tensor operator*(RealType scalar, const Tensor& T)
  {
    return T * scalar;
  }

  // Scalar division
  constexpr Tensor operator/(RealType scalar) const
  {
    Tensor result;

    for (unsigned int i = 0; i < size; ++i)
      result.data[i] = data[i] / scalar;

    return result;
  }

  // Exact equality
  constexpr bool operator==(const Tensor& other) const
  {
    for (unsigned int i = 0; i < size; ++i)
      if (data[i] != other.data[i])
        return false;

    return true;
  }

  constexpr bool operator!=(const Tensor& other) const
  {
    return !(*this == other);
  }

private:
  std::array<RealType, size> data{};
};
