#pragma once

#include <Kokkos_Core.hpp>
#include <config.hpp>

/**
 * @brief A fixed-size tensor.
 */
template<unsigned int rank, unsigned int dim, typename RealType>
class Tensor
{
public:
  static constexpr int n_components = []() constexpr {
    unsigned int n = 1;
    for (unsigned int i = 0; i < rank; ++i) {
      n *= dim;
    }
    return n;
  }();

  KOKKOS_INLINE_FUNCTION Tensor() { clear(); }

  KOKKOS_INLINE_FUNCTION
  explicit Tensor(const RealType val)
  {
    for (unsigned int i = 0; i < n_components; ++i) {
      data_[i] = val;
    }
  }

  KOKKOS_INLINE_FUNCTION
  Tensor(std::initializer_list<RealType> init)
  {
    ASSERT(init.size() == n_components, "Initializer list size mismatch");
    unsigned int i = 0;
    for (const RealType& v : init) {
      data_[i++] = v;
    }
  }

  template<unsigned int R = rank, typename = std::enable_if_t<R == 1>>
  KOKKOS_INLINE_FUNCTION RealType& operator()(unsigned int i)
  {
    ASSERT(i >= 0 && i < dim, "Index out of bounds");
    return data_[i];
  }

  template<unsigned int R = rank, typename = std::enable_if_t<R == 1>>
  KOKKOS_INLINE_FUNCTION const RealType& operator()(unsigned int i) const
  {
    ASSERT(i >= 0 && i < dim, "Index out of bounds");
    return data_[i];
  }

  template<unsigned int R = rank, typename = std::enable_if_t<R == 2>>
  KOKKOS_INLINE_FUNCTION RealType& operator()(unsigned int i, unsigned int j)
  {
    ASSERT(i >= 0 && i < dim, "Row index out of bounds");
    ASSERT(j >= 0 && j < dim, "Col index out of bounds");
    return data_[i * dim + j];
  }

  template<unsigned int R = rank, typename = std::enable_if_t<R == 2>>
  KOKKOS_INLINE_FUNCTION const RealType& operator()(unsigned int i,
                                                    unsigned int j) const
  {
    ASSERT(i >= 0 && i < dim, "Row index out of bounds");
    ASSERT(j >= 0 && j < dim, "Col index out of bounds");
    return data_[i * dim + j];
  }

  KOKKOS_INLINE_FUNCTION
  Tensor& operator+=(const Tensor& other)
  {
    for (int i = 0; i < n_components; ++i) {
      data_[i] += other.data_[i];
    }
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  Tensor& operator-=(const Tensor& other)
  {
    for (int i = 0; i < n_components; ++i) {
      data_[i] -= other.data_[i];
    }
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  Tensor& operator*=(const RealType scalar)
  {
    for (int i = 0; i < n_components; ++i) {
      data_[i] *= scalar;
    }
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  Tensor& operator/=(const RealType scalar)
  {
    ASSERT_DEBUG(scalar != RealType(0), "Division by zero");
    for (int i = 0; i < n_components; ++i) {
      data_[i] /= scalar;
    }
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  Tensor operator+(const Tensor& other) const
  {
    Tensor result;
    for (int i = 0; i < n_components; ++i) {
      result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Tensor operator-(const Tensor& other) const
  {
    Tensor result;
    for (int i = 0; i < n_components; ++i) {
      result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Tensor operator*(const RealType scalar) const
  {
    Tensor result;
    for (int i = 0; i < n_components; ++i) {
      result.data_[i] = data_[i] * scalar;
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Tensor operator-() const
  {
    Tensor result;
    for (int i = 0; i < n_components; ++i) {
      result.data_[i] = -data_[i];
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  void clear()
  {
    for (int i = 0; i < n_components; ++i) {
      data_[i] = RealType(0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  RealType norm() const { return Kokkos::sqrt(norm_square()); }

  KOKKOS_INLINE_FUNCTION
  RealType norm_square() const
  {
    RealType sum = RealType(0);
    for (int i = 0; i < n_components; ++i) {
      sum += data_[i] * data_[i];
    }
    return sum;
  }

  KOKKOS_INLINE_FUNCTION
  RealType* data() { return data_; }

  KOKKOS_INLINE_FUNCTION
  const RealType* data() const { return data_; }

private:
  RealType data_[n_components];
};

template<int rank, int dim, typename RealType>
KOKKOS_INLINE_FUNCTION Tensor<rank, dim, RealType>
operator*(RealType scalar, const Tensor<rank, dim, RealType>& t)
{
  return t * scalar;
}

template<int dim, typename RealType>
KOKKOS_INLINE_FUNCTION RealType
dot(const Tensor<1, dim, RealType>& a, const Tensor<1, dim, RealType>& b)
{
  RealType sum = RealType(0);
  for (int i = 0; i < dim; ++i) {
    sum += a(i) * b(i);
  }
  return sum;
}

template<int dim, typename RealType>
KOKKOS_INLINE_FUNCTION Tensor<1, dim, RealType>
operator*(const Tensor<2, dim, RealType>& A, const Tensor<1, dim, RealType>& v)
{
  Tensor<1, dim, RealType> result;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      result(i) += A(i, j) * v(j);
    }
  }
  return result;
}

template<int dim, typename RealType>
KOKKOS_INLINE_FUNCTION Tensor<2, dim, RealType>
outer(const Tensor<1, dim, RealType>& a, const Tensor<1, dim, RealType>& b)
{
  Tensor<2, dim, RealType> result;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      result(i, j) = a(i) * b(j);
    }
  }
  return result;
}

template<int dim, typename RealType>
KOKKOS_INLINE_FUNCTION Tensor<2, dim, RealType>
transpose(const Tensor<2, dim, RealType>& A)
{
  Tensor<2, dim, RealType> result;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      result(i, j) = A(j, i);
    }
  }
  return result;
}

template<int dim, typename RealType>
KOKKOS_INLINE_FUNCTION RealType
trace(const Tensor<2, dim, RealType>& A)
{
  RealType sum = RealType(0);
  for (int i = 0; i < dim; ++i) {
    sum += A(i, i);
  }
  return sum;
}

template<int dim, typename RealType>
KOKKOS_INLINE_FUNCTION RealType
double_contract(const Tensor<2, dim, RealType>& A,
                const Tensor<2, dim, RealType>& B)
{
  RealType sum = RealType(0);
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      sum += A(i, j) * B(i, j);
    }
  }
  return sum;
}
