#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <iostream>
#include <simplex_gm_rule.hpp>
#include <tensor.hpp>

/**
 * @brief Returns the number of points associated with a Grundmann-Moeller
 * quadrature rule for a unit simplex of dimension `dim` with order `p`
 *
 * Translated from John Burkardt's code
 */
constexpr unsigned int
i4_choose(unsigned int n, unsigned int k)
{
  int i;
  int mn;
  int mx;
  int value;

  mn = std::min(k, n - k);

  if (mn < 0) {
    value = 0;
  } else if (mn == 0) {
    value = 1;
  } else {
    mx = std::max(k, n - k);
    value = mx + 1;

    for (i = 2; i <= mn; i++) {
      value = (value * (mx + i)) / i;
    }
  }

  return value;
}

template<int dim>
constexpr unsigned int
gm_rule_size(unsigned int p)
{
  unsigned int arg1 = dim + p + 1;
  unsigned int n = i4_choose(arg1, p);

  return n;
}

/**
 * @brief Gauss quadrature on the reference simplex.
 */
template<unsigned int dim, typename RealType>
class QGaussSimplex
{
public:
  static constexpr unsigned int n_q_points(unsigned int p)
  {
    return gm_rule_size<dim>(p);
  }

  explicit QGaussSimplex(unsigned int order)
    : order_(order)
    , points_("quad simplex points", n_q_points(order), dim)
    , weights_("quad simplex weights", n_q_points(order))
  {
    ASSERT(order_ >= 1, "The order must be greater than or equal to 1.");
    compute();
  };

  unsigned int order() const { return order_; }
  unsigned int n_points() const { return n_q_points(order_); }

  Tensor<1, dim, RealType> point(unsigned int q) const
  {
    Tensor<1, dim, RealType> p;
    for (unsigned int d = 0; d < dim; ++d) {
      p(d) = points_(q, d);
    }
    return p;
  }

  RealType weight(unsigned int q) const { return weights_(q); };

private:
  unsigned int order_;

  Kokkos::View<RealType**, Layout, HostMemSpace> points_; // [n_q_points, dim]
  Kokkos::View<RealType*, Layout, HostMemSpace> weights_; // [n_q_points]

  void compute()
  {
    // Allocate c-arrays
    auto n = n_q_points(order_);
    double w[n];
    double x[n * dim];

    // Get points and weights
    gm_unit_rule_set(order_, dim, n, w, x);

    // Fill out our views
    for (unsigned int q = 0; q < n; ++q) {
      weights_(q) = w[q];
      for (unsigned int d = 0; d < dim; ++d) {
        points_(q, d) = x[d + q * dim];
      }
    }
  }
};
