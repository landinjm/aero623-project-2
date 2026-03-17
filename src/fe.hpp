#pragma once

#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <tensor.hpp>

/**
 * @brief Discontinuous Gauss-Legendre quadrature rule using full-order basis
 * for simplex elements.
 */
template<unsigned int dim, typename RealType>
class FE_DGP
{
  KOKKOS_INLINE_FUNCTION
  static constexpr unsigned int n_dofs_per_cell(unsigned int p)
  {
    if constexpr (dim == 1) {
      return p + 1;
    }
    if constexpr (dim == 2) {
      return (p + 1) * (p + 2) / 2;
    }
    return 0;
  }

  explicit FE_DGP(const unsigned int p)
    : p_(p)
    , n_dofs_(n_dofs_per_cell(p))
  {
    ASSERT(p <= max_degree_, "Polynomial degree exceeds maximum");
    build_monomial_table();
  };

  unsigned int degree() const { return p_; }
  unsigned int n_dofs() const { return n_dofs_; }

  KOKKOS_INLINE_FUNCTION
  RealType shape_value(unsigned int i,
                       const Tensor<1, dim, RealType>& point) const
  {
    ASSERT(i < n_dofs_, "Basis function index out of range");
    return eval_monomial(i, point);
  }

  KOKKOS_INLINE_FUNCTION
  Tensor<1, dim, RealType> shape_gradient(
    unsigned int i,
    const Tensor<1, dim, RealType>& point) const
  {
    ASSERT(i < n_dofs_, "Basis function index out of range");
    return eval_monomial_gradient(i, point);
  }

private:
  static constexpr unsigned int max_degree_ = 3;
  static constexpr unsigned int max_dofs_ = 10;

  unsigned int p_;
  unsigned int n_dofs_;

  unsigned int exponents_[max_dofs_][dim];

  void build_monomial_table()
  {
    unsigned int idx = 0;
    if constexpr (dim == 1) {
      for (unsigned int a = 0; a <= p_; ++a) {
        exponents_[idx++][0] = a;
      }
    } else if constexpr (dim == 2) {
      for (unsigned int deg = 0; deg <= p_; ++deg) {
        for (unsigned int a = 0; a <= deg; ++a) {
          exponents_[idx][0] = deg - a;
          exponents_[idx][1] = a;
          ++idx;
        }
      }
    }

    ASSERT(idx == n_dofs_, "Monomial table size mismatch");
  }

  KOKKOS_INLINE_FUNCTION RealType
  eval_monomial(unsigned int i, const Tensor<1, dim, RealType>& point) const
  {
    RealType val = RealType(1);
    for (unsigned int d = 0; d < dim; ++d) {
      const unsigned int exp = exponents_[i][d];
      for (unsigned int k = 0; k < exp; ++k) {
        val *= point(d);
      }
    }
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  Tensor<1, dim, RealType> eval_monomial_gradient(
    unsigned int i,
    const Tensor<1, dim, RealType>& point) const
  {
    Tensor<1, dim, RealType> grad;

    for (unsigned int d = 0; d < dim; ++d) {
      const unsigned int exp = exponents_[i][d];
      if (exp == 0) {
        grad(d) = RealType(0);
        continue;
      }

      RealType val = RealType(exp);
      for (unsigned int d2 = 0; d2 < dim; ++d2) {
        unsigned int e = exponents_[i][d2];
        if (d2 == d) {
          --e;
        }
        for (unsigned int k = 0; k < e; ++k) {
          val *= point(d2);
        }
      }
      grad(d) = val;
    }
    return grad;
  }
};

/**
 * @brief Gauss quadrature on the reference simplex.
 */
template<unsigned int dim, typename RealType>
class QGaussSimplex
{
public:
  using DeviceScalarView = VectorViewTrait<RealType, DeviceMemSpace>::type;
  using HostScalarView = VectorViewTrait<RealType, HostMemSpace>::type;
  using DevicePointView = MatrixViewTrait<RealType, DeviceMemSpace>::type;
  using HostPointView = MatrixViewTrait<RealType, HostMemSpace>::type;

  explicit QGaussSimplex(unsigned int order)
    : order_(order)
  {
    ASSERT(order >= 1, "Quadrature order must be at least 1");
    ASSERT(order <= max_order_, "Quadrature order exceeds maximum");

    // Get rule on host
    std::vector<std::array<RealType, dim>> pts;
    std::vector<RealType> wts;
    get_rule(order, pts, wts);

    n_points_ = static_cast<unsigned int>(wts.size());

    points_device_ = DevicePointView("simplex_quad_points", n_points_, dim);
    weights_device_ = DeviceScalarView("simplex_quad_weights", n_points_);

    auto points_h = Kokkos::create_mirror_view(points_device_);
    auto weights_h = Kokkos::create_mirror_view(weights_device_);

    for (unsigned int q = 0; q < n_points_; ++q) {
      for (unsigned int d = 0; d < dim; ++d) {
        points_h(q, d) = pts[q][d];
      }
      weights_h(q) = wts[q];
    }

    Kokkos::deep_copy(points_device_, points_h);
    Kokkos::deep_copy(weights_device_, weights_h);
  }

  unsigned int order() const { return order_; }
  unsigned int n_points() const { return n_points_; }

  DevicePointView points() const { return points_device_; }
  DeviceScalarView weights() const { return weights_device_; }

  HostPointView points_host() const
  {
    auto h = Kokkos::create_mirror_view(points_device_);
    Kokkos::deep_copy(h, points_device_);
    return h;
  }

  HostScalarView weights_host() const
  {
    auto h = Kokkos::create_mirror_view(weights_device_);
    Kokkos::deep_copy(h, weights_device_);
    return h;
  }

private:
  static constexpr unsigned int max_order_ = 4;
  unsigned int order_;
  unsigned int n_points_;
  DevicePointView points_device_;
  DeviceScalarView weights_device_;

  static void get_rule(unsigned int order,
                       std::vector<std::array<RealType, dim>>& pts,
                       std::vector<RealType>& wts)
  {
    if constexpr (dim == 1) {
      get_line_rule(order, pts, wts);
    }
    if constexpr (dim == 2) {
      get_triangle_rule(order, pts, wts);
    }
  }

  static void get_line_rule(unsigned int order,
                            std::vector<std::array<RealType, 1>>& pts,
                            std::vector<RealType>& wts)
  {
    // We can compute Gauss-Legendre points via Newton iteration
    const unsigned int n = order;
    pts.resize(n);
    wts.resize(n);

    constexpr RealType tol = std::numeric_limits<RealType>::epsilon();
    const int m = (n + 1) / 2;

    std::vector<RealType> x(n), w(n);
    for (int i = 0; i < m; ++i) {
      RealType xi = -std::cos(M_PI * (i + 0.75) / (n + 0.5));
      RealType dx = RealType(0);
      do {
        RealType p0 = RealType(1), p1 = xi;
        for (int k = 1; k < (int)n; ++k) {
          RealType p2 = ((2 * k + 1) * xi * p1 - k * p0) / (k + 1);
          p0 = p1;
          p1 = p2;
        }
        RealType dp = n * (p0 - xi * p1) / (RealType(1) - xi * xi);
        dx = p1 / dp;
        xi -= dx;
      } while (std::abs(dx) > tol);

      RealType p0 = RealType(1), p1 = xi;
      for (int k = 1; k < (int)n; ++k) {
        RealType p2 = ((2 * k + 1) * xi * p1 - k * p0) / (k + 1);
        p0 = p1;
        p1 = p2;
      }
      RealType dp = n * (p0 - xi * p1) / (RealType(1) - xi * xi);
      RealType wi = RealType(2) / ((RealType(1) - xi * xi) * dp * dp);

      // Remap [-1,1] -> [0,1]
      x[i] = (RealType(1) + xi) / RealType(2);
      x[n - 1 - i] = (RealType(1) - xi) / RealType(2);
      w[i] = wi / RealType(2);
      w[n - 1 - i] = wi / RealType(2);
    }

    for (unsigned int q = 0; q < n; ++q) {
      pts[q][0] = x[q];
      wts[q] = w[q];
    }
  }

  static void get_triangle_rule(unsigned int order,
                                std::vector<std::array<RealType, 2>>& pts,
                                std::vector<RealType>& wts)
  {
    switch (order) {
      case 1: {
        pts = { { { 1.0 / 3.0, 1.0 / 3.0 } } };
        wts = { 0.5 };
        break;
      }
      case 2: {
        const RealType a = 1.0 / 6.0;
        const RealType b = 2.0 / 3.0;
        pts = { { { a, a } }, { { b, a } }, { { a, b } } };
        wts = { 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0 };
        break;
      }
      case 3: {
        const RealType w0 = -27.0 / 96.0;
        const RealType w1 = 25.0 / 96.0;
        pts = { { { 1.0 / 3.0, 1.0 / 3.0 } },
                { { 1.0 / 5.0, 1.0 / 5.0 } },
                { { 3.0 / 5.0, 1.0 / 5.0 } },
                { { 1.0 / 5.0, 3.0 / 5.0 } } };
        wts = { w0, w1, w1, w1 };
        break;
      }
      case 4: {
        const RealType a1 = 0.445948490915965;
        const RealType b1 = 1.0 - 2.0 * a1;
        const RealType a2 = 0.091576213509771;
        const RealType b2 = 1.0 - 2.0 * a2;
        const RealType w1 = 0.223381589678011 * 0.5;
        const RealType w2 = 0.109951743655322 * 0.5;
        pts = { { { a1, a1 } }, { { b1, a1 } }, { { a1, b1 } },
                { { a2, a2 } }, { { b2, a2 } }, { { a2, b2 } } };
        wts = { w1, w1, w1, w2, w2, w2 };
        break;
      }
      default:
        ASSERT(false, "Unsupported quadrature order for triangle");
    }
  }
};

/**
 * @brief Precomputed shape values and gradients at quadrature points.
 */
template<unsigned int dim, typename RealType>
class FEValues
{
public:
  using ShapeValueView = VectorViewTrait<RealType, DeviceMemSpace>::type;
  using ShapeGradView = MatrixViewTrait<RealType, DeviceMemSpace>::type;

  FEValues(const FE_DGP<dim, RealType>& fe,
           const QGaussSimplex<dim, RealType>& quad)
    : n_dofs_(fe.n_dofs())
    , n_q_points_(quad.n_points())
  {
    phi_ = ShapeValueView("phi", n_dofs_, n_q_points_);
    grad_phi_ = ShapeGradView("grad_phi", n_dofs_, n_q_points_, dim);

    auto phi_h = Kokkos::create_mirror_view(phi_);
    auto grad_phi_h = Kokkos::create_mirror_view(grad_phi_);

    auto points_h = quad.points_host();

    for (unsigned int i = 0; i < n_dofs_; ++i) {
      for (unsigned int q = 0; q < n_q_points_; ++q) {
        Tensor<1, dim, RealType> xi;
        for (unsigned int d = 0; d < dim; ++d) {
          xi(d) = points_h(q, d);
        }

        phi_h(i, q) = fe.shape_value(i, xi);

        auto grad = fe.shape_gradient(i, xi);
        for (unsigned int d = 0; d < dim; ++d) {
          grad_phi_h(i, q, d) = grad(d);
        }
      }
    }

    Kokkos::deep_copy(phi_, phi_h);
    Kokkos::deep_copy(grad_phi_, grad_phi_h);
  }

  unsigned int n_dofs() const { return n_dofs_; }
  unsigned int n_q_points() const { return n_q_points_; }

  ShapeValueView phi() const { return phi_; }
  ShapeGradView grad_phi() const { return grad_phi_; }

private:
  unsigned int n_dofs_;
  unsigned int n_q_points_;
  ShapeValueView phi_;
  ShapeGradView grad_phi_;
};
