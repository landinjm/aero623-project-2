#pragma once

#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <tensor.hpp>

/**
 * @brief Discontinuous full-order Langrange basis
 */
template<unsigned int dim, typename RealType>
class FE_DGQLegendre
{
public:
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

  static constexpr unsigned int max_degree_ = 3;
  static constexpr unsigned int max_dofs_ = 10;

  explicit FE_DGQLegendre(const unsigned int p)
    : p_(p)
    , n_dofs_(n_dofs_per_cell(p))
  {
    ASSERT(p <= max_degree_, "Polynomial degree exceeds maximum");
    build_monomial_table();
  };

  unsigned int degree() const { return p_; }
  unsigned int n_dofs() const { return n_dofs_; }

  RealType shape_value(unsigned int i,
                       const Tensor<1, dim, RealType>& point) const
  {
    ASSERT(i < n_dofs_, "Basis function index out of range");
    return eval(i, point);
  }

  Tensor<1, dim, RealType> shape_gradient(
    unsigned int i,
    const Tensor<1, dim, RealType>& point) const
  {
    ASSERT(i < n_dofs_, "Basis function index out of range");
    return eval_gradient(i, point);
  }

  Tensor<1, dim, RealType> node(unsigned int i) const
  {
    ASSERT(i < n_dofs_, "Node index out of range");
    Tensor<1, dim, RealType> x;
    for (unsigned int d = 0; d < dim; ++d)
      x(d) = nodes_[i][d];
    return x;
  }

  const RealType (&get_coeffs() const)[max_dofs_][max_dofs_] { return coeffs_; }

private:
  unsigned int p_;
  unsigned int n_dofs_;

  RealType coeffs_[max_dofs_][max_dofs_];
  RealType nodes_[max_dofs_][dim];

  static void monomial_exponents(int k, int& r, int& s)
  {
    int idx = 0;
    for (s = 0; s <= (int)max_degree_; ++s)
      for (r = 0; r <= (int)max_degree_ - s; ++r, ++idx)
        if (idx == k)
          return;
  }

  static RealType fixed_pow(RealType x, int n)
  {
    RealType r = RealType(1);
    for (int i = 0; i < n; ++i)
      r *= x;
    return r;
  }

  void build_monomial_table()
  {
    const int N = n_dofs_;

    // Build nodes — same ordering as monomials: ix+iy <= p
    int idx = 0;
    for (int iy = 0; iy <= (int)p_; ++iy)
      for (int ix = 0; ix <= (int)p_ - iy; ++ix, ++idx) {
        nodes_[idx][0] =
          (p_ > 0) ? RealType(ix) / RealType(p_) : RealType(1) / 3;
        if constexpr (dim > 1)
          nodes_[idx][1] =
            (p_ > 0) ? RealType(iy) / RealType(p_) : RealType(1) / 3;
      }

    // Build Vandermonde matrix
    RealType A[max_dofs_][max_dofs_];
    for (int i = 0; i < N; ++i) {
      int k = 0;
      for (int s = 0; s <= (int)p_; ++s)
        for (int r = 0; r <= (int)p_ - s; ++r, ++k) {
          const RealType eta = (dim > 1) ? nodes_[i][1] : RealType(0);
          A[i][k] = fixed_pow(nodes_[i][0], r) * fixed_pow(eta, s);
        }
    }

    // rhs starts as identity, becomes A^{-1} = coeffs after solve
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        coeffs_[i][j] = (i == j) ? RealType(1) : RealType(0);

    // LU with partial pivoting
    int piv[max_dofs_];
    for (int k = 0; k < N; ++k) {
      // Find pivot
      int maxrow = k;
      RealType maxval = std::abs(A[k][k]);
      for (int i = k + 1; i < N; ++i)
        if (std::abs(A[i][k]) > maxval) {
          maxval = std::abs(A[i][k]);
          maxrow = i;
        }
      piv[k] = maxrow;

      for (int j = 0; j < N; ++j) {
        std::swap(A[k][j], A[maxrow][j]);
        std::swap(coeffs_[k][j], coeffs_[maxrow][j]);
      }

      for (int i = k + 1; i < N; ++i) {
        A[i][k] /= A[k][k];
        for (int j = k + 1; j < N; ++j)
          A[i][j] -= A[i][k] * A[k][j];
        for (int j = 0; j < N; ++j)
          coeffs_[i][j] -= A[i][k] * coeffs_[k][j];
      }
    }

    // Back substitution
    for (int k = N - 1; k >= 0; --k) {
      for (int j = 0; j < N; ++j)
        coeffs_[k][j] /= A[k][k];
      for (int i = 0; i < k; ++i)
        for (int j = 0; j < N; ++j)
          coeffs_[i][j] -= A[i][k] * coeffs_[k][j];
    }

    // Tranpose for cache locality
    RealType tmp[max_dofs_][max_dofs_];
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        tmp[j][i] = coeffs_[i][j];
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        coeffs_[i][j] = tmp[i][j];
  }

  RealType eval(unsigned int i, const Tensor<1, dim, RealType>& point) const
  {
    const RealType x = point(0);
    const RealType y = (dim > 1) ? point(1) : RealType(0);
    RealType val = RealType(0);
    int k = 0;
    for (int s = 0; s <= (int)p_; ++s)
      for (int r = 0; r <= (int)p_ - s; ++r, ++k)
        val += coeffs_[i][k] * fixed_pow(x, r) * fixed_pow(y, s);
    return val;
  }

  Tensor<1, dim, RealType> eval_gradient(
    unsigned int i,
    const Tensor<1, dim, RealType>& point) const
  {
    const RealType x = point(0);
    const RealType y = (dim > 1) ? point(1) : RealType(0);
    Tensor<1, dim, RealType> grad;
    int k = 0;
    for (int s = 0; s <= (int)p_; ++s) {
      for (int r = 0; r <= (int)p_ - s; ++r, ++k) {
        const RealType c = coeffs_[i][k];
        if (r > 0)
          grad(0) += c * RealType(r) * fixed_pow(x, r - 1) * fixed_pow(y, s);
        if (dim > 1 && s > 0)
          grad(1) += c * fixed_pow(x, r) * RealType(s) * fixed_pow(y, s - 1);
      }
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

  FEValues(const FE_DGQLegendre<dim, RealType>& fe,
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
