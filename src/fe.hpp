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
  using PhiView = typename MatrixViewTrait<RealType, DeviceMemSpace>::type;
  using GradView = Kokkos::View<RealType***, Layout, DeviceMemSpace>;
  using JxWView = typename VectorViewTrait<RealType, DeviceMemSpace>::type;
  using PointView = typename MatrixViewTrait<RealType, DeviceMemSpace>::type;

  struct DeviceProxy
  {
    PhiView phi;
    GradView grad_phi;
    JxWView JxW;
    PointView quadrature_points;

    unsigned int n_dofs;
    unsigned int n_q;

    KOKKOS_INLINE_FUNCTION
    RealType shape_value(unsigned int i, unsigned int q) const
    {
      return phi(i, q);
    }

    KOKKOS_INLINE_FUNCTION
    RealType shape_gradient(unsigned int i,
                            unsigned int q,
                            unsigned int d) const
    {
      return grad_phi(i, q, d);
    }

    KOKKOS_INLINE_FUNCTION
    RealType jxw(unsigned int q) const { return JxW(q); }

    KOKKOS_INLINE_FUNCTION
    RealType quadrature_point(unsigned int q, unsigned int d) const
    {
      return quadrature_points(q, d);
    }
  };

  FEValues(const FE_DGQLegendre<dim, RealType>& fe,
           const QGaussSimplex<dim, RealType>& quad)
    : fe_(fe)
    , quad_(quad)
    , n_dofs_(fe.n_dofs())
    , n_q_(quad.n_points())
  {
    // Allocate device views
    phi_ref_ = PhiView("phi_ref", n_dofs_, n_q_);
    grad_phi_ref_ = GradView("grad_phi_ref", n_dofs_, n_q_, dim);
    grad_phi_phys_ = GradView("grad_phi_phys", n_dofs_, n_q_, dim);
    JxW_ = JxWView("JxW", n_q_);
    quad_points_ = PointView("quad_points", n_q_, dim);

    // Precompute reference-space values — same for every cell
    auto phi_h = Kokkos::create_mirror_view(phi_ref_);
    auto grad_ref_h = Kokkos::create_mirror_view(grad_phi_ref_);
    auto pts_h = quad_.points_host();

    for (unsigned int i = 0; i < n_dofs_; ++i) {
      for (unsigned int q = 0; q < n_q_; ++q) {
        Tensor<1, dim, RealType> xi;
        for (unsigned int d = 0; d < dim; ++d)
          xi(d) = pts_h(q, d);

        phi_h(i, q) = fe_.shape_value(i, xi);

        auto g = fe_.shape_gradient(i, xi);
        for (unsigned int d = 0; d < dim; ++d)
          grad_ref_h(i, q, d) = g(d);
      }
    }

    Kokkos::deep_copy(phi_ref_, phi_h);
    Kokkos::deep_copy(grad_phi_ref_, grad_ref_h);
  }

  template<typename CellAccessor>
  void reinit(const CellAccessor& cell)
  {
    // Build Jacobian from cell vertices (affine map)
    // x0 = vertex 0,  J[:,d] = vertex(d+1) - vertex(0)
    RealType x0[dim], J[dim][dim], Jinv[dim][dim];

    for (unsigned int d = 0; d < dim; ++d)
      x0[d] = cell.vertex(0)[d];

    for (unsigned int col = 0; col < dim; ++col) {
      auto xv = cell.vertex(col + 1);
      for (unsigned int d = 0; d < dim; ++d)
        J[d][col] = xv[d] - x0[d];
    }

    const RealType detJ = compute_det(J);
    ASSERT(detJ > RealType(0), "Negative Jacobian — check cell orientation");
    compute_inv(J, detJ, Jinv);

    // Fill JxW and quadrature points — host mirrors then deep_copy
    auto JxW_h = Kokkos::create_mirror_view(JxW_);
    auto qpts_h = Kokkos::create_mirror_view(quad_points_);
    auto wts_h = quad_.weights_host();
    auto pts_h = quad_.points_host();

    for (unsigned int q = 0; q < n_q_; ++q) {
      JxW_h(q) = detJ * wts_h(q);
      for (unsigned int d = 0; d < dim; ++d) {
        RealType xd = x0[d];
        for (unsigned int d2 = 0; d2 < dim; ++d2)
          xd += J[d][d2] * pts_h(q, d2);
        qpts_h(q, d) = xd;
      }
    }

    Kokkos::deep_copy(JxW_, JxW_h);
    Kokkos::deep_copy(quad_points_, qpts_h);

    // Physical gradients: grad_phys(i,q,d) = sum_{d2} Jinv(d2,d) *
    // grad_ref(i,q,d2) Do this on device to avoid a large host-side loop
    auto grad_ref = grad_phi_ref_; // device view, captured by value
    auto grad_phys = grad_phi_phys_;
    const unsigned int ndofs = n_dofs_;
    const unsigned int nq = n_q_;

    // Copy Jinv into a flat array for capture
    RealType Jinv_flat[dim * dim];
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int d2 = 0; d2 < dim; ++d2)
        Jinv_flat[d * dim + d2] = Jinv[d][d2];

    Kokkos::parallel_for(
      "grad_transform",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { (int)ndofs, (int)nq }),
      KOKKOS_LAMBDA(int i, int q) {
        for (unsigned int d = 0; d < dim; ++d) {
          RealType val = RealType(0);
          for (unsigned int d2 = 0; d2 < dim; ++d2)
            // J^{-T}_{d,d2} = Jinv[d2][d]
            val += Jinv_flat[d2 * dim + d] * grad_ref(i, q, d2);
          grad_phys(i, q, d) = val;
        }
      });
    Kokkos::fence();
  }

  DeviceProxy device_proxy() const
  {
    return { phi_ref_, grad_phi_phys_, JxW_, quad_points_, n_dofs_, n_q_ };
  }

  unsigned int n_dofs() const { return n_dofs_; }
  unsigned int n_q_points() const { return n_q_; }

private:
  const FE_DGQLegendre<dim, RealType>& fe_;
  const QGaussSimplex<dim, RealType>& quad_;

  unsigned int n_dofs_;
  unsigned int n_q_;

  PhiView phi_ref_;
  GradView grad_phi_ref_;
  GradView grad_phi_phys_;
  JxWView JxW_;
  PointView quad_points_;

  // Plain C array Jacobian helpers — no virtual, no std::
  static RealType compute_det(const RealType J[dim][dim])
  {
    if constexpr (dim == 1)
      return J[0][0];
    if constexpr (dim == 2)
      return J[0][0] * J[1][1] - J[0][1] * J[1][0];
    if constexpr (dim == 3)
      return J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
             J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
             J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
  }

  static void compute_inv(const RealType J[dim][dim],
                          RealType detJ,
                          RealType Jinv[dim][dim])
  {
    const RealType inv = RealType(1) / detJ;
    if constexpr (dim == 1) {
      Jinv[0][0] = inv;
    }
    if constexpr (dim == 2) {
      Jinv[0][0] = J[1][1] * inv;
      Jinv[0][1] = -J[0][1] * inv;
      Jinv[1][0] = -J[1][0] * inv;
      Jinv[1][1] = J[0][0] * inv;
    }
    if constexpr (dim == 3) {
      Jinv[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) * inv;
      Jinv[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) * inv;
      Jinv[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) * inv;
      Jinv[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) * inv;
      Jinv[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) * inv;
      Jinv[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) * inv;
      Jinv[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) * inv;
      Jinv[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) * inv;
      Jinv[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) * inv;
    }
  }
};

template<unsigned int dim, typename RealType>
class FEFaceValues
{
public:
  using PhiView = typename MatrixViewTrait<RealType, DeviceMemSpace>::type;
  using GradView = Kokkos::View<RealType***, Layout, DeviceMemSpace>;
  using JxWView = typename VectorViewTrait<RealType, DeviceMemSpace>::type;
  using PointView = typename MatrixViewTrait<RealType, DeviceMemSpace>::type;
  using NormalView = typename MatrixViewTrait<RealType, DeviceMemSpace>::type;

  struct DeviceProxy
  {
    PhiView phi;
    GradView grad_phi;
    JxWView JxW;
    PointView quadrature_points;
    NormalView normals;

    unsigned int n_dofs;
    unsigned int n_q;

    KOKKOS_INLINE_FUNCTION
    RealType shape_value(unsigned int i, unsigned int q) const
    {
      return phi(i, q);
    }

    KOKKOS_INLINE_FUNCTION
    RealType shape_gradient(unsigned int i,
                            unsigned int q,
                            unsigned int d) const
    {
      return grad_phi(i, q, d);
    }

    KOKKOS_INLINE_FUNCTION
    RealType jxw(unsigned int q) const { return JxW(q); }

    KOKKOS_INLINE_FUNCTION
    RealType quadrature_point(unsigned int q, unsigned int d) const
    {
      return quadrature_points(q, d);
    }

    KOKKOS_INLINE_FUNCTION
    RealType normal(unsigned int q, unsigned int d) const
    {
      return normals(q, d);
    }
  };

  // face_quad is a (dim-1)-dimensional rule
  FEFaceValues(const FE_DGQLegendre<dim, RealType>& fe,
               const QGaussSimplex<dim - 1, RealType>& face_quad)
    : fe_(fe)
    , face_quad_(face_quad)
    , n_dofs_(fe.n_dofs())
    , n_q_(face_quad.n_points())
  {
    phi_ = PhiView("fev_face_phi", n_dofs_, n_q_);
    grad_phi_ = GradView("fev_face_grad", n_dofs_, n_q_, dim);
    JxW_ = JxWView("fev_face_JxW", n_q_);
    points_ = PointView("fev_face_pts", n_q_, dim);
    normals_ = NormalView("fev_face_nrm", n_q_, dim);
  }

  // Call on host once per (cell, face_no) pair before launching kernel
  template<typename CellAccessor>
  void reinit(const CellAccessor& cell, unsigned int face_no)
  {
    ASSERT(face_no < dim + 1, "Face index out of range");

    // ----------------------------------------------------------------
    // Build cell Jacobian (same as FEValues::reinit)
    // ----------------------------------------------------------------
    RealType x0[dim], J[dim][dim], Jinv[dim][dim];

    for (unsigned int d = 0; d < dim; ++d)
      x0[d] = cell.vertex(0)[d];

    for (unsigned int col = 0; col < dim; ++col) {
      auto xv = cell.vertex(col + 1);
      for (unsigned int d = 0; d < dim; ++d)
        J[d][col] = xv[d] - x0[d];
    }

    const RealType detJ = compute_det(J);
    ASSERT(detJ > RealType(0), "Negative Jacobian — check cell orientation");
    compute_inv(J, detJ, Jinv);

    // ----------------------------------------------------------------
    // Map (dim-1) face quad points -> cell reference coords -> physical
    // ----------------------------------------------------------------
    auto face_pts_h = face_quad_.points_host();
    auto face_wts_h = face_quad_.weights_host();

    // Cell-reference coords of each face quad point
    RealType cell_ref[/* n_q_ */ 10][dim]; // bounded by max quad points
    for (unsigned int q = 0; q < n_q_; ++q)
      face_to_cell_ref(face_no, face_pts_h, q, cell_ref[q]);

    // ----------------------------------------------------------------
    // Compute outward normal and face Jacobian (|tangent| in 2D)
    // These are constant along a face for an affine map
    // ----------------------------------------------------------------
    RealType normal[dim];
    RealType face_jac;
    compute_face_normal_and_jac(face_no, J, normal, face_jac);

    // ----------------------------------------------------------------
    // Fill host mirrors and deep_copy
    // ----------------------------------------------------------------
    auto phi_h = Kokkos::create_mirror_view(phi_);
    auto grad_h = Kokkos::create_mirror_view(grad_phi_);
    auto JxW_h = Kokkos::create_mirror_view(JxW_);
    auto pts_h = Kokkos::create_mirror_view(points_);
    auto normals_h = Kokkos::create_mirror_view(normals_);

    for (unsigned int q = 0; q < n_q_; ++q) {
      // Physical coords of this face quad point
      for (unsigned int d = 0; d < dim; ++d) {
        RealType xd = x0[d];
        for (unsigned int d2 = 0; d2 < dim; ++d2)
          xd += J[d][d2] * cell_ref[q][d2];
        pts_h(q, d) = xd;
      }

      JxW_h(q) = face_jac * face_wts_h(q);

      for (unsigned int d = 0; d < dim; ++d)
        normals_h(q, d) = normal[d];

      // Shape values and physical gradients at this face point
      Tensor<1, dim, RealType> xi;
      for (unsigned int d = 0; d < dim; ++d)
        xi(d) = cell_ref[q][d];

      for (unsigned int i = 0; i < n_dofs_; ++i) {
        phi_h(i, q) = fe_.shape_value(i, xi);

        auto g_ref = fe_.shape_gradient(i, xi);
        for (unsigned int d = 0; d < dim; ++d) {
          RealType val = RealType(0);
          for (unsigned int d2 = 0; d2 < dim; ++d2)
            val += Jinv[d2][d] * g_ref(d2); // J^{-T} * grad_ref
          grad_h(i, q, d) = val;
        }
      }
    }

    Kokkos::deep_copy(phi_, phi_h);
    Kokkos::deep_copy(grad_phi_, grad_h);
    Kokkos::deep_copy(JxW_, JxW_h);
    Kokkos::deep_copy(points_, pts_h);
    Kokkos::deep_copy(normals_, normals_h);
  }

  DeviceProxy device_proxy() const
  {
    return { phi_, grad_phi_, JxW_, points_, normals_, n_dofs_, n_q_ };
  }

  unsigned int n_dofs() const { return n_dofs_; }
  unsigned int n_q_points() const { return n_q_; }

private:
  const FE_DGQLegendre<dim, RealType>& fe_;
  const QGaussSimplex<dim - 1, RealType>& face_quad_;

  unsigned int n_dofs_;
  unsigned int n_q_;

  PhiView phi_;
  GradView grad_phi_;
  JxWView JxW_;
  PointView points_;
  NormalView normals_;

  // ----------------------------------------------------------------
  // Map a (dim-1) face quad point to dim cell reference coordinates.
  //
  // 2D simplex face numbering (outward normals in reference space):
  //   face 0: v0=(0,0) -> v1=(1,0)   param: (t, 0)
  //   face 1: v1=(1,0) -> v2=(0,1)   param: (1-t, t)
  //   face 2: v2=(0,1) -> v0=(0,0)   param: (0, 1-t)
  // ----------------------------------------------------------------
  static void face_to_cell_ref(unsigned int face_no,
                               const auto& face_pts_h,
                               unsigned int q,
                               RealType cell_ref_q[dim])
  {
    static_assert(dim == 2, "Only implemented for dim=2");
    const RealType t = face_pts_h(q, 0);
    switch (face_no) {
      case 0:
        cell_ref_q[0] = t;
        cell_ref_q[1] = RealType(0);
        break;
      case 1:
        cell_ref_q[0] = RealType(1) - t;
        cell_ref_q[1] = t;
        break;
      case 2:
        cell_ref_q[0] = RealType(0);
        cell_ref_q[1] = RealType(1) - t;
        break;
      default:
        ASSERT(false, "Invalid face number");
    }
  }

  // Physical tangent -> face length and outward unit normal
  static void compute_face_normal_and_jac(unsigned int face_no,
                                          const RealType J[dim][dim],
                                          RealType normal[dim],
                                          RealType& face_jac)
  {
    static_assert(dim == 2, "Only implemented for dim=2");

    // Tangent in physical space for each reference face
    // face 0: d/dt (t, 0)    -> J * (1, 0)^T = J[:,0]
    // face 1: d/dt (1-t, t)  -> J * (-1, 1)^T
    // face 2: d/dt (0, 1-t)  -> J * (0, -1)^T = -J[:,1]
    RealType tangent[2];
    switch (face_no) {
      case 0:
        tangent[0] = J[0][0];
        tangent[1] = J[1][0];
        break;
      case 1:
        tangent[0] = -J[0][0] + J[0][1];
        tangent[1] = -J[1][0] + J[1][1];
        break;
      case 2:
        tangent[0] = -J[0][1];
        tangent[1] = -J[1][1];
        break;
      default:
        ASSERT(false, "Invalid face number");
    }

    face_jac = Kokkos::sqrt(tangent[0] * tangent[0] + tangent[1] * tangent[1]);

    // Rotate 90 degrees to get normal
    normal[0] = tangent[1] / face_jac;
    normal[1] = -tangent[0] / face_jac;

    // Flip if not outward — reference outward normals:
    // face 0: (0,-1), face 1: (1,1)/sqrt(2), face 2: (-1,0)
    const RealType ref_nx = (face_no == 0)   ? RealType(0)
                            : (face_no == 1) ? RealType(1)
                                             : -RealType(1);
    const RealType ref_ny = (face_no == 0)   ? -RealType(1)
                            : (face_no == 1) ? RealType(1)
                                             : RealType(0);
    if (normal[0] * ref_nx + normal[1] * ref_ny < RealType(0)) {
      normal[0] = -normal[0];
      normal[1] = -normal[1];
    }
  }

  // Identical helpers to FEValues — factor into a shared header
  // if they diverge from each other
  static RealType compute_det(const RealType J[dim][dim])
  {
    if constexpr (dim == 1)
      return J[0][0];
    if constexpr (dim == 2)
      return J[0][0] * J[1][1] - J[0][1] * J[1][0];
    if constexpr (dim == 3)
      return J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
             J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
             J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
  }

  static void compute_inv(const RealType J[dim][dim],
                          RealType detJ,
                          RealType Jinv[dim][dim])
  {
    const RealType inv = RealType(1) / detJ;
    if constexpr (dim == 1) {
      Jinv[0][0] = inv;
    }
    if constexpr (dim == 2) {
      Jinv[0][0] = J[1][1] * inv;
      Jinv[0][1] = -J[0][1] * inv;
      Jinv[1][0] = -J[1][0] * inv;
      Jinv[1][1] = J[0][0] * inv;
    }
    if constexpr (dim == 3) {
      Jinv[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) * inv;
      Jinv[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) * inv;
      Jinv[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) * inv;
      Jinv[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) * inv;
      Jinv[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) * inv;
      Jinv[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) * inv;
      Jinv[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) * inv;
      Jinv[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) * inv;
      Jinv[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) * inv;
    }
  }
};
