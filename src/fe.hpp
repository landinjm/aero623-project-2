#pragma once

#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <iostream>
#include <tensor.hpp>
#include <triangulation.hpp>

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

    // Transpose for cache locality
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
  static constexpr unsigned int n_q_points(unsigned int p)
  {
    // TODO: This is not the best way to do this
    if constexpr (dim == 1) {
      return p;
    }
    if constexpr (dim == 2) {
      // From your switch: 1->1, 2->3, 3->4, 4->6
      constexpr unsigned int table[] = { 0, 1, 3, 4, 6 };
      return table[p];
    }
    return 0;
  }

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

    ASSERT(n_points_ == n_q_points(order), "IDK");

    points_ = Kokkos::View<RealType**, Layout, HostMemSpace>(
      "simplex_quad_points", n_points_, dim);
    weights_ = Kokkos::View<RealType*, Layout, HostMemSpace>(
      "simplex_quad_weights", n_points_);

    for (unsigned int q = 0; q < n_points_; ++q) {
      for (unsigned int d = 0; d < dim; ++d) {
        points_(q, d) = pts[q][d];
      }
      weights_(q) = wts[q];
    }
  }

  unsigned int order() const { return order_; }
  unsigned int n_points() const { return n_points_; }

  Tensor<1, dim, RealType> point(unsigned int q) const
  {
    Tensor<1, dim, RealType> p;
    for (unsigned int d = 0; d < dim; ++d) {
      p(d) = points_(q, d);
    }
    return p;
  }

  RealType weight(unsigned int q) const { return weights_(q); };

  [[deprecated]]
  const Kokkos::View<RealType**, Layout, HostMemSpace>& points_host() const
  {
    return points_;
  }

  [[deprecated]]
  const Kokkos::View<RealType*, Layout, HostMemSpace>& weights_host() const
  {
    return weights_;
  }

private:
  static constexpr unsigned int max_order_ = 4;
  unsigned int order_;
  unsigned int n_points_;
  Kokkos::View<RealType**, Layout, HostMemSpace> points_; // [q, dim]
  Kokkos::View<RealType*, Layout, HostMemSpace> weights_; // [q]

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

template<unsigned int dim, typename RealType>
class FEValues
{
public:
  FEValues(const FE_DGQLegendre<dim, RealType>& fe,
           const QGaussSimplex<dim, RealType>& quad)
    : fe_(fe)
    , quad_(quad)
    , n_dofs_(fe.n_dofs())
    , n_q_(quad.n_points())
  {
    // Allocate views
    JxW_ = Kokkos::View<RealType*, Layout, HostMemSpace>("JxW", n_q_);
    q_point_ =
      Kokkos::View<RealType**, Layout, HostMemSpace>("q_point", n_q_, dim);
    phi_ = Kokkos::View<RealType**, Layout, HostMemSpace>("JxW", n_dofs_, n_q_);
    grad_phi_ = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "JxW", n_dofs_, n_q_, dim);
  }

  /**
   * @brief Reinit the cell so JxW and quadrature point values reflect the real
   * geometry.
   */
  template<typename CellAccessor>
  void reinit(const CellAccessor& cell)
  {
    static_assert(dim == 2);

    const unsigned int n_vertices = cell.n_vertices();

    for (unsigned int q = 0; q < n_q_; ++q) {

      const auto xi = quad_.point(q);

      // compute geometry using FE basis
      RealType x[dim] = {0,0};
      RealType J[dim][dim] = {{0,0},{0,0}};

      for (unsigned int i = 0; i < n_vertices; ++i) {

        const RealType phi_i = fe_.shape_value(i, xi);
        const auto dphi_i = fe_.shape_gradient(i, xi);

        const RealType xi0 = cell.vertex(i)(0);
        const RealType xi1 = cell.vertex(i)(1);

        x[0] += phi_i * xi0;
        x[1] += phi_i * xi1;

        J[0][0] += xi0 * dphi_i(0);
        J[0][1] += xi0 * dphi_i(1);
        J[1][0] += xi1 * dphi_i(0);
        J[1][1] += xi1 * dphi_i(1);
      }

      const RealType det_J =
        J[0][0]*J[1][1] - J[0][1]*J[1][0];

      const RealType J_inv[2][2] = {
        { J[1][1]/det_J, -J[0][1]/det_J },
        {-J[1][0]/det_J,  J[0][0]/det_J }
      };

      JxW_(q) = std::abs(det_J) * quad_.weight(q);

      q_point_(q,0) = x[0];
      q_point_(q,1) = x[1];

      for (unsigned int i = 0; i < n_dofs_; ++i) {

        phi_(i,q) = fe_.shape_value(i, xi);

        const auto tmp = fe_.shape_gradient(i, xi);

        grad_phi_(i,q,0) =
          J_inv[0][0]*tmp(0) + J_inv[1][0]*tmp(1);

        grad_phi_(i,q,1) =
          J_inv[0][1]*tmp(0) + J_inv[1][1]*tmp(1);
      }
    }
  }

  unsigned int n_dofs() const { return n_dofs_; }
  unsigned int n_q_points() const { return n_q_; }

  RealType JxW(unsigned int q) { return JxW_(q); };

  Tensor<1, dim, RealType> q_point(unsigned int q)
  {
    Tensor<1, dim, RealType> p;
    for (unsigned int d = 0; d < dim; ++d) {
      p(d) = q_point_(q, d);
    }
    return p;
  }

  RealType shape_value(unsigned int i, unsigned int q) { return phi_(i, q); }

  Tensor<1, dim, RealType> shape_gradient(unsigned int i, unsigned int q)
  {
    Tensor<1, dim, RealType> grad;
    for (unsigned int d = 0; d < dim; ++d) {
      grad(d) = grad_phi_(i, q, d);
    }
    return grad;
  }

private:
  const FE_DGQLegendre<dim, RealType>& fe_;
  const QGaussSimplex<dim, RealType>& quad_;

  unsigned int n_dofs_;
  unsigned int n_q_;

  Kokkos::View<RealType*, Layout, HostMemSpace> JxW_;        // [q]
  Kokkos::View<RealType**, Layout, HostMemSpace> q_point_;   // [q, dim]
  Kokkos::View<RealType**, Layout, HostMemSpace> phi_;       // [dof, q]
  Kokkos::View<RealType***, Layout, HostMemSpace> grad_phi_; // [dof, q, dim]
};

template<unsigned int dim, typename RealType>
class FEFaceValues
{
public:
  FEFaceValues(const FE_DGQLegendre<dim, RealType>& fe,
               const QGaussSimplex<dim - 1, RealType>& quad)
    : fe_(fe)
    , quad_(quad)
    , n_dofs_(fe.n_dofs())
    , n_q_(quad.n_points())
  {
    // Allocate views
    JxW_ = Kokkos::View<RealType*, Layout, HostMemSpace>("JxW", n_q_);
    q_point_ =
      Kokkos::View<RealType**, Layout, HostMemSpace>("q_point", n_q_, dim);
    normal_ =
      Kokkos::View<RealType**, Layout, HostMemSpace>("normal", n_q_, dim);
    phi_ = Kokkos::View<RealType**, Layout, HostMemSpace>("JxW", n_dofs_, n_q_);
    grad_phi_ = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "JxW", n_dofs_, n_q_, dim);
  }

  /**
   * @brief Reinit the cell so JxW and quadrature point values reflect the real
   * geometry.
   */
  template<typename CellAccessor>
  void reinit(const CellAccessor& cell, unsigned int face)
  {
    static_assert(dim == 2);

    ASSERT(face < SimplexTopology<dim>::faces_per_cell,
          "Local face number must be less than the number of faces per cell");

    const unsigned int n_vertices = cell.n_vertices();

    for (unsigned int q = 0; q < n_q_; ++q) {

      const auto xi_face = quad_.point(q);
      const RealType t = xi_face(0);

      RealType r,s;

      switch(face) {
        case 0: r = 1.0 - t; s = t; break;
        case 1: r = 0.0;     s = 1.0 - t; break;
        case 2: r = t;       s = 0.0; break;
      }

      Tensor<1,2,RealType> xi;
      xi(0)=r;
      xi(1)=s;

      RealType x[dim]={0,0};
      RealType dxdt[dim]={0,0};

      RealType drdt, dsdt;

      switch(face) {
        case 0: drdt=-1; dsdt=1; break;
        case 1: drdt=0;  dsdt=-1; break;
        case 2: drdt=1;  dsdt=0; break;
      }

      for(unsigned int i=0;i<n_vertices;++i){

        const RealType phi_i = fe_.shape_value(i,xi);
        const auto dphi_i = fe_.shape_gradient(i,xi);

        const RealType dNdt =
          dphi_i(0)*drdt + dphi_i(1)*dsdt;

        const RealType vx = cell.vertex(i)(0);
        const RealType vy = cell.vertex(i)(1);

        x[0] += phi_i * vx;
        x[1] += phi_i * vy;

        dxdt[0] += vx * dNdt;
        dxdt[1] += vy * dNdt;
      }

      const RealType edge_len =
        std::sqrt(dxdt[0]*dxdt[0] + dxdt[1]*dxdt[1]);

      normal_(q,0) =  dxdt[1]/edge_len;
      normal_(q,1) = -dxdt[0]/edge_len;

      JxW_(q) = edge_len * quad_.weight(q);

      q_point_(q,0)=x[0];
      q_point_(q,1)=x[1];

      // Jacobian for gradient transform
      RealType J[dim][dim]={{0,0},{0,0}};

      for(unsigned int i=0;i<n_vertices;++i){
        const auto dNi = fe_.shape_gradient(i,xi);
        J[0][0]+=cell.vertex(i)(0)*dNi(0);
        J[0][1]+=cell.vertex(i)(0)*dNi(1);
        J[1][0]+=cell.vertex(i)(1)*dNi(0);
        J[1][1]+=cell.vertex(i)(1)*dNi(1);
      }

      const RealType det =
        J[0][0]*J[1][1]-J[0][1]*J[1][0];

      const RealType J_inv[2][2]={
        { J[1][1]/det, -J[0][1]/det},
        {-J[1][0]/det,  J[0][0]/det}
      };

      for(unsigned int i=0;i<n_dofs_;++i){

        phi_(i,q)=fe_.shape_value(i,xi);

        const auto tmp=fe_.shape_gradient(i,xi);

        grad_phi_(i,q,0)=
          J_inv[0][0]*tmp(0)+J_inv[1][0]*tmp(1);

        grad_phi_(i,q,1)=
          J_inv[0][1]*tmp(0)+J_inv[1][1]*tmp(1);
      }
    }
  }

  unsigned int n_dofs() const { return n_dofs_; }
  unsigned int n_q_points() const { return n_q_; }

  RealType JxW(unsigned int q) { return JxW_(q); };

  Tensor<1, dim, RealType> q_point(unsigned int q)
  {
    Tensor<1, dim, RealType> p;
    for (unsigned int d = 0; d < dim; ++d) {
      p(d) = q_point_(q, d);
    }
    return p;
  }

  Tensor<1, dim, RealType> normal(unsigned int q) const
  {
    Tensor<1, dim, RealType> n;
    for (unsigned int d = 0; d < dim; ++d) {
      n(d) = normal_(q, d);
    }
    return n;
  }

  RealType shape_value(unsigned int i, unsigned int q) { return phi_(i, q); }

  Tensor<1, dim, RealType> shape_gradient(unsigned int i, unsigned int q)
  {
    Tensor<1, dim, RealType> grad;
    for (unsigned int d = 0; d < dim; ++d) {
      grad(d) = grad_phi_(i, q, d);
    }
    return grad;
  }

private:
  const FE_DGQLegendre<dim, RealType>& fe_;
  const QGaussSimplex<dim - 1, RealType>& quad_;

  unsigned int n_dofs_;
  unsigned int n_q_;

  Kokkos::View<RealType*, Layout, HostMemSpace> JxW_;        // [q]
  Kokkos::View<RealType**, Layout, HostMemSpace> q_point_;   // [q, dim]
  Kokkos::View<RealType**, Layout, HostMemSpace> normal_;    // [q, dim]
  Kokkos::View<RealType**, Layout, HostMemSpace> phi_;       // [dof, q]
  Kokkos::View<RealType***, Layout, HostMemSpace> grad_phi_; // [dof, q, dim]
};
