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
    init_nodes();
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

private:
  unsigned int p_;
  unsigned int n_dofs_;

  RealType nodes_[max_dofs_][dim];

  static RealType fixed_pow(RealType x, int n)
  {
    RealType r = RealType(1);
    for (int i = 0; i < n; ++i)
      r *= x;
    return r;
  }

  void init_nodes()
  {
    switch (p_) {
      case 0:
        nodes_[0][0] = 1.0 / 3.0;
        nodes_[0][1] = 1.0 / 3.0;
        break;
      case 1:
        nodes_[0][0] = 0.0;
        nodes_[0][1] = 0.0;
        nodes_[1][0] = 1.0;
        nodes_[1][1] = 0.0;
        nodes_[2][0] = 0.0;
        nodes_[2][1] = 1.0;
        break;
      case 2:
        nodes_[0][0] = 0.0;
        nodes_[0][1] = 0.0;
        nodes_[1][0] = 1.0;
        nodes_[1][1] = 0.0;
        nodes_[2][0] = 0.0;
        nodes_[2][1] = 1.0;
        nodes_[3][0] = 0.5;
        nodes_[3][1] = 0.5;
        nodes_[4][0] = 0.0;
        nodes_[4][1] = 0.5;
        nodes_[5][0] = 0.5;
        nodes_[5][1] = 0.0;
        break;
      case 3:
        nodes_[0][0] = 0.0;
        nodes_[0][1] = 0.0;
        nodes_[1][0] = 1.0;
        nodes_[1][1] = 0.0;
        nodes_[2][0] = 0.0;
        nodes_[2][1] = 1.0;
        nodes_[3][0] = 2.0 / 3.0;
        nodes_[3][1] = 1.0 / 3.0;
        nodes_[4][0] = 1.0 / 3.0;
        nodes_[4][1] = 2.0 / 3.0;
        nodes_[5][0] = 0.0;
        nodes_[5][1] = 2.0 / 3.0;
        nodes_[6][0] = 0.0;
        nodes_[6][1] = 1.0 / 3.0;
        nodes_[7][0] = 1.0 / 3.0;
        nodes_[7][1] = 0.0;
        nodes_[8][0] = 2.0 / 3.0;
        nodes_[8][1] = 0.0;
        nodes_[9][0] = 1.0 / 3.0;
        nodes_[9][1] = 1.0 / 3.0;
        break;
    }
    check_basis();
  }

  RealType eval(unsigned int i, const Tensor<1, dim, RealType>& point) const
  {
    const RealType x = point(0);
    const RealType y = point(1);
    switch (p_) {
      case 0:
        return RealType(1);
      case 1:
        switch (i) {
          case 0:
            return 1.0 - x - y;
          case 1:
            return x;
          case 2:
            return y;
        }
      case 2:
        switch (i) {
          case 0:
            return 1.0 - 3.0 * x - 3.0 * y + 2.0 * x * x + 4.0 * x * y +
                   2.0 * y * y;
          case 1:
            return -x + 2.0 * x * x;
          case 2:
            return -y + 2.0 * y * y;
          case 3:
            return 4.0 * x * y;
          case 4:
            return 4.0 * y - 4.0 * x * y - 4.0 * y * y;
          case 5:
            return 4.0 * x - 4.0 * x * x - 4.0 * x * y;
        }
      case 3:
        switch (i) {
          case 0:
            return 1.0 - 11.0 / 2.0 * x - 11.0 / 2.0 * y + 9.0 * x * x +
                   18.0 * x * y + 9.0 * y * y - 9.0 / 2.0 * x * x * x -
                   27.0 / 2.0 * x * x * y - 27.0 / 2.0 * x * y * y -
                   9.0 / 2.0 * y * y * y;
          case 1:
            return x - 9.0 / 2.0 * x * x + 9.0 / 2.0 * x * x * x;
          case 2:
            return y - 9.0 / 2.0 * y * y + 9.0 / 2.0 * y * y * y;
          case 3:
            return -9.0 / 2.0 * x * y + 27.0 / 2.0 * x * x * y;
          case 4:
            return -9.0 / 2.0 * x * y + 27.0 / 2.0 * x * y * y;
          case 5:
            return -9.0 / 2.0 * y + 9.0 / 2.0 * x * y + 18.0 * y * y -
                   27.0 / 2.0 * x * y * y - 27.0 / 2.0 * y * y * y;
          case 6:
            return 9.0 * y - 45.0 / 2.0 * x * y - 45.0 / 2.0 * y * y +
                   27.0 / 2.0 * x * x * y + 27.0 * x * y * y +
                   27.0 / 2.0 * y * y * y;
          case 7:
            return 9.0 * x - 45.0 / 2.0 * x * x - 45.0 / 2.0 * x * y +
                   27.0 / 2.0 * x * x * x + 27.0 * x * x * y +
                   27.0 / 2.0 * x * y * y;
          case 8:
            return -9.0 / 2.0 * x + 18.0 * x * x + 9.0 / 2.0 * x * y -
                   27.0 / 2.0 * x * x * x - 27.0 / 2.0 * x * x * y;
          case 9:
            return 27.0 * x * y - 27.0 * x * x * y - 27.0 * x * y * y;
        }
    }
    return RealType(0);
  }

  Tensor<1, dim, RealType> eval_gradient(
    unsigned int i,
    const Tensor<1, dim, RealType>& point) const
  {
    const RealType x = point(0);
    const RealType y = point(1);
    Tensor<1, dim, RealType> grad;

    switch (p_) {
      case 0:
        grad(0) = 0.0;
        grad(1) = 0.0;
        break;
      case 1:
        switch (i) {
          case 0:
            grad(0) = -1.0;
            grad(1) = -1.0;
            break;
          case 1:
            grad(0) = 1.0;
            grad(1) = 0.0;
            break;
          case 2:
            grad(0) = 0.0;
            grad(1) = 1.0;
            break;
        }
        break;
      case 2:
        switch (i) {
          case 0:
            grad(0) = -3.0 + 4.0 * x + 4.0 * y;
            grad(1) = -3.0 + 4.0 * x + 4.0 * y;
            break;
          case 1:
            grad(0) = -1.0 + 4.0 * x;
            grad(1) = 0.0;
            break;
          case 2:
            grad(0) = 0.0;
            grad(1) = -1.0 + 4.0 * y;
            break;
          case 3:
            grad(0) = 4.0 * y;
            grad(1) = 4.0 * x;
            break;
          case 4:
            grad(0) = -4.0 * y;
            grad(1) = 4.0 - 4.0 * x - 8.0 * y;
            break;
          case 5:
            grad(0) = 4.0 - 8.0 * x - 4.0 * y;
            grad(1) = -4.0 * x;
            break;
        }
        break;
      case 3:
        switch (i) {
          case 0:
            grad(0) = -11.0 / 2.0 + 18.0 * x + 18.0 * y - 27.0 / 2.0 * x * x -
                      27.0 * x * y - 27.0 / 2.0 * y * y;
            grad(1) = -11.0 / 2.0 + 18.0 * x + 18.0 * y - 27.0 / 2.0 * x * x -
                      27.0 * x * y - 27.0 / 2.0 * y * y;
            break;
          case 1:
            grad(0) = 1.0 - 9.0 * x + 27.0 / 2.0 * x * x;
            grad(1) = 0.0;
            break;
          case 2:
            grad(0) = 0.0;
            grad(1) = 1.0 - 9.0 * y + 27.0 / 2.0 * y * y;
            break;
          case 3:
            grad(0) = -9.0 / 2.0 * y + 27.0 * x * y;
            grad(1) = -9.0 / 2.0 * x + 27.0 / 2.0 * x * x;
            break;
          case 4:
            grad(0) = -9.0 / 2.0 * y + 27.0 / 2.0 * y * y;
            grad(1) = -9.0 / 2.0 * x + 27.0 * x * y;
            break;
          case 5:
            grad(0) = 9.0 / 2.0 * y - 27.0 / 2.0 * y * y;
            grad(1) = -9.0 / 2.0 + 9.0 / 2.0 * x + 36.0 * y - 27.0 * x * y -
                      81.0 / 2.0 * y * y;
            break;
          case 6:
            grad(0) = -45.0 / 2.0 * y + 27.0 * x * y + 27.0 * y * y;
            grad(1) = 9.0 - 45.0 / 2.0 * x - 45.0 * y + 27.0 / 2.0 * x * x +
                      54.0 * x * y + 81.0 / 2.0 * y * y;
            break;
          case 7:
            grad(0) = 9.0 - 45.0 * x - 45.0 / 2.0 * y + 81.0 / 2.0 * x * x +
                      54.0 * x * y + 27.0 / 2.0 * y * y;
            grad(1) = -45.0 / 2.0 * x + 27.0 * x * x + 27.0 * x * y;
            break;
          case 8:
            grad(0) = -9.0 / 2.0 + 36.0 * x + 9.0 / 2.0 * y -
                      81.0 / 2.0 * x * x - 27.0 * x * y;
            grad(1) = 9.0 / 2.0 * x - 27.0 / 2.0 * x * x;
            break;
          case 9:
            grad(0) = 27.0 * y - 54.0 * x * y - 27.0 * y * y;
            grad(1) = 27.0 * x - 27.0 * x * x - 54.0 * x * y;
            break;
        }
        break;
    }
    return grad;
  }

  void check_basis() const
  {
    // Check φᵢ(node_j) = δᵢⱼ
    for (int i = 0; i < (int)n_dofs_; ++i) {
      for (int j = 0; j < (int)n_dofs_; ++j) {
        Tensor<1, dim, RealType> pt;
        for (int d = 0; d < (int)dim; ++d)
          pt(d) = nodes_[j][d];
        RealType val = eval(i, pt);
        RealType expected = (i == j) ? RealType(1) : RealType(0);
        if (std::abs(val - expected) >= 1.0e-10) {
          std::cout << "FAIL: phi_" << i << "(node_" << j << ") = " << val
                    << " expected " << expected << " node=(" << nodes_[j][0]
                    << "," << nodes_[j][1] << ")" << std::endl;
          ASSERT(std::abs(val - expected) < 1.0e-10, "Failed basis");
        }
      }
    }
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
      constexpr unsigned int table[] = { 0, 1, 3, 4, 6, 7, 12, 13 };
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
  static constexpr unsigned int max_order_ = 7;
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
        const RealType a = 0.166666666666667;
        const RealType b = 0.666666666666667;
        pts = { { { b, a } }, { { a, a } }, { { a, b } } };
        wts = { 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0 };
        break;
      }
      case 3: {
        const RealType w0 = -0.281250000000000;
        const RealType w1 = 0.260416666666667;
        pts = { { { 0.333333333333333, 0.333333333333333 } },
                { { 0.600000000000000, 0.200000000000000 } },
                { { 0.200000000000000, 0.200000000000000 } },
                { { 0.200000000000000, 0.600000000000000 } } };
        wts = { w0, w1, w1, w1 };
        break;
      }
      case 4: {
        const RealType w1 = 0.111690794839005;
        const RealType w2 = 0.054975871827661;
        pts = { { { 0.108103018168070, 0.445948490915965 } },
                { { 0.445948490915965, 0.445948490915965 } },
                { { 0.445948490915965, 0.108103018168070 } },
                { { 0.816847572980459, 0.091576213509771 } },
                { { 0.091576213509771, 0.091576213509771 } },
                { { 0.091576213509771, 0.816847572980459 } } };
        wts = { w1, w1, w1, w2, w2, w2 };
        break;
      }
      case 5: {
        const RealType w0 = 0.112500000000000;
        const RealType w1 = 0.066197076394253;
        const RealType w2 = 0.062969590272414;
        pts = { { { 0.333333333333333, 0.333333333333333 } },
                { { 0.059715871789770, 0.470142064105115 } },
                { { 0.470142064105115, 0.470142064105115 } },
                { { 0.470142064105115, 0.059715871789770 } },
                { { 0.797426985353087, 0.101286507323456 } },
                { { 0.101286507323456, 0.101286507323456 } },
                { { 0.101286507323456, 0.797426985353087 } } };
        wts = { w0, w1, w1, w1, w2, w2, w2 };
        break;
      }
      case 6: {
        const RealType w1 = 0.058393137863189;
        const RealType w2 = 0.025422453185103;
        const RealType w3 = 0.041425537809187;
        pts = { { { 0.501426509658179, 0.249286745170910 } },
                { { 0.249286745170910, 0.249286745170910 } },
                { { 0.249286745170910, 0.501426509658179 } },
                { { 0.873821971016996, 0.063089014491502 } },
                { { 0.063089014491502, 0.063089014491502 } },
                { { 0.063089014491502, 0.873821971016996 } },
                { { 0.053145049844817, 0.310352451033784 } },
                { { 0.310352451033784, 0.636502499121399 } },
                { { 0.636502499121399, 0.053145049844817 } },
                { { 0.310352451033784, 0.053145049844817 } },
                { { 0.053145049844817, 0.636502499121399 } },
                { { 0.636502499121399, 0.310352451033784 } } };
        wts = { w1, w1, w1, w2, w2, w2, w3, w3, w3, w3, w3, w3 };
        break;
      }
      case 7: {
        const RealType w0 = -0.074785022233841;
        const RealType w1 = 0.087807628716604;
        const RealType w2 = 0.026673617804419;
        const RealType w3 = 0.038556880445128;
        pts = { { { 0.333333333333333, 0.333333333333333 } },
                { { 0.479308067841920, 0.260345966079040 } },
                { { 0.260345966079040, 0.260345966079040 } },
                { { 0.260345966079040, 0.479308067841920 } },
                { { 0.869739794195568, 0.065130102902216 } },
                { { 0.065130102902216, 0.065130102902216 } },
                { { 0.065130102902216, 0.869739794195568 } },
                { { 0.048690315425316, 0.312865496004874 } },
                { { 0.312865496004874, 0.638444188569810 } },
                { { 0.638444188569810, 0.048690315425316 } },
                { { 0.312865496004874, 0.048690315425316 } },
                { { 0.048690315425316, 0.638444188569810 } },
                { { 0.638444188569810, 0.312865496004874 } } };
        wts = { w0, w1, w1, w1, w2, w2, w2, w3, w3, w3, w3, w3, w3 };
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
   * @brief Reinit the cell so JxW and quadrature point values reflect the
   * real geometry.
   */
  template<typename CellAccessor>
  void reinit(const CellAccessor& cell)
  {
    static_assert(dim == 2);

    const unsigned int n_geom = cell.n_geometry_nodes();

    for (unsigned int q = 0; q < n_q_; ++q) {

      const auto xi = quad_.point(q);

      // compute geometry using FE basis
      RealType x[dim] = {0,0};
      RealType J[dim][dim] = {{0,0},{0,0}};

      for (unsigned int i = 0; i < n_geom; ++i) {

        const RealType phi_i = geom_shape(i,xi,n_geom);
        const auto dphi_i = geom_grad(i,xi,n_geom);

        const auto X = cell.geometry_node(i);
        const RealType xi0 = X(0);
        const RealType xi1 = X(1);

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

        phi_(i,q)=fe_.shape_value(i,xi);

        const auto tmp = fe_.shape_gradient(i, xi);

        grad_phi_(i,q,0) =
          J_inv[0][0]*tmp(0) + J_inv[1][0]*tmp(1);

        grad_phi_(i,q,1) =
          J_inv[0][1]*tmp(0) + J_inv[1][1]*tmp(1);
      }
    }
  }

  RealType geom_shape(unsigned int i,
                      const Tensor<1,dim,RealType>& xi,
                      unsigned int n_geom)
  {
      const RealType x = xi[0];
      const RealType y = xi[1];

      const RealType l1 = 1.0 - x - y;
      const RealType l2 = x;
      const RealType l3 = y;

      if (n_geom == 3) {
          switch(i) {
              case 0: return l1;
              case 1: return l2;
              case 2: return l3;
          }
      }

      if (n_geom == 6) {
          switch(i) {
              case 0: return l1*(2*l1-1);
              case 1: return l2*(2*l2-1);
              case 2: return l3*(2*l3-1);
              case 3: return 4*l1*l2;
              case 4: return 4*l2*l3;
              case 5: return 4*l3*l1;
          }
      }

      ASSERT(false,"Unsupported geometry order");
  }

  Tensor<1,dim,RealType>
  geom_grad(
      unsigned int i,
      const Tensor<1,dim,RealType>& xi,
      unsigned int n_geom)
  {
      const RealType x = xi[0];
      const RealType y = xi[1];

      const RealType l1 = 1.0 - x - y;
      const RealType l2 = x;
      const RealType l3 = y;

      if (n_geom == 3) {
          switch(i) {
              case 0: return { -1.0, -1.0 }; // grad(l1)
              case 1: return {  1.0,  0.0 }; // grad(l2)
              case 2: return {  0.0,  1.0 }; // grad(l3)
          }
      }

      if (n_geom == 6) {
          switch(i) {
              case 0: return { -4*l1 + 1, -4*l1 + 1 };
              case 1: return {  4*l2 - 1,  0.0 };
              case 2: return {  0.0,  4*l3 - 1 };
              case 3: return {  4*(l1 - l2), -4*l2 };
              case 4: return {  4*l3,  4*l2 };
              case 5: return { -4*l3,  4*(l1 - l3) };
          }
      }

      ASSERT(false,"Unsupported geometry order");
      return {}; // silence compiler
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
   * @brief Reinit the cell so JxW and quadrature point values reflect the
   * real geometry.
   */

  template<typename CellAccessor>
  void reinit(const CellAccessor& cell, unsigned int face)
  {
    static_assert(dim == 2);

    ASSERT(face < SimplexTopology<dim>::faces_per_cell,
          "Local face number must be less than the number of faces per cell");

    const unsigned int n_geom = cell.n_geometry_nodes();

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

      for (unsigned int i = 0; i < n_geom; ++i) {

        const RealType phi_i = geom_shape(i,xi,n_geom);
        const auto dphi_i = geom_grad(i,xi,n_geom);

        const RealType dNdt =
          dphi_i(0)*drdt + dphi_i(1)*dsdt;

        const auto X = cell.geometry_node(i);
        const RealType xi0 = X(0);
        const RealType xi1 = X(1);

        x[0] += phi_i * xi0;
        x[1] += phi_i * xi1;

        dxdt[0] += xi0 * dNdt;
        dxdt[1] += xi1 * dNdt;
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

      for (unsigned int i = 0; i < n_geom; ++i) {
        const auto dNi = geom_grad(i, xi, n_geom);
        const auto X = cell.geometry_node(i);
        J[0][0]+=X(0)*dNi(0);
        J[0][1]+=X(0)*dNi(1);
        J[1][0]+=X(1)*dNi(0);
        J[1][1]+=X(1)*dNi(1);
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

  RealType geom_shape(unsigned int i,
                    const Tensor<1,dim,RealType>& xi,
                    unsigned int n_geom)
  {
      const RealType x = xi[0];
      const RealType y = xi[1];

      const RealType l1 = 1.0 - x - y;
      const RealType l2 = x;
      const RealType l3 = y;

      if (n_geom == 3) {
          switch(i) {
              case 0: return l1;
              case 1: return l2;
              case 2: return l3;
          }
      }

      if (n_geom == 6) {
          switch(i) {
              case 0: return l1*(2*l1-1);
              case 1: return l2*(2*l2-1);
              case 2: return l3*(2*l3-1);
              case 3: return 4*l1*l2;
              case 4: return 4*l2*l3;
              case 5: return 4*l3*l1;
          }
      }

      ASSERT(false,"Unsupported geometry order");
  }

  Tensor<1,dim,RealType>
  geom_grad(
      unsigned int i,
      const Tensor<1,dim,RealType>& xi,
      unsigned int n_geom)
  {
      const RealType x = xi[0];
      const RealType y = xi[1];

      const RealType l1 = 1.0 - x - y;
      const RealType l2 = x;
      const RealType l3 = y;

      if (n_geom == 3) {
          switch(i) {
              case 0: return { -1.0, -1.0 }; // grad(l1)
              case 1: return {  1.0,  0.0 }; // grad(l2)
              case 2: return {  0.0,  1.0 }; // grad(l3)
          }
      }

      if (n_geom == 6) {
          switch(i) {
              case 0: return { -4*l1 + 1, -4*l1 + 1 };
              case 1: return {  4*l2 - 1,  0.0 };
              case 2: return {  0.0,  4*l3 - 1 };
              case 3: return {  4*(l1 - l2), -4*l2 };
              case 4: return {  4*l3,  4*l2 };
              case 5: return { -4*l3,  4*(l1 - l3) };
          }
      }

      ASSERT(false,"Unsupported geometry order");
      return {}; // silence compiler
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
