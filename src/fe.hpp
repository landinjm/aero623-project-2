#pragma once

#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <iostream>
#include <quad.hpp>
#include <tensor.hpp>
#include <triangulation.hpp>

/**
 * @brief Discontinuous full-order Langrange basis
 */
template<unsigned int dim, typename RealType>
class FE_DGLagrangeSimplex
{
public:
  static constexpr unsigned int n_dofs_per_cell(unsigned int p)
  {
    unsigned int result = 1;
    for (unsigned int i = 0; i < dim; ++i) {
      result = result * (p + 1 + i) / (i + 1);
    }
    return result;
  }

  explicit FE_DGLagrangeSimplex(unsigned int p)
    : p_(p)
    , n_dofs_(n_dofs_per_cell(p))
  {
    static_assert(dim >= 1 && dim <= 3, "Only dim=1,2,3 supported");
    nodes_.resize(n_dofs_, std::array<RealType, dim>{});
    init_nodes();
    check_basis();
  }

  unsigned int degree() const { return p_; }
  unsigned int n_dofs() const { return n_dofs_; }

  RealType shape_value(unsigned int i, const Tensor<1, dim, RealType>& pt) const
  {
    ASSERT(i < n_dofs_, "Index out of range");
    return eval(i, pt);
  }

  Tensor<1, dim, RealType> shape_gradient(
    unsigned int i,
    const Tensor<1, dim, RealType>& pt) const
  {
    ASSERT(i < n_dofs_, "Index out of range");
    return eval_gradient(i, pt);
  }

  Tensor<1, dim, RealType> node(unsigned int i) const
  {
    ASSERT(i < n_dofs_, "Index out of range");
    Tensor<1, dim, RealType> x;
    for (unsigned int d = 0; d < dim; ++d) {
      x(d) = nodes_[i][d];
    }
    return x;
  }

private:
  unsigned int p_;
  unsigned int n_dofs_;

  std::vector<std::array<RealType, dim>> nodes_;

  /**
   * Initialize the equally spaced nodes on the reference simplex
   */
  void init_nodes()
  {
    // Special case p = 0 has one node at the centroid
    if (p_ == 0) {
      for (unsigned int d = 0; d < dim; ++d) {
        nodes_[0][d] = RealType(1) / RealType(dim + 1);
      }
      return;
    }

    unsigned int idx = 0;

    // For dim = 1 evenly place nodes
    if constexpr (dim == 1) {
      for (unsigned int i0 = 0; i0 <= p_; ++i0) {
        nodes_[idx][0] = RealType(i0) / p_;
        ++idx;
      }
    }
    // For dim = 2 evenly place nodes in barycentric coords
    else if constexpr (dim == 2) {
      for (unsigned int i1 = 0; i1 <= p_; ++i1) {
        for (unsigned int i0 = 0; i0 <= p_ - i1; ++i0) {
          nodes_[idx][0] = RealType(i0) / p_;
          nodes_[idx][1] = RealType(i1) / p_;
          ++idx;
        }
      }
    }
    // For dim = 3 evenly placec nodes in barycentric coords
    else if constexpr (dim == 3) {
      for (unsigned int i2 = 0; i2 <= p_; ++i2) {
        for (unsigned int i1 = 0; i1 <= p_ - i2; ++i1) {
          for (unsigned int i0 = 0; i0 <= p_ - i1 - i2; ++i0) {
            nodes_[idx][0] = RealType(i0) / p_;
            nodes_[idx][1] = RealType(i1) / p_;
            nodes_[idx][2] = RealType(i2) / p_;
            ++idx;
          }
        }
      }
    }
  }

  /**
   * @brief 1-D Lagrange basis polynomial
   */
  RealType lagrange_1d(unsigned int a, unsigned int deg, RealType t) const
  {
    RealType result = RealType(1);
    for (unsigned int j = 0; j <= deg; ++j) {
      if (j == a) {
        continue;
      }
      result *= (RealType(deg) * t - RealType(j)) / (RealType(a) - RealType(j));
    }
    return result;
  }

  /**
   * @brief 1-D gradient of Lagrange basis polynomial
   */
  RealType lagrange_1d_deriv(unsigned int a, unsigned int deg, RealType t) const
  {
    RealType deriv = RealType(0);
    for (unsigned int m = 0; m <= deg; ++m) {
      if (m == a) {
        continue;
      }
      RealType term = RealType(deg) / (RealType(a) - RealType(m));
      for (unsigned int j = 0; j <= deg; ++j) {
        if (j == a || j == m) {
          continue;
        }
        term *= (RealType(deg) * t - RealType(j)) / (RealType(a) - RealType(j));
      }
      deriv += term;
    }
    return deriv;
  }

  /**
   * Given an index return the three indices that where used to generate the
   * evenly spaced nodes.
   *
   * These correspond to i0, i1, and i2 above.
   */
  std::array<unsigned int, dim> multi_index(unsigned int i) const
  {
    std::array<unsigned int, dim> a{};
    // dim = 1 is a flat index
    if constexpr (dim == 1) {
      a[0] = i;
    }
    // dim = 2 is barycentric
    else if constexpr (dim == 2) {
      unsigned int idx = 0;
      for (unsigned int i1 = 0; i1 <= p_; ++i1) {
        for (unsigned int i0 = 0; i0 <= p_ - i1; ++i0, ++idx) {
          if (idx == i) {
            a[0] = i0;
            a[1] = i1;
            return a;
          }
        }
      }
    }
    // dim = 3 is barycentric
    else if constexpr (dim == 3) {
      unsigned int idx = 0;
      for (unsigned int i2 = 0; i2 <= p_; ++i2) {
        for (unsigned int i1 = 0; i1 <= p_ - i2; ++i1) {
          for (unsigned int i0 = 0; i0 <= p_ - i1 - i2; ++i0, ++idx) {
            if (idx == i) {
              a[0] = i0;
              a[1] = i1;
              a[2] = i2;
              return a;
            }
          }
        }
      }
    }
    return a;
  }

  /**
   * Take a point in cartesian space and map it to barycentric space
   */
  std::array<RealType, dim + 1> barycentric(
    const Tensor<1, dim, RealType>& pt) const
  {
    std::array<RealType, dim + 1> lam{};
    RealType sum = RealType(0);
    for (unsigned int d = 0; d < dim; ++d) {
      lam[d] = pt(d);
      sum += pt(d);
    }
    lam[dim] = RealType(1) - sum;
    return lam;
  }

  RealType eval(unsigned int i, const Tensor<1, dim, RealType>& pt) const
  {
    // Special case p = 0
    if (p_ == 0) {
      return RealType(1);
    }

    auto a = multi_index(i);
    auto lam = barycentric(pt);

    RealType result = RealType(1);
    unsigned int used = 0;
    RealType remaining = RealType(1);

    for (unsigned int k = 0; k < dim; ++k) {
      const unsigned int deg = p_ - used;

      if (remaining < RealType(1e-14)) {
        if (a[k] != 0) {
          return RealType(0);
        }
      } else {
        const RealType t = lam[k] / remaining;
        result *= lagrange_1d(a[k], deg, t);
      }

      remaining -= lam[k];
      used += a[k];
    }

    return result;
  }

  Tensor<1, dim, RealType> eval_gradient(
    unsigned int i,
    const Tensor<1, dim, RealType>& pt) const
  {
    // Special case p = 0
    Tensor<1, dim, RealType> grad;
    if (p_ == 0) {
      return grad;
    }

    auto a = multi_index(i);
    auto lam = barycentric(pt);

    struct FactorInfo
    {
      RealType t;
      RealType R;
      RealType L;
      RealType Ld;
      unsigned int deg;
      unsigned int used_before;
    };

    std::array<FactorInfo, dim> finfo{};
    unsigned int used = 0;
    RealType remaining = RealType(1);

    for (unsigned int k = 0; k < dim; ++k) {
      finfo[k].R = remaining;
      finfo[k].used_before = used;
      finfo[k].deg = p_ - used;

      if (remaining < RealType(1e-14)) {
        finfo[k].t = RealType(0);
        finfo[k].L = (a[k] == 0) ? RealType(1) : RealType(0);
        finfo[k].Ld = RealType(0);
      } else {
        finfo[k].t = lam[k] / remaining;
        finfo[k].L = lagrange_1d(a[k], finfo[k].deg, finfo[k].t);
        finfo[k].Ld = lagrange_1d_deriv(a[k], finfo[k].deg, finfo[k].t);
      }

      remaining -= lam[k];
      used += a[k];
    }

    std::array<RealType, dim + 1> prefix{}, suffix{};
    prefix[0] = RealType(1);
    for (unsigned int k = 0; k < dim; ++k) {
      prefix[k + 1] = prefix[k] * finfo[k].L;
    }

    suffix[dim] = RealType(1);
    for (int k = int(dim) - 1; k >= 0; --k) {
      suffix[k] = suffix[k + 1] * finfo[k].L;
    }

    for (unsigned int d = 0; d < dim; ++d) {
      RealType g = RealType(0);

      for (unsigned int k = 0; k < dim; ++k) {
        if (finfo[k].R < RealType(1e-14)) {
          continue;
        }

        RealType dtk_dxd = RealType(0);
        if (d == k) {
          dtk_dxd += RealType(1) / finfo[k].R;
        }
        if (d < k) {
          dtk_dxd += lam[k] / (finfo[k].R * finfo[k].R);
        }
        RealType dphidk = prefix[k] * finfo[k].Ld * dtk_dxd * suffix[k + 1];
        g += dphidk;
      }

      grad(d) = g;
    }

    return grad;
  }

  /**
   * Check for the kronecker delta rule of basis functions
   */
  void check_basis() const
  {
    for (unsigned int i = 0; i < n_dofs_; ++i) {
      for (unsigned int j = 0; j < n_dofs_; ++j) {
        RealType val = eval(i, node(j));
        RealType expected = (i == j) ? RealType(1) : RealType(0);
        ASSERT(std::abs(val - expected) < 1.0e-10, "Failed basis check");
      }
    }
  }
};

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
    // I don't want to deal with other dimensions
    static_assert(dim == 2);

    // Build a Jacobian from the vertices of the cell.
    RealType J[dim][dim];
    RealType x0[dim];

    for (unsigned int d = 0; d < dim; ++d) {
      x0[d] = cell.vertex(0)(d);
    }

    // J columns are edge vectors from vertex 0
    for (unsigned int d = 0; d < dim; ++d) {
      J[d][0] = cell.vertex(1)(d) - cell.vertex(0)(d);
      J[d][1] = cell.vertex(2)(d) - cell.vertex(0)(d);
    }

    // Take the inverse and determinant
    const RealType det_J = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    const RealType J_inv[dim][dim] = { { J[1][1] / det_J, -J[0][1] / det_J },
                                       { -J[1][0] / det_J, J[0][0] / det_J } };

    // det_J should be positive
    ASSERT(det_J > 0,
           "Negative Jacobian on cell " + std::to_string(cell.index()) +
             ", det_J = " + std::to_string(det_J));

    // After computing q_points, verify reference points are in reference
    // triangle
    for (unsigned int q = 0; q < n_q_; ++q) {
      const auto xi = quad_.point(q);
      const RealType x = xi(0);
      const RealType y = xi(1);
      ASSERT(x >= -1e-10 && y >= -1e-10 && x + y <= 1.0 + 1e-10,
             "Quadrature point outside reference triangle: (" +
               std::to_string(x) + ", " + std::to_string(y) + ")");
    }

    for (unsigned int q = 0; q < n_q_; ++q) {
      JxW_(q) = std::abs(det_J) * quad_.weight(q);

      const auto xi = quad_.point(q);

      for (unsigned int i = 0; i < n_dofs_; ++i) {
        phi_(i, q) = fe_.shape_value(i, xi);

        const auto tmp = fe_.shape_gradient(i, xi);
        for (unsigned int d = 0; d < dim; ++d) {
          grad_phi_(i, q, d) = J_inv[0][d] * tmp(0) + J_inv[1][d] * tmp(1);
        }
      }

      for (unsigned int d = 0; d < dim; ++d) {
        q_point_(q, d) = x0[d] + J[d][0] * xi(0) + J[d][1] * xi(1);
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
   * @brief Reinit the cell so JxW and quadrature point values reflect the
   * real geometry.
   */
  template<typename CellAccessor>
  void reinit(const CellAccessor& cell, unsigned int face)
  {
    // I don't want to deal with other dimensions
    static_assert(dim == 2);

    ASSERT(face < SimplexTopology<dim>::faces_per_cell,
           "Local face number must be less than the number of faces per cell");

    // The challenge with this class is that we have the quadrature rule
    // defined along the face, but the basis functions defined along the cell.
    // As such, we must map from reference line to reference triangle.

    // Build a Jacobian from the vertices of the cell.
    RealType J[dim][dim];
    RealType x0[dim];

    for (unsigned int d = 0; d < dim; ++d) {
      x0[d] = cell.vertex(0)(d);
    }

    // J columns are edge vectors from vertex 0
    for (unsigned int d = 0; d < dim; ++d) {
      J[d][0] = cell.vertex(1)(d) - cell.vertex(0)(d);
      J[d][1] = cell.vertex(2)(d) - cell.vertex(0)(d);
    }

    // Take the inverse and determinant
    const RealType det_J = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    const RealType J_inv[dim][dim] = { { J[1][1] / det_J, -J[0][1] / det_J },
                                       { -J[1][0] / det_J, J[0][0] / det_J } };

    // Here is where things become a little different than FEValues.
    // 1. Each face maps a 1D quad point to the 2D coordinates on the
    // reference triangle.
    // 2. Each face maps to a reference normal vector.
    // 3. The edge length in reference space gives the face Jacobian.

    RealType ref_tangent[dim];
    RealType ref_origin[dim];
    RealType ref_normal[dim];

    switch (face) {
      case 0: {
        // Face 0 -> v1=(1,0) and v2=(0,1)
        ref_origin[0] = 1.0;
        ref_origin[1] = 0.0;
        ref_tangent[0] = -1.0;
        ref_tangent[1] = 1.0;
        ref_normal[0] = 1.0;
        ref_normal[1] = 1.0;
        break;
      }
      case 1: {
        // Face 1 -> v2=(0,1) and v0=(0,0)
        ref_origin[0] = 0.0;
        ref_origin[1] = 1.0;
        ref_tangent[0] = 0.0;
        ref_tangent[1] = -1.0;
        ref_normal[0] = -1.0;
        ref_normal[1] = 0.0;
        break;
      }
      case 2: {
        // Face 2 -> v0=(0,0) and v1=(1,0)
        ref_origin[0] = 0.0;
        ref_origin[1] = 0.0;
        ref_tangent[0] = 1.0;
        ref_tangent[1] = 0.0;
        ref_normal[0] = 0.0;
        ref_normal[1] = -1.0;
        break;
      }
    }

    // Now grab the physical normal and tangent
    RealType n_phys[dim];
    n_phys[0] = J_inv[0][0] * ref_normal[0] + J_inv[1][0] * ref_normal[1];
    n_phys[1] = J_inv[0][1] * ref_normal[0] + J_inv[1][1] * ref_normal[1];

    RealType t_phys[dim];
    t_phys[0] = J[0][0] * ref_tangent[0] + J[0][1] * ref_tangent[1];
    t_phys[1] = J[1][0] * ref_tangent[0] + J[1][1] * ref_tangent[1];

    const RealType phys_edge_len =
      std::sqrt(t_phys[0] * t_phys[0] + t_phys[1] * t_phys[1]);

    // Normalize physical normal
    const RealType n_phys_norm =
      std::sqrt(n_phys[0] * n_phys[0] + n_phys[1] * n_phys[1]);

    for (unsigned int q = 0; q < n_q_; ++q) {
      // Grab the 1D quad points
      const auto xi_face = quad_.point(q);
      const RealType t = xi_face(0);

      // Map to 2D reference coords
      RealType xi_ref[dim];
      xi_ref[0] = ref_origin[0] + t * ref_tangent[0];
      xi_ref[1] = ref_origin[1] + t * ref_tangent[1];

      // Wrap in a Tensor
      Tensor<1, dim, RealType> xi;
      xi(0) = xi_ref[0];
      xi(1) = xi_ref[1];

      JxW_(q) = phys_edge_len * quad_.weight(q);

      for (unsigned int d = 0; d < dim; ++d) {
        q_point_(q, d) = x0[d] + J[d][0] * xi_ref[0] + J[d][1] * xi_ref[1];
      }

      for (unsigned int d = 0; d < dim; ++d) {
        normal_(q, d) = n_phys[d] / n_phys_norm;
      }

      for (unsigned int i = 0; i < n_dofs_; ++i) {
        phi_(i, q) = fe_.shape_value(i, xi);

        const auto tmp = fe_.shape_gradient(i, xi);
        for (unsigned int d = 0; d < dim; ++d) {
          grad_phi_(i, q, d) = J_inv[0][d] * tmp(0) + J_inv[1][d] * tmp(1);
        }
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
