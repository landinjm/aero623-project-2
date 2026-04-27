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
    if constexpr (dim == 1) {
      return p + 1;
    }
    if constexpr (dim == 2) {
      return (p + 1) * (p + 2) / 2;
    }
    if constexpr (dim == 3) {
      return (p + 1) * (p + 2) * (p + 3) / 6;
    }
    return 0;
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

  static RealType fixed_pow(RealType x, int n)
  {
    RealType r = RealType(1);
    for (int i = 0; i < n; ++i)
      r *= x;
    return r;
  }

  void init_nodes()
  {
    if constexpr (dim == 1) {
      for (unsigned int i = 0; i <= p_; ++i)
        nodes_[i][0] =
          (p_ == 0) ? RealType(0.5) : static_cast<RealType>(i) / p_;
    }
    if constexpr (dim == 2) {
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
    }
    if constexpr (dim == 3) {
      // Reference tet: v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), v3=(0,0,1)
      switch (p_) {
        case 0:
          nodes_[0][0] = 0.25;
          nodes_[0][1] = 0.25;
          nodes_[0][2] = 0.25;
          break;
        case 1:
          nodes_[0][0] = 0.0;
          nodes_[0][1] = 0.0;
          nodes_[0][2] = 0.0;
          nodes_[1][0] = 1.0;
          nodes_[1][1] = 0.0;
          nodes_[1][2] = 0.0;
          nodes_[2][0] = 0.0;
          nodes_[2][1] = 1.0;
          nodes_[2][2] = 0.0;
          nodes_[3][0] = 0.0;
          nodes_[3][1] = 0.0;
          nodes_[3][2] = 1.0;
          break;
        case 2:
          // 4 vertices + 6 edge midpoints = 10 nodes
          nodes_[0][0] = 0.0;
          nodes_[0][1] = 0.0;
          nodes_[0][2] = 0.0;
          nodes_[1][0] = 1.0;
          nodes_[1][1] = 0.0;
          nodes_[1][2] = 0.0;
          nodes_[2][0] = 0.0;
          nodes_[2][1] = 1.0;
          nodes_[2][2] = 0.0;
          nodes_[3][0] = 0.0;
          nodes_[3][1] = 0.0;
          nodes_[3][2] = 1.0;
          nodes_[4][0] = 0.5;
          nodes_[4][1] = 0.5;
          nodes_[4][2] = 0.0; // mid v1-v2
          nodes_[5][0] = 0.0;
          nodes_[5][1] = 0.5;
          nodes_[5][2] = 0.0; // mid v0-v2
          nodes_[6][0] = 0.5;
          nodes_[6][1] = 0.0;
          nodes_[6][2] = 0.0; // mid v0-v1
          nodes_[7][0] = 0.5;
          nodes_[7][1] = 0.0;
          nodes_[7][2] = 0.5; // mid v1-v3
          nodes_[8][0] = 0.0;
          nodes_[8][1] = 0.5;
          nodes_[8][2] = 0.5; // mid v2-v3
          nodes_[9][0] = 0.0;
          nodes_[9][1] = 0.0;
          nodes_[9][2] = 0.5; // mid v0-v3
          break;
        case 3:
          // 4 vertices + 12 edge-third points + 4 face points = 20 nodes
          nodes_[0][0] = 0.0;
          nodes_[0][1] = 0.0;
          nodes_[0][2] = 0.0;
          nodes_[1][0] = 1.0;
          nodes_[1][1] = 0.0;
          nodes_[1][2] = 0.0;
          nodes_[2][0] = 0.0;
          nodes_[2][1] = 1.0;
          nodes_[2][2] = 0.0;
          nodes_[3][0] = 0.0;
          nodes_[3][1] = 0.0;
          nodes_[3][2] = 1.0;
          // edge v0-v1 thirds
          nodes_[4][0] = 1.0 / 3.0;
          nodes_[4][1] = 0.0;
          nodes_[4][2] = 0.0;
          nodes_[5][0] = 2.0 / 3.0;
          nodes_[5][1] = 0.0;
          nodes_[5][2] = 0.0;
          // edge v0-v2 thirds
          nodes_[6][0] = 0.0;
          nodes_[6][1] = 1.0 / 3.0;
          nodes_[6][2] = 0.0;
          nodes_[7][0] = 0.0;
          nodes_[7][1] = 2.0 / 3.0;
          nodes_[7][2] = 0.0;
          // edge v0-v3 thirds
          nodes_[8][0] = 0.0;
          nodes_[8][1] = 0.0;
          nodes_[8][2] = 1.0 / 3.0;
          nodes_[9][0] = 0.0;
          nodes_[9][1] = 0.0;
          nodes_[9][2] = 2.0 / 3.0;
          // edge v1-v2 thirds
          nodes_[10][0] = 2.0 / 3.0;
          nodes_[10][1] = 1.0 / 3.0;
          nodes_[10][2] = 0.0;
          nodes_[11][0] = 1.0 / 3.0;
          nodes_[11][1] = 2.0 / 3.0;
          nodes_[11][2] = 0.0;
          // edge v1-v3 thirds
          nodes_[12][0] = 2.0 / 3.0;
          nodes_[12][1] = 0.0;
          nodes_[12][2] = 1.0 / 3.0;
          nodes_[13][0] = 1.0 / 3.0;
          nodes_[13][1] = 0.0;
          nodes_[13][2] = 2.0 / 3.0;
          // edge v2-v3 thirds
          nodes_[14][0] = 0.0;
          nodes_[14][1] = 2.0 / 3.0;
          nodes_[14][2] = 1.0 / 3.0;
          nodes_[15][0] = 0.0;
          nodes_[15][1] = 1.0 / 3.0;
          nodes_[15][2] = 2.0 / 3.0;
          // face centroids
          nodes_[16][0] = 1.0 / 3.0;
          nodes_[16][1] = 1.0 / 3.0;
          nodes_[16][2] = 0.0;
          nodes_[17][0] = 1.0 / 3.0;
          nodes_[17][1] = 0.0;
          nodes_[17][2] = 1.0 / 3.0;
          nodes_[18][0] = 0.0;
          nodes_[18][1] = 1.0 / 3.0;
          nodes_[18][2] = 1.0 / 3.0;
          nodes_[19][0] = 1.0 / 3.0;
          nodes_[19][1] = 1.0 / 3.0;
          nodes_[19][2] = 1.0 / 3.0;
          break;
      }
    }
    check_basis();
  }

  RealType eval(unsigned int i, const Tensor<1, dim, RealType>& point) const
  {
    if constexpr (dim == 1) {
      const RealType x = point(0);
      RealType result = RealType(1);
      for (unsigned int j = 0; j <= p_; ++j) {
        if (j == i)
          continue;
        const RealType xj =
          (p_ == 0) ? RealType(0.5) : static_cast<RealType>(j) / p_;
        const RealType xi =
          (p_ == 0) ? RealType(0.5) : static_cast<RealType>(i) / p_;
        result *= (x - xj) / (xi - xj);
      }
      return result;
    }
    if constexpr (dim == 2) {
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
          break;
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
          break;
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
          break;
      }
    }
    if constexpr (dim == 3) {
      const RealType x = point(0);
      const RealType y = point(1);
      const RealType z = point(2);
      const RealType l0 = 1.0 - x - y - z;
      const RealType l1 = x, l2 = y, l3 = z;
      switch (p_) {
        case 0:
          return RealType(1);
        case 1:
          switch (i) {
            case 0:
              return l0;
            case 1:
              return l1;
            case 2:
              return l2;
            case 3:
              return l3;
          }
          break;
        case 2:
          switch (i) {
            case 0:
              return l0 * (2.0 * l0 - 1.0);
            case 1:
              return l1 * (2.0 * l1 - 1.0);
            case 2:
              return l2 * (2.0 * l2 - 1.0);
            case 3:
              return l3 * (2.0 * l3 - 1.0);
            case 4:
              return 4.0 * l1 * l2;
            case 5:
              return 4.0 * l0 * l2;
            case 6:
              return 4.0 * l0 * l1;
            case 7:
              return 4.0 * l1 * l3;
            case 8:
              return 4.0 * l2 * l3;
            case 9:
              return 4.0 * l0 * l3;
          }
          break;
        case 3:
          switch (i) {
            case 0:
              return 0.5 * l0 * (3.0 * l0 - 1.0) * (3.0 * l0 - 2.0);
            case 1:
              return 0.5 * l1 * (3.0 * l1 - 1.0) * (3.0 * l1 - 2.0);
            case 2:
              return 0.5 * l2 * (3.0 * l2 - 1.0) * (3.0 * l2 - 2.0);
            case 3:
              return 0.5 * l3 * (3.0 * l3 - 1.0) * (3.0 * l3 - 2.0);
            case 4:
              return 4.5 * l0 * l1 * (3.0 * l0 - 1.0);
            case 5:
              return 4.5 * l0 * l1 * (3.0 * l1 - 1.0);
            case 6:
              return 4.5 * l0 * l2 * (3.0 * l0 - 1.0);
            case 7:
              return 4.5 * l0 * l2 * (3.0 * l2 - 1.0);
            case 8:
              return 4.5 * l0 * l3 * (3.0 * l0 - 1.0);
            case 9:
              return 4.5 * l0 * l3 * (3.0 * l3 - 1.0);
            case 10:
              return 4.5 * l1 * l2 * (3.0 * l1 - 1.0);
            case 11:
              return 4.5 * l1 * l2 * (3.0 * l2 - 1.0);
            case 12:
              return 4.5 * l1 * l3 * (3.0 * l1 - 1.0);
            case 13:
              return 4.5 * l1 * l3 * (3.0 * l3 - 1.0);
            case 14:
              return 4.5 * l2 * l3 * (3.0 * l2 - 1.0);
            case 15:
              return 4.5 * l2 * l3 * (3.0 * l3 - 1.0);
            case 16:
              return 27.0 * l0 * l1 * l2;
            case 17:
              return 27.0 * l0 * l1 * l3;
            case 18:
              return 27.0 * l0 * l2 * l3;
            case 19:
              return 27.0 * l1 * l2 * l3;
          }
          break;
      }
    }
    return RealType(0);
  }

  Tensor<1, dim, RealType> eval_gradient(
    unsigned int i,
    const Tensor<1, dim, RealType>& point) const
  {
    Tensor<1, dim, RealType> grad;
    if constexpr (dim == 1) {
      const RealType x = point(0);
      RealType result = RealType(0);
      for (unsigned int k = 0; k <= p_; ++k) {
        if (k == i)
          continue;
        RealType term = RealType(1);
        for (unsigned int j = 0; j <= p_; ++j) {
          if (j == i || j == k)
            continue;
          const RealType xj = static_cast<RealType>(j) / p_;
          const RealType xi = static_cast<RealType>(i) / p_;
          term *= (x - xj) / (xi - xj);
        }
        const RealType xk = static_cast<RealType>(k) / p_;
        const RealType xi = static_cast<RealType>(i) / p_;
        result += term / (xi - xk);
      }
      grad(0) = result;
      return grad;
    }
    if constexpr (dim == 2) {
      const RealType x = point(0);
      const RealType y = point(1);

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
    }
    if constexpr (dim == 3) {
      const RealType x = point(0);
      const RealType y = point(1);
      const RealType z = point(2);
      const RealType l0 = 1.0 - x - y - z;
      const RealType l1 = x, l2 = y, l3 = z;

      // dl_n/d(coord d): dl0/d* = -1, dl1/dx=1, dl2/dy=1, dl3/dz=1
      auto dL = [](int n, int d) -> RealType {
        if (n == 0)
          return RealType(-1);
        return (d == n - 1) ? RealType(1) : RealType(0);
      };
      auto Lv = [&](int n) -> RealType {
        return (n == 0 ? l0 : n == 1 ? l1 : n == 2 ? l2 : l3);
      };

      switch (p_) {
        case 0:
          grad(0) = 0.0;
          grad(1) = 0.0;
          grad(2) = 0.0;
          break;
        case 1:
          for (int d = 0; d < 3; ++d)
            grad(d) = dL(i, d);
          break;
        case 2: {
          // vertex: l*(2l-1), d/d* = (4l-1)*dL
          // edge:   4*la*lb,  d/d* = 4*(la*dLb + lb*dLa)
          auto dVtx = [&](int n) {
            for (int d = 0; d < 3; d++)
              grad(d) = (4.0 * Lv(n) - 1.0) * dL(n, d);
          };
          auto dEdg = [&](int a, int b) {
            for (int d = 0; d < 3; d++)
              grad(d) = 4.0 * (Lv(a) * dL(b, d) + Lv(b) * dL(a, d));
          };
          switch (i) {
            case 0:
              dVtx(0);
              break;
            case 1:
              dVtx(1);
              break;
            case 2:
              dVtx(2);
              break;
            case 3:
              dVtx(3);
              break;
            case 4:
              dEdg(1, 2);
              break;
            case 5:
              dEdg(0, 2);
              break;
            case 6:
              dEdg(0, 1);
              break;
            case 7:
              dEdg(1, 3);
              break;
            case 8:
              dEdg(2, 3);
              break;
            case 9:
              dEdg(0, 3);
              break;
          }
          break;
        }
        case 3: {
          // vertex: 0.5*l*(3l-1)*(3l-2), d/d* = 0.5*(27l^2-18l+2)*dL
          auto dVtx3 = [&](int n) {
            const RealType l = Lv(n);
            const RealType df = 0.5 * (27.0 * l * l - 18.0 * l + 2.0);
            for (int d = 0; d < 3; d++)
              grad(d) = df * dL(n, d);
          };
          // edge type 1: 4.5*la*lb*(3*la-1)
          // d/d* = 4.5*[(3la-1)*lb*dLa + la*lb*3*dLa + la*(3la-1)*dLb]
          //      = 4.5*[lb*(3la-1+3la)*dLa + la*(3la-1)*dLb]
          //      = 4.5*[lb*(6la-1)*dLa + la*(3la-1)*dLb]
          auto dE1 = [&](int a, int b) {
            const RealType la = Lv(a), lb = Lv(b);
            for (int d = 0; d < 3; d++)
              grad(d) = 4.5 * (lb * (6.0 * la - 1.0) * dL(a, d) +
                               la * (3.0 * la - 1.0) * dL(b, d));
          };
          // edge type 2: 4.5*la*lb*(3*lb-1)
          auto dE2 = [&](int a, int b) {
            const RealType la = Lv(a), lb = Lv(b);
            for (int d = 0; d < 3; d++)
              grad(d) = 4.5 * (lb * (3.0 * lb - 1.0) * dL(a, d) +
                               la * (6.0 * lb - 1.0) * dL(b, d));
          };
          // face: 27*la*lb*lc
          auto dFace = [&](int a, int b, int c) {
            const RealType la = Lv(a), lb = Lv(b), lc = Lv(c);
            for (int d = 0; d < 3; d++)
              grad(d) = 27.0 * (lb * lc * dL(a, d) + la * lc * dL(b, d) +
                                la * lb * dL(c, d));
          };
          switch (i) {
            case 0:
              dVtx3(0);
              break;
            case 1:
              dVtx3(1);
              break;
            case 2:
              dVtx3(2);
              break;
            case 3:
              dVtx3(3);
              break;
            case 4:
              dE1(0, 1);
              break;
            case 5:
              dE2(0, 1);
              break;
            case 6:
              dE1(0, 2);
              break;
            case 7:
              dE2(0, 2);
              break;
            case 8:
              dE1(0, 3);
              break;
            case 9:
              dE2(0, 3);
              break;
            case 10:
              dE1(1, 2);
              break;
            case 11:
              dE2(1, 2);
              break;
            case 12:
              dE1(1, 3);
              break;
            case 13:
              dE2(1, 3);
              break;
            case 14:
              dE1(2, 3);
              break;
            case 15:
              dE2(2, 3);
              break;
            case 16:
              dFace(0, 1, 2);
              break;
            case 17:
              dFace(0, 1, 3);
              break;
            case 18:
              dFace(0, 2, 3);
              break;
            case 19:
              dFace(1, 2, 3);
              break;
          }
          break;
        }
      }
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

template<unsigned int dim, unsigned int mesh_q, typename RealType>
class FEValues
{
public:
  FEValues(const FE_DGLagrangeSimplex<dim, RealType>& fe,
           const QGaussSimplex<dim, RealType>& quad)
    : fe_(fe)
    , quad_(quad)
    , mesh_fe_(mesh_q) 
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
    // Number of geometry nodes: 3 (Q1) or 6 (Q2) in 2D,
    //                           4 (Q1) or 10 (Q2) in 3D
    constexpr unsigned int n_mesh_nodes =
      FE_DGLagrangeSimplex<dim, RealType>::n_dofs_per_cell(mesh_q);

    // Collect physical coordinates of all geometry nodes on this cell.
    // cell.vertex(I) works for both corner and midpoint nodes because
    // they are all stored contiguously in tria->cell_vertices.
    RealType mesh_coords[n_mesh_nodes][dim];
    for (unsigned int I = 0; I < n_mesh_nodes; ++I)
      for (unsigned int d = 0; d < dim; ++d)
        mesh_coords[I][d] = cell.vertex(I)(d);

    for (unsigned int q = 0; q < n_q_; ++q) {
      const auto xi = quad_.point(q);

      // ----------------------------------------------------------------
      // Build J(xi) = sum_I  x_I ⊗ ∇N_I(xi)
      // For mesh_q=1 this is constant and identical to the old
      // vertex-differencing formula. For mesh_q=2 it varies per point.
      // ----------------------------------------------------------------
      RealType J[dim][dim] = {};
      for (unsigned int I = 0; I < n_mesh_nodes; ++I) {
        const auto dN = mesh_fe_.shape_gradient(I, xi);
        for (unsigned int i = 0; i < dim; ++i)
          for (unsigned int j = 0; j < dim; ++j)
            J[i][j] += mesh_coords[I][i] * dN(j);
      }

      // ----------------------------------------------------------------
      // Invert J and compute det(J) — same arithmetic as before,
      // now inside the loop so it is re-evaluated at each quad point.
      // ----------------------------------------------------------------
      RealType det_J;
      RealType J_inv[dim][dim];

      if constexpr (dim == 2) {
        det_J = J[0][0] * J[1][1] - J[0][1] * J[1][0];

        J_inv[0][0] =  J[1][1] / det_J;
        J_inv[0][1] = -J[0][1] / det_J;
        J_inv[1][0] = -J[1][0] / det_J;
        J_inv[1][1] =  J[0][0] / det_J;

      } else if constexpr (dim == 3) {
        det_J = J[0][0] * (J[1][1]*J[2][2] - J[1][2]*J[2][1])
              - J[0][1] * (J[1][0]*J[2][2] - J[1][2]*J[2][0])
              + J[0][2] * (J[1][0]*J[2][1] - J[1][1]*J[2][0]);

        J_inv[0][0] = (J[1][1]*J[2][2] - J[1][2]*J[2][1]) / det_J;
        J_inv[0][1] = (J[0][2]*J[2][1] - J[0][1]*J[2][2]) / det_J;
        J_inv[0][2] = (J[0][1]*J[1][2] - J[0][2]*J[1][1]) / det_J;

        J_inv[1][0] = (J[1][2]*J[2][0] - J[1][0]*J[2][2]) / det_J;
        J_inv[1][1] = (J[0][0]*J[2][2] - J[0][2]*J[2][0]) / det_J;
        J_inv[1][2] = (J[0][2]*J[1][0] - J[0][0]*J[1][2]) / det_J;

        J_inv[2][0] = (J[1][0]*J[2][1] - J[1][1]*J[2][0]) / det_J;
        J_inv[2][1] = (J[0][1]*J[2][0] - J[0][0]*J[2][1]) / det_J;
        J_inv[2][2] = (J[0][0]*J[1][1] - J[0][1]*J[1][0]) / det_J;

      } else {
        static_assert(dim == 2 || dim == 3,
                      "Only dim = 2 and dim = 3 are supported");
      }

      // ----------------------------------------------------------------
      // JxW, shape values, physical gradients — unchanged in form
      // ----------------------------------------------------------------
      JxW_(q) = std::abs(det_J) * quad_.weight(q);

      for (unsigned int i = 0; i < n_dofs_; ++i) {
        phi_(i, q) = fe_.shape_value(i, xi);

        const auto tmp = fe_.shape_gradient(i, xi);
        for (unsigned int d = 0; d < dim; ++d) {
          RealType g = 0.0;
          for (unsigned int k = 0; k < dim; ++k)
            g += J_inv[k][d] * tmp(k);
          grad_phi_(i, q, d) = g;
        }
      }

      // ----------------------------------------------------------------
      // Physical quad point: x(xi) = sum_I N_I(xi) * x_I
      // Replaces the old affine  x0 + J*xi  expression.
      // For mesh_q=1 these are identical.
      // ----------------------------------------------------------------
      for (unsigned int d = 0; d < dim; ++d) {
        RealType xq = 0.0;
        for (unsigned int I = 0; I < n_mesh_nodes; ++I)
          xq += mesh_fe_.shape_value(I, xi) * mesh_coords[I][d];
        q_point_(q, d) = xq;
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
  const FE_DGLagrangeSimplex<dim, RealType>& fe_; // solution FE (degree = fe.degree())
  const QGaussSimplex<dim, RealType>& quad_;
  FE_DGLagrangeSimplex<dim, RealType> mesh_fe_; // geometry FE (degree = mesh_q)

  unsigned int n_dofs_;
  unsigned int n_q_;

  Kokkos::View<RealType*, Layout, HostMemSpace> JxW_;        // [q]
  Kokkos::View<RealType**, Layout, HostMemSpace> q_point_;   // [q, dim]
  Kokkos::View<RealType**, Layout, HostMemSpace> phi_;       // [dof, q]
  Kokkos::View<RealType***, Layout, HostMemSpace> grad_phi_; // [dof, q, dim]
};

template<unsigned int dim, unsigned int mesh_q, typename RealType>
class FEFaceValues
{
public:
  FEFaceValues(const FE_DGLagrangeSimplex<dim, RealType>& fe,
               const QGaussSimplex<dim - 1, RealType>& quad)
    : fe_(fe)
    , quad_(quad)
    , mesh_fe_(mesh_q) 
    , n_dofs_(fe.n_dofs())
    , n_q_(quad.n_points())
  {
    JxW_ = Kokkos::View<RealType*, Layout, HostMemSpace>("JxW", n_q_);
    q_point_ = Kokkos::View<RealType**, Layout, HostMemSpace>("q_point", n_q_, dim);
    normal_ = Kokkos::View<RealType**, Layout, HostMemSpace>("normal", n_q_, dim);
    phi_ = Kokkos::View<RealType**, Layout, HostMemSpace>("phi", n_dofs_, n_q_);
    grad_phi_ = Kokkos::View<RealType***, Layout, HostMemSpace>("grad_phi", n_dofs_, n_q_, dim);
  }

  template<typename CellAccessor>
  void reinit(const CellAccessor& cell, unsigned int face)
  {
    ASSERT((face < SimplexTopology<dim, mesh_q>::faces_per_cell),
           "Local face number must be less than the number of faces per cell");

    // We only sort the 'dim' corner vertices to find the canonical orientation.
    // This perfectly supports both q=1 and q=2 (where there are extra midpoint nodes).
    std::array<unsigned int, dim> global_ids;
    for (unsigned int v = 0; v < dim; ++v) {
      global_ids[v] = cell.face(face).vertex_index(v);
    }

    // Canonical = sorted ascending
    std::array<unsigned int, dim> sorted_ids = global_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());

    // Compute the permutation: where does each sorted vertex appear in local order?
    std::array<unsigned int, dim> perm;
    for (unsigned int v = 0; v < dim; ++v) {
      perm[v] = std::find(global_ids.begin(), global_ids.end(), sorted_ids[v]) -
                global_ids.begin();
    }

    // Physical positions of face corners in canonical order
    RealType fv[dim][dim]; // fv[v][coord]
    for (unsigned int v = 0; v < dim; ++v) {
      // perm[v] gives the local face-vertex index corresponding to canonical v
      for (unsigned int d = 0; d < dim; ++d) {
        fv[v][d] = cell.face(face).vertex(perm[v])(d);
      }
    }

    // Build a Jacobian from the vertices of the cell.
    RealType J[dim][dim];
    RealType x0[dim];

    for (unsigned int d = 0; d < dim; ++d) {
      x0[d] = cell.vertex(0)(d);
    }

    // J columns are edge vectors from vertex 0
    for (unsigned int i = 0; i < dim; ++i) {
      for (unsigned int j = 0; j < dim; ++j) {
        J[i][j] = cell.vertex(j + 1)(i) - cell.vertex(0)(i);
      }
    }

    // Take the inverse and determinant
    RealType det_J;
    RealType J_inv[dim][dim];
    if constexpr (dim == 2) {
      det_J = J[0][0] * J[1][1] - J[0][1] * J[1][0];

      J_inv[0][0] =  J[1][1] / det_J;
      J_inv[0][1] = -J[0][1] / det_J;
      J_inv[1][0] = -J[1][0] / det_J;
      J_inv[1][1] =  J[0][0] / det_J;
    } else if constexpr (dim == 3) {
      det_J = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
              J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
              J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

      J_inv[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) / det_J;
      J_inv[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) / det_J;
      J_inv[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) / det_J;

      J_inv[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) / det_J;
      J_inv[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) / det_J;
      J_inv[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) / det_J;

      J_inv[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) / det_J;
      J_inv[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) / det_J;
      J_inv[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) / det_J;
    }

    RealType ref_origin[dim];
    RealType ref_tangent[dim - 1][dim];
    RealType ref_normal[dim];

    // Get reference coordinates of canonical face vertices directly from J_inv. 
    // We must use 'fv[v]' here to respect the permutation mapping backwards!
    RealType xi_fv[dim][dim]; 
    for (unsigned int v = 0; v < dim; ++v) {
      for (unsigned int d = 0; d < dim; ++d) {
        xi_fv[v][d] = 0;
        for (unsigned int k = 0; k < dim; ++k)
          xi_fv[v][d] += J_inv[d][k] * (fv[v][k] - x0[k]);
      }
    }

    // ref_origin = first face vertex in reference space
    for (unsigned int d = 0; d < dim; ++d)
      ref_origin[d] = xi_fv[0][d];

    // ref_tangents from edges of face in reference space
    for (unsigned int s = 0; s < dim - 1; ++s)
      for (unsigned int d = 0; d < dim; ++d)
        ref_tangent[s][d] = xi_fv[s + 1][d] - xi_fv[0][d];

    // Fix: Calculate normal for 2D meshes (orthogonal to the single 1D tangent segment)
    if constexpr (dim == 2) {
      ref_normal[0] = ref_tangent[0][1];
      ref_normal[1] = -ref_tangent[0][0];

      // Ensure outward pointing: dot with (origin - centroid) > 0
      RealType centroid[dim] = {RealType(1.0) / RealType(3.0), RealType(1.0) / RealType(3.0)};
      RealType dot = 0;
      for (unsigned int d = 0; d < dim; ++d)
        dot += ref_normal[d] * (ref_origin[d] - centroid[d]);
      if (dot < 0) {
        ref_normal[0] = -ref_normal[0];
        ref_normal[1] = -ref_normal[1];
      }
    } 
    // Original 3D normal block (cross product of 2D tangents)
    else if constexpr (dim == 3) {
      ref_normal[0] = ref_tangent[0][1] * ref_tangent[1][2] -
                      ref_tangent[0][2] * ref_tangent[1][1];
      ref_normal[1] = ref_tangent[0][2] * ref_tangent[1][0] -
                      ref_tangent[0][0] * ref_tangent[1][2];
      ref_normal[2] = ref_tangent[0][0] * ref_tangent[1][1] -
                      ref_tangent[0][1] * ref_tangent[1][0];
      
      // Ensure outward pointing
      RealType centroid[dim] = {0.25, 0.25, 0.25};
      RealType dot = 0;
      for (unsigned int d = 0; d < dim; ++d)
        dot += ref_normal[d] * (ref_origin[d] - centroid[d]);
      if (dot < 0) {
        for (unsigned int d = 0; d < dim; ++d)
          ref_normal[d] = -ref_normal[d];
      }
    }

    // ref_origin = first face vertex in reference space
    for (unsigned int d = 0; d < dim; ++d)
      ref_origin[d] = xi_fv[0][d];

    // ref_tangents from edges of face in reference space
    for (unsigned int s = 0; s < dim - 1; ++s)
      for (unsigned int d = 0; d < dim; ++d)
        ref_tangent[s][d] = xi_fv[s + 1][d] - xi_fv[0][d];

    // ref_normal = cross product of ref_tangents (3D), pointing outward
    if constexpr (dim == 3) {
      ref_normal[0] = ref_tangent[0][1] * ref_tangent[1][2] -
                      ref_tangent[0][2] * ref_tangent[1][1];
      ref_normal[1] = ref_tangent[0][2] * ref_tangent[1][0] -
                      ref_tangent[0][0] * ref_tangent[1][2];
      ref_normal[2] = ref_tangent[0][0] * ref_tangent[1][1] -
                      ref_tangent[0][1] * ref_tangent[1][0];
      // Ensure outward: dot with (origin - centroid) should be positive
      RealType centroid[dim] = {};
      for (unsigned int d = 0; d < dim; ++d)
        centroid[d] = 0.25; // ref tet centroid
      RealType dot = 0;
      for (unsigned int d = 0; d < dim; ++d)
        dot += ref_normal[d] * (ref_origin[d] - centroid[d]);
      if (dot < 0)
        for (unsigned int d = 0; d < dim; ++d)
          ref_normal[d] = -ref_normal[d];
    }

    // Now grab the physical normal and tangent
    RealType n_phys[dim] = {};
    for (unsigned int d = 0; d < dim; ++d) {
      for (unsigned int k = 0; k < dim; ++k) {
        // Covariant transformation for normal vector
        n_phys[d] += J_inv[k][d] * ref_normal[k];
      }
    }

    RealType t_phys[dim - 1][dim] = {};
    for (unsigned int s = 0; s < dim - 1; ++s) {
      for (unsigned int d = 0; d < dim; ++d) {
        for (unsigned int k = 0; k < dim; ++k) {
          t_phys[s][d] += J[d][k] * ref_tangent[s][k];
        }
      }
    }

    RealType n_phys_norm = 0.0;
    for (unsigned int d = 0; d < dim; ++d)
      n_phys_norm += n_phys[d] * n_phys[d];
    n_phys_norm = std::sqrt(n_phys_norm);

    RealType face_jac = 0.0;
    if constexpr (dim == 2) {
      for (unsigned int d = 0; d < dim; ++d) {
        face_jac += t_phys[0][d] * t_phys[0][d];
      }
      face_jac = std::sqrt(face_jac);
    } else if constexpr (dim == 3) {
      // Cross product of the two physical tangent vectors
      const RealType cx =
        t_phys[0][1] * t_phys[1][2] - t_phys[0][2] * t_phys[1][1];
      const RealType cy =
        t_phys[0][2] * t_phys[1][0] - t_phys[0][0] * t_phys[1][2];
      const RealType cz =
        t_phys[0][0] * t_phys[1][1] - t_phys[0][1] * t_phys[1][0];
      face_jac = std::sqrt(cx * cx + cy * cy + cz * cz);
    }

    for (unsigned int q = 0; q < n_q_; ++q) {
      // Grab the dim - 1 quad points
      const auto xi_face = quad_.point(q);

      // Map dim - 1 to dim in reference space
      RealType xi_ref[dim];
      for (unsigned int d = 0; d < dim; ++d) {
        xi_ref[d] = ref_origin[d];
        for (unsigned int s = 0; s < dim - 1; ++s)
          xi_ref[d] += xi_face(s) * ref_tangent[s][d];
      }

      // Map to physical space
      RealType x_phys[dim];
      for (unsigned int d = 0; d < dim; ++d) {
        x_phys[d] = x0[d];
        for (unsigned int k = 0; k < dim; ++k)
          x_phys[d] += J[d][k] * xi_ref[k];
      }

      // Wrap in a Tensor
      Tensor<1, dim, RealType> xi;
      for (unsigned int d = 0; d < dim; ++d) {
        xi(d) = xi_ref[d];
      }

      JxW_(q) = std::abs(face_jac) * quad_.weight(q);

      for (unsigned int d = 0; d < dim; ++d) {
        q_point_(q, d) = x_phys[d];
        normal_(q, d) = n_phys[d] / n_phys_norm;
      }

      for (unsigned int i = 0; i < n_dofs_; ++i) {
        phi_(i, q) = fe_.shape_value(i, xi);

        const auto tmp = fe_.shape_gradient(i, xi);
        for (unsigned int d = 0; d < dim; ++d) {
          RealType g = 0.0;
          for (unsigned int k = 0; k < dim; ++k) {
            g += J_inv[k][d] * tmp(k);
          }
          grad_phi_(i, q, d) = g;
        }
      }
    }
  }

  unsigned int n_dofs() const { return n_dofs_; }
  unsigned int n_q_points() const { return n_q_; }

  RealType JxW(unsigned int q) { return JxW_(q); }

  Tensor<1, dim, RealType> q_point(unsigned int q)
  {
    Tensor<1, dim, RealType> p;
    for (unsigned int d = 0; d < dim; ++d)
      p(d) = q_point_(q, d);
    return p;
  }

  Tensor<1, dim, RealType> normal(unsigned int q) const
  {
    Tensor<1, dim, RealType> n;
    for (unsigned int d = 0; d < dim; ++d)
      n(d) = normal_(q, d);
    return n;
  }

  RealType shape_value(unsigned int i, unsigned int q) { return phi_(i, q); }

  Tensor<1, dim, RealType> shape_gradient(unsigned int i, unsigned int q)
  {
    Tensor<1, dim, RealType> grad;
    for (unsigned int d = 0; d < dim; ++d)
      grad(d) = grad_phi_(i, q, d);
    return grad;
  }

private:
  const FE_DGLagrangeSimplex<dim, RealType>& fe_;
  const QGaussSimplex<dim - 1, RealType>& quad_;
  FE_DGLagrangeSimplex<dim, RealType> mesh_fe_;

  unsigned int n_dofs_;
  unsigned int n_q_;

  Kokkos::View<RealType*, Layout, HostMemSpace> JxW_;
  Kokkos::View<RealType**, Layout, HostMemSpace> q_point_;
  Kokkos::View<RealType**, Layout, HostMemSpace> normal_;
  Kokkos::View<RealType**, Layout, HostMemSpace> phi_;
  Kokkos::View<RealType***, Layout, HostMemSpace> grad_phi_;
};