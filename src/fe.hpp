#pragma once

#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <iostream>
#include <tensor.hpp>
#include <triangulation.hpp>

template<unsigned int dim, typename RealType>
class FE_DGQLegendre
{
public:
  static constexpr unsigned int n_dofs_per_cell(unsigned int p)
  {
<<<<<<< HEAD
    if constexpr (dim == 1) return p + 1;
    if constexpr (dim == 2) return (p + 1) * (p + 2) / 2;
    if constexpr (dim == 3) return (p + 1) * (p + 2) * (p + 3) / 6;
=======
    if constexpr (dim == 1) {
      return p + 1;
    }
    if constexpr (dim == 2) {
      return (p + 1) * (p + 2) / 2;
    }
    if constexpr (dim == 3) {
      return (p + 1) * (p + 2) * (p + 3) / 6;
    }
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
    return 0;
  }

  static constexpr unsigned int max_degree_ = 3;
<<<<<<< HEAD
  static constexpr unsigned int max_dofs_   = 20; // p=3 tet has 20 dofs
=======
  // 3D p=3: (4)(5)(6)/6 = 20 dofs
  static constexpr unsigned int max_dofs_ = 20;
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))

  explicit FE_DGQLegendre(const unsigned int p)
    : p_(p)
    , n_dofs_(n_dofs_per_cell(p))
  {
    ASSERT(p <= max_degree_, "Polynomial degree exceeds maximum");
    init_nodes();
  }

  unsigned int degree()  const { return p_; }
  unsigned int n_dofs()  const { return n_dofs_; }

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
  RealType nodes_[max_dofs_][dim == 0 ? 1 : dim];

<<<<<<< HEAD
  // ── node initialisation ──────────────────────────────────────────────────

  void init_nodes()
  {
    if constexpr (dim == 1) {
      // Equally-spaced Lagrange nodes on [0,1]
      for (unsigned int i = 0; i <= p_; ++i)
        nodes_[i][0] = (p_ == 0) ? RealType(0.5)
                                  : static_cast<RealType>(i) / p_;
    }

    if constexpr (dim == 2) {
      switch (p_) {
        case 0:
          nodes_[0][0] = 1.0/3.0; nodes_[0][1] = 1.0/3.0;
=======
  RealType nodes_[max_dofs_][dim > 0 ? dim : 1];

  static RealType fixed_pow(RealType x, int n)
  {
    RealType r = RealType(1);
    for (int i = 0; i < n; ++i)
      r *= x;
    return r;
  }

  void init_nodes()
  {
    if constexpr (dim == 2) {
      switch (p_) {
        case 0:
          nodes_[0][0] = 1.0 / 3.0;
          nodes_[0][1] = 1.0 / 3.0;
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
          break;
        case 1:
          nodes_[0][0] = 0.0; nodes_[0][1] = 0.0;
          nodes_[1][0] = 1.0; nodes_[1][1] = 0.0;
          nodes_[2][0] = 0.0; nodes_[2][1] = 1.0;
          break;
        case 2:
          nodes_[0][0] = 0.0; nodes_[0][1] = 0.0;
          nodes_[1][0] = 1.0; nodes_[1][1] = 0.0;
          nodes_[2][0] = 0.0; nodes_[2][1] = 1.0;
          nodes_[3][0] = 0.5; nodes_[3][1] = 0.5;
          nodes_[4][0] = 0.0; nodes_[4][1] = 0.5;
          nodes_[5][0] = 0.5; nodes_[5][1] = 0.0;
          break;
        case 3:
          nodes_[0][0] = 0.0;       nodes_[0][1] = 0.0;
          nodes_[1][0] = 1.0;       nodes_[1][1] = 0.0;
          nodes_[2][0] = 0.0;       nodes_[2][1] = 1.0;
<<<<<<< HEAD
          nodes_[3][0] = 2.0/3.0;   nodes_[3][1] = 1.0/3.0;
          nodes_[4][0] = 1.0/3.0;   nodes_[4][1] = 2.0/3.0;
          nodes_[5][0] = 0.0;       nodes_[5][1] = 2.0/3.0;
          nodes_[6][0] = 0.0;       nodes_[6][1] = 1.0/3.0;
          nodes_[7][0] = 1.0/3.0;   nodes_[7][1] = 0.0;
          nodes_[8][0] = 2.0/3.0;   nodes_[8][1] = 0.0;
          nodes_[9][0] = 1.0/3.0;   nodes_[9][1] = 1.0/3.0;
          break;
      }
      check_basis();
    }

    if constexpr (dim == 3) {
      switch (p_) {
        case 0:
          // Centroid of reference tet
          nodes_[0][0] = 0.25; nodes_[0][1] = 0.25; nodes_[0][2] = 0.25;
          break;
        case 1:
          // Four vertices of reference tet
=======
          nodes_[3][0] = 2.0 / 3.0; nodes_[3][1] = 1.0 / 3.0;
          nodes_[4][0] = 1.0 / 3.0; nodes_[4][1] = 2.0 / 3.0;
          nodes_[5][0] = 0.0;       nodes_[5][1] = 2.0 / 3.0;
          nodes_[6][0] = 0.0;       nodes_[6][1] = 1.0 / 3.0;
          nodes_[7][0] = 1.0 / 3.0; nodes_[7][1] = 0.0;
          nodes_[8][0] = 2.0 / 3.0; nodes_[8][1] = 0.0;
          nodes_[9][0] = 1.0 / 3.0; nodes_[9][1] = 1.0 / 3.0;
          break;
      }
    }
    if constexpr (dim == 3) {
      // Reference tet: v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), v3=(0,0,1)
      switch (p_) {
        case 0:
          nodes_[0][0] = 0.25; nodes_[0][1] = 0.25; nodes_[0][2] = 0.25;
          break;
        case 1:
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
          nodes_[0][0] = 0.0; nodes_[0][1] = 0.0; nodes_[0][2] = 0.0;
          nodes_[1][0] = 1.0; nodes_[1][1] = 0.0; nodes_[1][2] = 0.0;
          nodes_[2][0] = 0.0; nodes_[2][1] = 1.0; nodes_[2][2] = 0.0;
          nodes_[3][0] = 0.0; nodes_[3][1] = 0.0; nodes_[3][2] = 1.0;
          break;
        case 2:
<<<<<<< HEAD
          // 10 nodes: 4 vertices + 6 edge midpoints
          // Vertices
=======
          // 4 vertices + 6 edge midpoints = 10 nodes
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
          nodes_[0][0] = 0.0; nodes_[0][1] = 0.0; nodes_[0][2] = 0.0;
          nodes_[1][0] = 1.0; nodes_[1][1] = 0.0; nodes_[1][2] = 0.0;
          nodes_[2][0] = 0.0; nodes_[2][1] = 1.0; nodes_[2][2] = 0.0;
          nodes_[3][0] = 0.0; nodes_[3][1] = 0.0; nodes_[3][2] = 1.0;
<<<<<<< HEAD
          // Edge midpoints
          nodes_[4][0] = 0.5; nodes_[4][1] = 0.0; nodes_[4][2] = 0.0; // edge 01
          nodes_[5][0] = 0.0; nodes_[5][1] = 0.5; nodes_[5][2] = 0.0; // edge 02... 
          nodes_[6][0] = 0.0; nodes_[6][1] = 0.0; nodes_[6][2] = 0.5; // edge 03
          nodes_[7][0] = 0.5; nodes_[7][1] = 0.5; nodes_[7][2] = 0.0; // edge 12... 
          nodes_[8][0] = 0.0; nodes_[8][1] = 0.5; nodes_[8][2] = 0.5; // edge 13... 
          nodes_[9][0] = 0.5; nodes_[9][1] = 0.0; nodes_[9][2] = 0.5; // edge 23... 
          break;
        case 3:
          // 20 nodes: 4 vertices + 12 edge-third-points + 4 face centroids
          // Vertices
          nodes_[0][0]  = 0.0;     nodes_[0][1]  = 0.0;     nodes_[0][2]  = 0.0;
          nodes_[1][0]  = 1.0;     nodes_[1][1]  = 0.0;     nodes_[1][2]  = 0.0;
          nodes_[2][0]  = 0.0;     nodes_[2][1]  = 1.0;     nodes_[2][2]  = 0.0;
          nodes_[3][0]  = 0.0;     nodes_[3][1]  = 0.0;     nodes_[3][2]  = 1.0;
          // Edge v0-v1: 1/3, 2/3
          nodes_[4][0]  = 1.0/3.0; nodes_[4][1]  = 0.0;     nodes_[4][2]  = 0.0;
          nodes_[5][0]  = 2.0/3.0; nodes_[5][1]  = 0.0;     nodes_[5][2]  = 0.0;
          // Edge v0-v2: 1/3, 2/3
          nodes_[6][0]  = 0.0;     nodes_[6][1]  = 1.0/3.0; nodes_[6][2]  = 0.0;
          nodes_[7][0]  = 0.0;     nodes_[7][1]  = 2.0/3.0; nodes_[7][2]  = 0.0;
          // Edge v0-v3: 1/3, 2/3
          nodes_[8][0]  = 0.0;     nodes_[8][1]  = 0.0;     nodes_[8][2]  = 1.0/3.0;
          nodes_[9][0]  = 0.0;     nodes_[9][1]  = 0.0;     nodes_[9][2]  = 2.0/3.0;
          // Edge v1-v2: from (1,0,0) toward (0,1,0)
          nodes_[10][0] = 2.0/3.0; nodes_[10][1] = 1.0/3.0; nodes_[10][2] = 0.0;
          nodes_[11][0] = 1.0/3.0; nodes_[11][1] = 2.0/3.0; nodes_[11][2] = 0.0;
          // Edge v1-v3: from (1,0,0) toward (0,0,1)
          nodes_[12][0] = 2.0/3.0; nodes_[12][1] = 0.0;     nodes_[12][2] = 1.0/3.0;
          nodes_[13][0] = 1.0/3.0; nodes_[13][1] = 0.0;     nodes_[13][2] = 2.0/3.0;
          // Edge v2-v3: from (0,1,0) toward (0,0,1)
          nodes_[14][0] = 0.0;     nodes_[14][1] = 2.0/3.0; nodes_[14][2] = 1.0/3.0;
          nodes_[15][0] = 0.0;     nodes_[15][1] = 1.0/3.0; nodes_[15][2] = 2.0/3.0;
          // Face centroids (interior to each face)
          // Face opp v3 (z=0): centroid (1/3,1/3,0)
          nodes_[16][0] = 1.0/3.0; nodes_[16][1] = 1.0/3.0; nodes_[16][2] = 0.0;
          // Face opp v2 (y=0): centroid (1/3,0,1/3)
          nodes_[17][0] = 1.0/3.0; nodes_[17][1] = 0.0;     nodes_[17][2] = 1.0/3.0;
          // Face opp v1 (x=0): centroid (0,1/3,1/3)
          nodes_[18][0] = 0.0;     nodes_[18][1] = 1.0/3.0; nodes_[18][2] = 1.0/3.0;
          // Face opp v0 (x+y+z=1): centroid (1/3,1/3,1/3)
          nodes_[19][0] = 1.0/3.0; nodes_[19][1] = 1.0/3.0; nodes_[19][2] = 1.0/3.0;
          break;
      }
      check_basis();
=======
          nodes_[4][0] = 0.5; nodes_[4][1] = 0.5; nodes_[4][2] = 0.0; // mid v1-v2
          nodes_[5][0] = 0.0; nodes_[5][1] = 0.5; nodes_[5][2] = 0.0; // mid v0-v2
          nodes_[6][0] = 0.5; nodes_[6][1] = 0.0; nodes_[6][2] = 0.0; // mid v0-v1
          nodes_[7][0] = 0.5; nodes_[7][1] = 0.0; nodes_[7][2] = 0.5; // mid v1-v3
          nodes_[8][0] = 0.0; nodes_[8][1] = 0.5; nodes_[8][2] = 0.5; // mid v2-v3
          nodes_[9][0] = 0.0; nodes_[9][1] = 0.0; nodes_[9][2] = 0.5; // mid v0-v3
          break;
        case 3:
          // 4 vertices + 12 edge-third points + 4 face points = 20 nodes
          nodes_[0][0]  = 0.0;       nodes_[0][1]  = 0.0;       nodes_[0][2]  = 0.0;
          nodes_[1][0]  = 1.0;       nodes_[1][1]  = 0.0;       nodes_[1][2]  = 0.0;
          nodes_[2][0]  = 0.0;       nodes_[2][1]  = 1.0;       nodes_[2][2]  = 0.0;
          nodes_[3][0]  = 0.0;       nodes_[3][1]  = 0.0;       nodes_[3][2]  = 1.0;
          // edge v0-v1 thirds
          nodes_[4][0]  = 1.0/3.0;   nodes_[4][1]  = 0.0;       nodes_[4][2]  = 0.0;
          nodes_[5][0]  = 2.0/3.0;   nodes_[5][1]  = 0.0;       nodes_[5][2]  = 0.0;
          // edge v0-v2 thirds
          nodes_[6][0]  = 0.0;       nodes_[6][1]  = 1.0/3.0;   nodes_[6][2]  = 0.0;
          nodes_[7][0]  = 0.0;       nodes_[7][1]  = 2.0/3.0;   nodes_[7][2]  = 0.0;
          // edge v0-v3 thirds
          nodes_[8][0]  = 0.0;       nodes_[8][1]  = 0.0;       nodes_[8][2]  = 1.0/3.0;
          nodes_[9][0]  = 0.0;       nodes_[9][1]  = 0.0;       nodes_[9][2]  = 2.0/3.0;
          // edge v1-v2 thirds
          nodes_[10][0] = 2.0/3.0;   nodes_[10][1] = 1.0/3.0;   nodes_[10][2] = 0.0;
          nodes_[11][0] = 1.0/3.0;   nodes_[11][1] = 2.0/3.0;   nodes_[11][2] = 0.0;
          // edge v1-v3 thirds
          nodes_[12][0] = 2.0/3.0;   nodes_[12][1] = 0.0;       nodes_[12][2] = 1.0/3.0;
          nodes_[13][0] = 1.0/3.0;   nodes_[13][1] = 0.0;       nodes_[13][2] = 2.0/3.0;
          // edge v2-v3 thirds
          nodes_[14][0] = 0.0;       nodes_[14][1] = 2.0/3.0;   nodes_[14][2] = 1.0/3.0;
          nodes_[15][0] = 0.0;       nodes_[15][1] = 1.0/3.0;   nodes_[15][2] = 2.0/3.0;
          // face centroids
          nodes_[16][0] = 1.0/3.0;   nodes_[16][1] = 1.0/3.0;   nodes_[16][2] = 0.0;
          nodes_[17][0] = 1.0/3.0;   nodes_[17][1] = 0.0;       nodes_[17][2] = 1.0/3.0;
          nodes_[18][0] = 0.0;       nodes_[18][1] = 1.0/3.0;   nodes_[18][2] = 1.0/3.0;
          nodes_[19][0] = 1.0/3.0;   nodes_[19][1] = 1.0/3.0;   nodes_[19][2] = 1.0/3.0;
          break;
      }
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
    }
  }

  // ── basis evaluation ─────────────────────────────────────────────────────

  RealType eval(unsigned int i, const Tensor<1, dim, RealType>& point) const
  {
<<<<<<< HEAD
    if constexpr (dim == 1) {
      // Lagrange basis through equally-spaced nodes
      const RealType x = point(0);
      RealType result = RealType(1);
      for (unsigned int j = 0; j <= p_; ++j) {
        if (j == i) continue;
        const RealType xj = (p_ == 0) ? RealType(0.5)
                                      : static_cast<RealType>(j) / p_;
        const RealType xi = (p_ == 0) ? RealType(0.5)
                                      : static_cast<RealType>(i) / p_;
        result *= (x - xj) / (xi - xj);
      }
      return result;
    }

=======
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
    if constexpr (dim == 2) {
      const RealType x = point(0);
      const RealType y = point(1);
      switch (p_) {
<<<<<<< HEAD
        case 0: return RealType(1);
=======
        case 0:
          return RealType(1);
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
        case 1:
          switch (i) {
            case 0: return 1.0 - x - y;
            case 1: return x;
            case 2: return y;
          }
<<<<<<< HEAD
=======
          break;
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
        case 2:
          switch (i) {
            case 0: return 1.0 - 3.0*x - 3.0*y + 2.0*x*x + 4.0*x*y + 2.0*y*y;
            case 1: return -x + 2.0*x*x;
            case 2: return -y + 2.0*y*y;
            case 3: return 4.0*x*y;
            case 4: return 4.0*y - 4.0*x*y - 4.0*y*y;
            case 5: return 4.0*x - 4.0*x*x - 4.0*x*y;
          }
<<<<<<< HEAD
        case 3:
          switch (i) {
            case 0: return 1.0 - 11.0/2.0*x - 11.0/2.0*y + 9.0*x*x + 18.0*x*y + 9.0*y*y - 9.0/2.0*x*x*x - 27.0/2.0*x*x*y - 27.0/2.0*x*y*y - 9.0/2.0*y*y*y;
=======
          break;
        case 3:
          switch (i) {
            case 0: return 1.0 - 11.0/2.0*x - 11.0/2.0*y + 9.0*x*x + 18.0*x*y + 9.0*y*y
                           - 9.0/2.0*x*x*x - 27.0/2.0*x*x*y - 27.0/2.0*x*y*y - 9.0/2.0*y*y*y;
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
            case 1: return x - 9.0/2.0*x*x + 9.0/2.0*x*x*x;
            case 2: return y - 9.0/2.0*y*y + 9.0/2.0*y*y*y;
            case 3: return -9.0/2.0*x*y + 27.0/2.0*x*x*y;
            case 4: return -9.0/2.0*x*y + 27.0/2.0*x*y*y;
            case 5: return -9.0/2.0*y + 9.0/2.0*x*y + 18.0*y*y - 27.0/2.0*x*y*y - 27.0/2.0*y*y*y;
            case 6: return 9.0*y - 45.0/2.0*x*y - 45.0/2.0*y*y + 27.0/2.0*x*x*y + 27.0*x*y*y + 27.0/2.0*y*y*y;
            case 7: return 9.0*x - 45.0/2.0*x*x - 45.0/2.0*x*y + 27.0/2.0*x*x*x + 27.0*x*x*y + 27.0/2.0*x*y*y;
            case 8: return -9.0/2.0*x + 18.0*x*x + 9.0/2.0*x*y - 27.0/2.0*x*x*x - 27.0/2.0*x*x*y;
            case 9: return 27.0*x*y - 27.0*x*x*y - 27.0*x*y*y;
          }
<<<<<<< HEAD
      }
    }

=======
          break;
      }
    }
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
    if constexpr (dim == 3) {
      const RealType x = point(0);
      const RealType y = point(1);
      const RealType z = point(2);
<<<<<<< HEAD
      // Barycentric coords: L0=1-x-y-z, L1=x, L2=y, L3=z
      const RealType L0 = 1.0 - x - y - z;
      const RealType L1 = x;
      const RealType L2 = y;
      const RealType L3 = z;
      switch (p_) {
        case 0: return RealType(1);
        case 1:
          switch (i) {
            case 0: return L0;
            case 1: return L1;
            case 2: return L2;
            case 3: return L3;
          }
        case 2:
          // Serendipity quadratic: phi_i = L_i*(2*L_i - 1) for vertices,
          // phi_ij = 4*L_i*L_j for edge midpoints
          switch (i) {
            case 0: return L0*(2.0*L0 - 1.0);
            case 1: return L1*(2.0*L1 - 1.0);
            case 2: return L2*(2.0*L2 - 1.0);
            case 3: return L3*(2.0*L3 - 1.0);
            case 4: return 4.0*L0*L1; // edge 01
            case 5: return 4.0*L0*L2; // edge 02
            case 6: return 4.0*L0*L3; // edge 03
            case 7: return 4.0*L1*L2; // edge 12
            case 8: return 4.0*L2*L3; // edge 23
            case 9: return 4.0*L1*L3; // edge 13
          }
        case 3: {
          // Cubic: using the standard cubic tetrahedral basis
          // phi_i = (1/2)*L_i*(3*L_i-1)*(3*L_i-2)  for vertices
          // phi_ij^a = (9/2)*L_i*L_j*(3*L_i-1)     for edge nodes (closer to i)
          // phi_ij^b = (9/2)*L_i*L_j*(3*L_j-1)     for edge nodes (closer to j)
          // phi_ijk  = 27*L_i*L_j*L_k               for face nodes
          switch (i) {
            // Vertex nodes
            case 0:  return 0.5*L0*(3.0*L0-1.0)*(3.0*L0-2.0);
            case 1:  return 0.5*L1*(3.0*L1-1.0)*(3.0*L1-2.0);
            case 2:  return 0.5*L2*(3.0*L2-1.0)*(3.0*L2-2.0);
            case 3:  return 0.5*L3*(3.0*L3-1.0)*(3.0*L3-2.0);
            // Edge v0-v1 nodes (4: closer v0, 5: closer v1)
            case 4:  return 4.5*L0*L1*(3.0*L0-1.0);
            case 5:  return 4.5*L0*L1*(3.0*L1-1.0);
            // Edge v0-v2 nodes
            case 6:  return 4.5*L0*L2*(3.0*L0-1.0);
            case 7:  return 4.5*L0*L2*(3.0*L2-1.0);
            // Edge v0-v3 nodes
            case 8:  return 4.5*L0*L3*(3.0*L0-1.0);
            case 9:  return 4.5*L0*L3*(3.0*L3-1.0);
            // Edge v1-v2 nodes (10: closer v1, 11: closer v2)
            case 10: return 4.5*L1*L2*(3.0*L1-1.0);
            case 11: return 4.5*L1*L2*(3.0*L2-1.0);
            // Edge v1-v3 nodes
            case 12: return 4.5*L1*L3*(3.0*L1-1.0);
            case 13: return 4.5*L1*L3*(3.0*L3-1.0);
            // Edge v2-v3 nodes
            case 14: return 4.5*L2*L3*(3.0*L2-1.0);
            case 15: return 4.5*L2*L3*(3.0*L3-1.0);
            // Face nodes (opposite to each vertex)
            case 16: return 27.0*L0*L1*L2; // face opp v3
            case 17: return 27.0*L0*L1*L3; // face opp v2
            case 18: return 27.0*L0*L2*L3; // face opp v1
            case 19: return 27.0*L1*L2*L3; // face opp v0
          }
        }
=======
      const RealType l0 = 1.0 - x - y - z;
      const RealType l1 = x, l2 = y, l3 = z;
      switch (p_) {
        case 0:
          return RealType(1);
        case 1:
          switch (i) {
            case 0: return l0;
            case 1: return l1;
            case 2: return l2;
            case 3: return l3;
          }
          break;
        case 2:
          switch (i) {
            case 0: return l0*(2.0*l0 - 1.0);
            case 1: return l1*(2.0*l1 - 1.0);
            case 2: return l2*(2.0*l2 - 1.0);
            case 3: return l3*(2.0*l3 - 1.0);
            case 4: return 4.0*l1*l2;
            case 5: return 4.0*l0*l2;
            case 6: return 4.0*l0*l1;
            case 7: return 4.0*l1*l3;
            case 8: return 4.0*l2*l3;
            case 9: return 4.0*l0*l3;
          }
          break;
        case 3:
          switch (i) {
            case 0:  return 0.5*l0*(3.0*l0 - 1.0)*(3.0*l0 - 2.0);
            case 1:  return 0.5*l1*(3.0*l1 - 1.0)*(3.0*l1 - 2.0);
            case 2:  return 0.5*l2*(3.0*l2 - 1.0)*(3.0*l2 - 2.0);
            case 3:  return 0.5*l3*(3.0*l3 - 1.0)*(3.0*l3 - 2.0);
            case 4:  return 4.5*l0*l1*(3.0*l0 - 1.0);
            case 5:  return 4.5*l0*l1*(3.0*l1 - 1.0);
            case 6:  return 4.5*l0*l2*(3.0*l0 - 1.0);
            case 7:  return 4.5*l0*l2*(3.0*l2 - 1.0);
            case 8:  return 4.5*l0*l3*(3.0*l0 - 1.0);
            case 9:  return 4.5*l0*l3*(3.0*l3 - 1.0);
            case 10: return 4.5*l1*l2*(3.0*l1 - 1.0);
            case 11: return 4.5*l1*l2*(3.0*l2 - 1.0);
            case 12: return 4.5*l1*l3*(3.0*l1 - 1.0);
            case 13: return 4.5*l1*l3*(3.0*l3 - 1.0);
            case 14: return 4.5*l2*l3*(3.0*l2 - 1.0);
            case 15: return 4.5*l2*l3*(3.0*l3 - 1.0);
            case 16: return 27.0*l0*l1*l2;
            case 17: return 27.0*l0*l1*l3;
            case 18: return 27.0*l0*l2*l3;
            case 19: return 27.0*l1*l2*l3;
          }
          break;
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
      }
    }
    return RealType(0);
  }

  // ── gradient evaluation ───────────────────────────────────────────────────

  Tensor<1, dim, RealType> eval_gradient(
    unsigned int i,
    const Tensor<1, dim, RealType>& point) const
  {
    Tensor<1, dim, RealType> grad;

<<<<<<< HEAD
    if constexpr (dim == 1) {
      const RealType x = point(0);
      RealType result = RealType(0);
      for (unsigned int k = 0; k <= p_; ++k) {
        if (k == i) continue;
        RealType term = RealType(1);
        for (unsigned int j = 0; j <= p_; ++j) {
          if (j == i || j == k) continue;
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
=======
    if constexpr (dim == 2) {
      const RealType x = point(0);
      const RealType y = point(1);

      switch (p_) {
        case 0:
          grad(0) = 0.0; grad(1) = 0.0;
          break;
        case 1:
          switch (i) {
            case 0: grad(0) = -1.0; grad(1) = -1.0; break;
            case 1: grad(0) =  1.0; grad(1) =  0.0; break;
            case 2: grad(0) =  0.0; grad(1) =  1.0; break;
          }
          break;
        case 2:
          switch (i) {
            case 0: grad(0) = -3.0 + 4.0*x + 4.0*y; grad(1) = -3.0 + 4.0*x + 4.0*y; break;
            case 1: grad(0) = -1.0 + 4.0*x;          grad(1) = 0.0;                   break;
            case 2: grad(0) = 0.0;                    grad(1) = -1.0 + 4.0*y;          break;
            case 3: grad(0) = 4.0*y;                  grad(1) = 4.0*x;                 break;
            case 4: grad(0) = -4.0*y;                 grad(1) = 4.0 - 4.0*x - 8.0*y;  break;
            case 5: grad(0) = 4.0 - 8.0*x - 4.0*y;   grad(1) = -4.0*x;                break;
          }
          break;
        case 3:
          switch (i) {
            case 0:
              grad(0) = -11.0/2.0 + 18.0*x + 18.0*y - 27.0/2.0*x*x - 27.0*x*y - 27.0/2.0*y*y;
              grad(1) = -11.0/2.0 + 18.0*x + 18.0*y - 27.0/2.0*x*x - 27.0*x*y - 27.0/2.0*y*y;
              break;
            case 1: grad(0) = 1.0 - 9.0*x + 27.0/2.0*x*x; grad(1) = 0.0; break;
            case 2: grad(0) = 0.0; grad(1) = 1.0 - 9.0*y + 27.0/2.0*y*y; break;
            case 3:
              grad(0) = -9.0/2.0*y + 27.0*x*y;
              grad(1) = -9.0/2.0*x + 27.0/2.0*x*x;
              break;
            case 4:
              grad(0) = -9.0/2.0*y + 27.0/2.0*y*y;
              grad(1) = -9.0/2.0*x + 27.0*x*y;
              break;
            case 5:
              grad(0) = 9.0/2.0*y - 27.0/2.0*y*y;
              grad(1) = -9.0/2.0 + 9.0/2.0*x + 36.0*y - 27.0*x*y - 81.0/2.0*y*y;
              break;
            case 6:
              grad(0) = -45.0/2.0*y + 27.0*x*y + 27.0*y*y;
              grad(1) = 9.0 - 45.0/2.0*x - 45.0*y + 27.0/2.0*x*x + 54.0*x*y + 81.0/2.0*y*y;
              break;
            case 7:
              grad(0) = 9.0 - 45.0*x - 45.0/2.0*y + 81.0/2.0*x*x + 54.0*x*y + 27.0/2.0*y*y;
              grad(1) = -45.0/2.0*x + 27.0*x*x + 27.0*x*y;
              break;
            case 8:
              grad(0) = -9.0/2.0 + 36.0*x + 9.0/2.0*y - 81.0/2.0*x*x - 27.0*x*y;
              grad(1) = 9.0/2.0*x - 27.0/2.0*x*x;
              break;
            case 9:
              grad(0) = 27.0*y - 54.0*x*y - 27.0*y*y;
              grad(1) = 27.0*x - 27.0*x*x - 54.0*x*y;
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
        if (n == 0) return RealType(-1);
        return (d == n - 1) ? RealType(1) : RealType(0);
      };
      auto Lv = [&](int n) -> RealType {
        return (n==0?l0:n==1?l1:n==2?l2:l3);
      };

      switch (p_) {
        case 0:
          grad(0) = 0.0; grad(1) = 0.0; grad(2) = 0.0;
          break;
        case 1:
          for (int d = 0; d < 3; ++d) grad(d) = dL(i, d);
          break;
        case 2: {
          // vertex: l*(2l-1), d/d* = (4l-1)*dL
          // edge:   4*la*lb,  d/d* = 4*(la*dLb + lb*dLa)
          auto dVtx = [&](int n) {
            for(int d=0;d<3;d++) grad(d) = (4.0*Lv(n)-1.0)*dL(n,d);
          };
          auto dEdg = [&](int a, int b) {
            for(int d=0;d<3;d++) grad(d) = 4.0*(Lv(a)*dL(b,d)+Lv(b)*dL(a,d));
          };
          switch (i) {
            case 0: dVtx(0); break; case 1: dVtx(1); break;
            case 2: dVtx(2); break; case 3: dVtx(3); break;
            case 4: dEdg(1,2); break; case 5: dEdg(0,2); break;
            case 6: dEdg(0,1); break; case 7: dEdg(1,3); break;
            case 8: dEdg(2,3); break; case 9: dEdg(0,3); break;
          }
          break;
        }
        case 3: {
          // vertex: 0.5*l*(3l-1)*(3l-2), d/d* = 0.5*(27l^2-18l+2)*dL
          auto dVtx3 = [&](int n) {
            const RealType l = Lv(n);
            const RealType df = 0.5*(27.0*l*l - 18.0*l + 2.0);
            for(int d=0;d<3;d++) grad(d) = df*dL(n,d);
          };
          // edge type 1: 4.5*la*lb*(3*la-1)
          // d/d* = 4.5*[(3la-1)*lb*dLa + la*lb*3*dLa + la*(3la-1)*dLb]
          //      = 4.5*[lb*(3la-1+3la)*dLa + la*(3la-1)*dLb]
          //      = 4.5*[lb*(6la-1)*dLa + la*(3la-1)*dLb]
          auto dE1 = [&](int a, int b) {
            const RealType la=Lv(a), lb=Lv(b);
            for(int d=0;d<3;d++)
              grad(d) = 4.5*(lb*(6.0*la-1.0)*dL(a,d) + la*(3.0*la-1.0)*dL(b,d));
          };
          // edge type 2: 4.5*la*lb*(3*lb-1)
          auto dE2 = [&](int a, int b) {
            const RealType la=Lv(a), lb=Lv(b);
            for(int d=0;d<3;d++)
              grad(d) = 4.5*(lb*(3.0*lb-1.0)*dL(a,d) + la*(6.0*lb-1.0)*dL(b,d));
          };
          // face: 27*la*lb*lc
          auto dFace = [&](int a, int b, int c) {
            const RealType la=Lv(a), lb=Lv(b), lc=Lv(c);
            for(int d=0;d<3;d++)
              grad(d) = 27.0*(lb*lc*dL(a,d) + la*lc*dL(b,d) + la*lb*dL(c,d));
          };
          switch (i) {
            case 0:  dVtx3(0); break; case 1:  dVtx3(1); break;
            case 2:  dVtx3(2); break; case 3:  dVtx3(3); break;
            case 4:  dE1(0,1); break; case 5:  dE2(0,1); break;
            case 6:  dE1(0,2); break; case 7:  dE2(0,2); break;
            case 8:  dE1(0,3); break; case 9:  dE2(0,3); break;
            case 10: dE1(1,2); break; case 11: dE2(1,2); break;
            case 12: dE1(1,3); break; case 13: dE2(1,3); break;
            case 14: dE1(2,3); break; case 15: dE2(2,3); break;
            case 16: dFace(0,1,2); break; case 17: dFace(0,1,3); break;
            case 18: dFace(0,2,3); break; case 19: dFace(1,2,3); break;
          }
          break;
        }
      }
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
    }

    if constexpr (dim == 2) {
      const RealType x = point(0);
      const RealType y = point(1);
      switch (p_) {
        case 0: grad(0) = 0.0; grad(1) = 0.0; break;
        case 1:
          switch (i) {
            case 0: grad(0) = -1.0; grad(1) = -1.0; break;
            case 1: grad(0) =  1.0; grad(1) =  0.0; break;
            case 2: grad(0) =  0.0; grad(1) =  1.0; break;
          } break;
        case 2:
          switch (i) {
            case 0: grad(0) = -3.0+4.0*x+4.0*y;     grad(1) = -3.0+4.0*x+4.0*y;     break;
            case 1: grad(0) = -1.0+4.0*x;            grad(1) =  0.0;                  break;
            case 2: grad(0) =  0.0;                   grad(1) = -1.0+4.0*y;            break;
            case 3: grad(0) =  4.0*y;                 grad(1) =  4.0*x;                break;
            case 4: grad(0) = -4.0*y;                 grad(1) =  4.0-4.0*x-8.0*y;     break;
            case 5: grad(0) =  4.0-8.0*x-4.0*y;      grad(1) = -4.0*x;                break;
          } break;
        case 3:
          switch (i) {
            case 0: grad(0) = -11.0/2.0+18.0*x+18.0*y-27.0/2.0*x*x-27.0*x*y-27.0/2.0*y*y; grad(1) = -11.0/2.0+18.0*x+18.0*y-27.0/2.0*x*x-27.0*x*y-27.0/2.0*y*y; break;
            case 1: grad(0) = 1.0-9.0*x+27.0/2.0*x*x;                                       grad(1) = 0.0; break;
            case 2: grad(0) = 0.0;                                                             grad(1) = 1.0-9.0*y+27.0/2.0*y*y; break;
            case 3: grad(0) = -9.0/2.0*y+27.0*x*y;                                           grad(1) = -9.0/2.0*x+27.0/2.0*x*x; break;
            case 4: grad(0) = -9.0/2.0*y+27.0/2.0*y*y;                                       grad(1) = -9.0/2.0*x+27.0*x*y; break;
            case 5: grad(0) = 9.0/2.0*y-27.0/2.0*y*y;                                        grad(1) = -9.0/2.0+9.0/2.0*x+36.0*y-27.0*x*y-81.0/2.0*y*y; break;
            case 6: grad(0) = -45.0/2.0*y+27.0*x*y+27.0*y*y;                                 grad(1) = 9.0-45.0/2.0*x-45.0*y+27.0/2.0*x*x+54.0*x*y+81.0/2.0*y*y; break;
            case 7: grad(0) = 9.0-45.0*x-45.0/2.0*y+81.0/2.0*x*x+54.0*x*y+27.0/2.0*y*y;    grad(1) = -45.0/2.0*x+27.0*x*x+27.0*x*y; break;
            case 8: grad(0) = -9.0/2.0+36.0*x+9.0/2.0*y-81.0/2.0*x*x-27.0*x*y;              grad(1) = 9.0/2.0*x-27.0/2.0*x*x; break;
            case 9: grad(0) = 27.0*y-54.0*x*y-27.0*y*y;                                      grad(1) = 27.0*x-27.0*x*x-54.0*x*y; break;
          } break;
      }
      return grad;
    }

    if constexpr (dim == 3) {
      const RealType x = point(0);
      const RealType y = point(1);
      const RealType z = point(2);
      const RealType L0 = 1.0 - x - y - z;
      const RealType L1 = x;
      const RealType L2 = y;
      const RealType L3 = z;
      // dL0/dx=-1, dL0/dy=-1, dL0/dz=-1
      // dL1/dx= 1, dL2/dy= 1, dL3/dz= 1, all others zero
      switch (p_) {
        case 0:
          grad(0) = 0.0; grad(1) = 0.0; grad(2) = 0.0;
          break;
        case 1:
          switch (i) {
            case 0: grad(0) = -1.0; grad(1) = -1.0; grad(2) = -1.0; break;
            case 1: grad(0) =  1.0; grad(1) =  0.0; grad(2) =  0.0; break;
            case 2: grad(0) =  0.0; grad(1) =  1.0; grad(2) =  0.0; break;
            case 3: grad(0) =  0.0; grad(1) =  0.0; grad(2) =  1.0; break;
          } break;
        case 2:
          // phi_i vertex = L_i*(2L_i-1), grad = (4L_i-1)*grad(L_i)
          // phi_ij edge  = 4*L_i*L_j,   grad = 4*(L_j*grad(L_i) + L_i*grad(L_j))
          switch (i) {
            case 0: grad(0) = -(4.0*L0-1.0); grad(1) = -(4.0*L0-1.0); grad(2) = -(4.0*L0-1.0); break;
            case 1: grad(0) =  (4.0*L1-1.0); grad(1) =  0.0;           grad(2) =  0.0;           break;
            case 2: grad(0) =  0.0;           grad(1) =  (4.0*L2-1.0); grad(2) =  0.0;           break;
            case 3: grad(0) =  0.0;           grad(1) =  0.0;           grad(2) =  (4.0*L3-1.0); break;
            // edge 01: 4*L0*L1
            case 4: grad(0) = 4.0*(L1*(-1.0) + L0*1.0); grad(1) = 4.0*L1*(-1.0); grad(2) = 4.0*L1*(-1.0); break;
            // edge 02: 4*L0*L2
            case 5: grad(0) = 4.0*L2*(-1.0); grad(1) = 4.0*(L2*(-1.0) + L0*1.0); grad(2) = 4.0*L2*(-1.0); break;
            // edge 03: 4*L0*L3
            case 6: grad(0) = 4.0*L3*(-1.0); grad(1) = 4.0*L3*(-1.0); grad(2) = 4.0*(L3*(-1.0) + L0*1.0); break;
            // edge 12: 4*L1*L2
            case 7: grad(0) = 4.0*L2;        grad(1) = 4.0*L1;        grad(2) = 0.0;                        break;
            // edge 23: 4*L2*L3
            case 8: grad(0) = 0.0;            grad(1) = 4.0*L3;        grad(2) = 4.0*L2;                     break;
            // edge 13: 4*L1*L3
            case 9: grad(0) = 4.0*L3;        grad(1) = 0.0;            grad(2) = 4.0*L1;                     break;
          } break;
        case 3: {
          // d/dx[ 0.5*L_i*(3L_i-1)*(3L_i-2) ] = 0.5*(3L_i-1)*(3L_i-2)*dL_i
          //   + 0.5*L_i*3*(3L_i-2)*dL_i + 0.5*L_i*(3L_i-1)*3*dL_i
          //   = 0.5*dL_i * [ (3L_i-1)(3L_i-2) + 3L_i(3L_i-2) + 3L_i(3L_i-1) ]
          //   = 0.5*dL_i * (27L_i^2 - 18L_i + 2)
          auto dvert = [](RealType L) { return 0.5*(27.0*L*L - 18.0*L + 2.0); };
          // d/dx[ 4.5*L_i*L_j*(3L_i-1) ] = 4.5*[ L_j*(3L_i-1)*dL_i + L_i*L_j*3*dL_i + L_i*(3L_i-1)*dL_j ]
          //   = 4.5*[ dL_i*(L_j*(3L_i-1) + 3*L_i*L_j) + dL_j*L_i*(3L_i-1) ]
          //   = 4.5*[ dL_i*L_j*(3L_i-1+3L_i) + dL_j*L_i*(3L_i-1) ]
          //   = 4.5*[ dL_i*L_j*(6L_i-1) + dL_j*L_i*(3L_i-1) ]
          // d/dx[ 27*L_i*L_j*L_k ] = 27*[ dL_i*L_j*L_k + L_i*dL_j*L_k + L_i*L_j*dL_k ]
          switch (i) {
            case 0:  { RealType c=dvert(L0); grad(0)=-c;  grad(1)=-c;  grad(2)=-c;  break; }
            case 1:  { RealType c=dvert(L1); grad(0)= c;  grad(1)= 0;  grad(2)= 0;  break; }
            case 2:  { RealType c=dvert(L2); grad(0)= 0;  grad(1)= c;  grad(2)= 0;  break; }
            case 3:  { RealType c=dvert(L3); grad(0)= 0;  grad(1)= 0;  grad(2)= c;  break; }
            // Edge v0-v1: 4.5*L0*L1*(3L0-1), node 4 closer to v0
            case 4: {
              grad(0) = 4.5*((-1.0)*L1*(6.0*L0-1.0) + (1.0)*L0*(3.0*L0-1.0));   // dL0=-1, dL1=1 wrt x
              grad(1) = 4.5*((-1.0)*L1*(6.0*L0-1.0) + (-1.0)*L0*(3.0*L0-1.0));  // dL0=-1, dL1=0 wrt y
              grad(2) = 4.5*((-1.0)*L1*(6.0*L0-1.0) + (-1.0)*L0*(3.0*L0-1.0));  // dL0=-1, dL1=0 wrt z
              break;
            }
            // Edge v0-v1: 4.5*L0*L1*(3L1-1), node 5 closer to v1
            case 5: {
              grad(0) = 4.5*((-1.0)*L1*(3.0*L1-1.0) + (1.0)*L0*(6.0*L1-1.0));
              grad(1) = 4.5*((-1.0)*L1*(3.0*L1-1.0) + (-1.0)*L0*(6.0*L1-1.0) + (-1.0)*L0*(3.0*L1-1.0) + 0.0);
              // Simplify: d/dy[4.5*L0*L1*(3L1-1)] = 4.5*(dL0/dy*L1*(3L1-1) + L0*dL1/dy*(3L1-1) + L0*L1*3*dL1/dy)
              //         = 4.5*(-L1*(3L1-1) + 0 + 0) = -4.5*L1*(3L1-1)
              grad(1) = -4.5*L1*(3.0*L1-1.0);
              grad(2) = -4.5*L1*(3.0*L1-1.0);
              break;
            }
            // Edge v0-v2: 4.5*L0*L2*(3L0-1), node 6 closer to v0
            case 6: {
              // d/dx = 4.5*(dL0/dx*L2*(6L0-1) + dL2/dx*L0*(3L0-1)) = 4.5*(-L2*(6L0-1))
              grad(0) = -4.5*L2*(6.0*L0-1.0);
              grad(1) =  4.5*((-1.0)*L2*(6.0*L0-1.0) + (1.0)*L0*(3.0*L0-1.0));
              grad(2) = -4.5*L2*(6.0*L0-1.0);
              break;
            }
            // Edge v0-v2: 4.5*L0*L2*(3L2-1), node 7 closer to v2
            case 7: {
              grad(0) = -4.5*L2*(3.0*L2-1.0);
              grad(1) =  4.5*((-1.0)*L2*(3.0*L2-1.0) + (1.0)*L0*(6.0*L2-1.0));
              grad(2) = -4.5*L2*(3.0*L2-1.0);
              break;
            }
            // Edge v0-v3: 4.5*L0*L3*(3L0-1), node 8 closer to v0
            case 8: {
              grad(0) = -4.5*L3*(6.0*L0-1.0);
              grad(1) = -4.5*L3*(6.0*L0-1.0);
              grad(2) =  4.5*((-1.0)*L3*(6.0*L0-1.0) + (1.0)*L0*(3.0*L0-1.0));
              break;
            }
            // Edge v0-v3: 4.5*L0*L3*(3L3-1), node 9 closer to v3
            case 9: {
              grad(0) = -4.5*L3*(3.0*L3-1.0);
              grad(1) = -4.5*L3*(3.0*L3-1.0);
              grad(2) =  4.5*((-1.0)*L3*(3.0*L3-1.0) + (1.0)*L0*(6.0*L3-1.0));
              break;
            }
            // Edge v1-v2: 4.5*L1*L2*(3L1-1), node 10 closer to v1
            case 10: {
              grad(0) =  4.5*L2*(6.0*L1-1.0);
              grad(1) =  4.5*L1*(3.0*L1-1.0);
              grad(2) =  0.0;
              break;
            }
            // Edge v1-v2: 4.5*L1*L2*(3L2-1), node 11 closer to v2
            case 11: {
              grad(0) =  4.5*L2*(3.0*L2-1.0);
              grad(1) =  4.5*L1*(6.0*L2-1.0);
              grad(2) =  0.0;
              break;
            }
            // Edge v1-v3: 4.5*L1*L3*(3L1-1), node 12 closer to v1
            case 12: {
              grad(0) =  4.5*L3*(6.0*L1-1.0);
              grad(1) =  0.0;
              grad(2) =  4.5*L1*(3.0*L1-1.0);
              break;
            }
            // Edge v1-v3: 4.5*L1*L3*(3L3-1), node 13 closer to v3
            case 13: {
              grad(0) =  4.5*L3*(3.0*L3-1.0);
              grad(1) =  0.0;
              grad(2) =  4.5*L1*(6.0*L3-1.0);
              break;
            }
            // Edge v2-v3: 4.5*L2*L3*(3L2-1), node 14 closer to v2
            case 14: {
              grad(0) =  0.0;
              grad(1) =  4.5*L3*(6.0*L2-1.0);
              grad(2) =  4.5*L2*(3.0*L2-1.0);
              break;
            }
            // Edge v2-v3: 4.5*L2*L3*(3L3-1), node 15 closer to v3
            case 15: {
              grad(0) =  0.0;
              grad(1) =  4.5*L3*(3.0*L3-1.0);
              grad(2) =  4.5*L2*(6.0*L3-1.0);
              break;
            }
            // Face opp v3: 27*L0*L1*L2
            case 16: {
              grad(0) = 27.0*((-1.0)*L1*L2 + (1.0)*L0*L2);
              grad(1) = 27.0*((-1.0)*L1*L2 + (-1.0)*L0*L2 + (1.0)*L0*L1);
              // d/dy[27*L0*L1*L2] = 27*(dL0/dy*L1*L2 + L0*dL1/dy*L2 + L0*L1*dL2/dy)
              //                   = 27*(-L1*L2 + 0 + L0*L1)
              grad(1) = 27.0*(L0*L1 - L1*L2);
              grad(2) = 27.0*(-L1*L2);
              break;
            }
            // Face opp v2: 27*L0*L1*L3
            case 17: {
              grad(0) = 27.0*(L0*L3 - L1*L3);
              grad(1) = 27.0*(-L1*L3);
              grad(2) = 27.0*(L0*L1 - L1*L3);
              break;
            }
            // Face opp v1: 27*L0*L2*L3
            case 18: {
              grad(0) = 27.0*(-L2*L3);
              grad(1) = 27.0*(L0*L3 - L2*L3);
              grad(2) = 27.0*(L0*L2 - L2*L3);
              break;
            }
            // Face opp v0: 27*L1*L2*L3
            case 19: {
              grad(0) = 27.0*L2*L3;
              grad(1) = 27.0*L1*L3;
              grad(2) = 27.0*L1*L2;
              break;
            }
          } break;
        }
      }
      return grad;
    }

    return grad;
  }

  // ── basis check ───────────────────────────────────────────────────────────

  void check_basis() const
  {
    for (int i = 0; i < (int)n_dofs_; ++i) {
      for (int j = 0; j < (int)n_dofs_; ++j) {
        Tensor<1, dim, RealType> pt;
        for (int d = 0; d < (int)dim; ++d)
          pt(d) = nodes_[j][d];
        RealType val = eval(i, pt);
        RealType expected = (i == j) ? RealType(1) : RealType(0);
        if (std::abs(val - expected) >= 1.0e-10) {
          std::cout << "FAIL: phi_" << i << "(node_" << j << ") = " << val
                    << " expected " << expected << std::endl;
          ASSERT(false, "Basis check failed");
        }
      }
    }
  }
};

// ── QGaussSimplex ─────────────────────────────────────────────────────────────

template<unsigned int dim, typename RealType>
class QGaussSimplex
{
public:
  static constexpr unsigned int n_q_points(unsigned int p)
  {
<<<<<<< HEAD
    if constexpr (dim == 1) return p;
=======
    if constexpr (dim == 1) {
      return p;
    }
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
    if constexpr (dim == 2) {
      constexpr unsigned int table[] = { 0, 1, 3, 4, 6, 7, 12, 13 };
      return table[p];
    }
    if constexpr (dim == 3) {
<<<<<<< HEAD
      // Points for orders 1-4 on the tetrahedron
      constexpr unsigned int table[] = { 0, 1, 4, 5, 11 };
=======
      // Tet quadrature point counts for orders 1-7
      constexpr unsigned int table[] = { 0, 1, 4, 5, 11, 14, 24, 31 };
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
      return table[p];
    }
    return 0;
  }

  explicit QGaussSimplex(unsigned int order)
    : order_(order)
  {
    ASSERT(order >= 1, "Quadrature order must be at least 1");
    ASSERT(order <= max_order_, "Quadrature order exceeds maximum");

    std::vector<std::array<RealType, dim>> pts;
    std::vector<RealType> wts;
    get_rule(order, pts, wts);

    n_points_ = static_cast<unsigned int>(wts.size());
    ASSERT(n_points_ == n_q_points(order), "Point count mismatch");

    points_  = Kokkos::View<RealType**, Layout, HostMemSpace>("simplex_quad_points",  n_points_, dim);
    weights_ = Kokkos::View<RealType*,  Layout, HostMemSpace>("simplex_quad_weights", n_points_);

    for (unsigned int q = 0; q < n_points_; ++q) {
      for (unsigned int d = 0; d < dim; ++d)
        points_(q, d) = pts[q][d];
      weights_(q) = wts[q];
    }
  }

  unsigned int order()    const { return order_; }
  unsigned int n_points() const { return n_points_; }

  Tensor<1, dim, RealType> point(unsigned int q) const
  {
    Tensor<1, dim, RealType> p;
    for (unsigned int d = 0; d < dim; ++d)
      p(d) = points_(q, d);
    return p;
  }

  RealType weight(unsigned int q) const { return weights_(q); }

  [[deprecated]]
  const Kokkos::View<RealType**, Layout, HostMemSpace>& points_host()  const { return points_; }
  [[deprecated]]
  const Kokkos::View<RealType*,  Layout, HostMemSpace>& weights_host() const { return weights_; }

private:
  static constexpr unsigned int max_order_ = 7;
  unsigned int order_;
  unsigned int n_points_;
  Kokkos::View<RealType**, Layout, HostMemSpace> points_;
  Kokkos::View<RealType*,  Layout, HostMemSpace> weights_;

  static void get_rule(unsigned int order,
                       std::vector<std::array<RealType, dim>>& pts,
                       std::vector<RealType>& wts)
  {
<<<<<<< HEAD
    if constexpr (dim == 1) get_line_rule(order, pts, wts);
    if constexpr (dim == 2) get_triangle_rule(order, pts, wts);
    if constexpr (dim == 3) get_tet_rule(order, pts, wts);
=======
    if constexpr (dim == 1) {
      get_line_rule(order, pts, wts);
    }
    if constexpr (dim == 2) {
      get_triangle_rule(order, pts, wts);
    }
    if constexpr (dim == 3) {
      get_tet_rule(order, pts, wts);
    }
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
  }

  static void get_line_rule(unsigned int order,
                            std::vector<std::array<RealType, 1>>& pts,
                            std::vector<RealType>& wts)
  {
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

<<<<<<< HEAD

=======
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
  static void get_tet_rule(unsigned int order,
                           std::vector<std::array<RealType, 3>>& pts,
                           std::vector<RealType>& wts)
  {
<<<<<<< HEAD
    // Reference tet: vertices (0,0,0),(1,0,0),(0,1,0),(0,0,1), volume = 1/6
    switch (order) {
      case 1: {
        // 1-point centroid rule, exact for degree 1
        pts = { {{ 0.25, 0.25, 0.25 }} };
=======
    // Weights already include the 1/6 factor for the reference tet volume.
    switch (order) {
      case 1: {
        // 1-point centroid, exact degree 1
        pts = { { { 0.25, 0.25, 0.25 } } };
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
        wts = { 1.0/6.0 };
        break;
      }
      case 2: {
<<<<<<< HEAD
        // 4-point rule, exact for degree 2
        const RealType a = 0.138196601125011; // (5-sqrt(5))/20
        const RealType b = 0.585410196624969; // (5+3*sqrt(5))/20
        pts = {
          {{ a, a, a }},
          {{ b, a, a }},
          {{ a, b, a }},
          {{ a, a, b }}
        };
        wts = { 1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/24.0 };
        break;
      }
      case 3: {
        // 5-point rule, exact for degree 3
        const RealType w0 = -0.1333333333333333; // -4/30
        const RealType w1 =  0.075;              //  9/120
        pts = {
          {{ 0.25,              0.25,              0.25              }},
          {{ 1.0/6.0,           1.0/6.0,           1.0/6.0           }},
          {{ 0.5,               1.0/6.0,           1.0/6.0           }},
          {{ 1.0/6.0,           0.5,               1.0/6.0           }},
          {{ 1.0/6.0,           1.0/6.0,           0.5               }}
        };
=======
        // 4-point, exact degree 2
        const RealType a = 0.1381966011250105;
        const RealType b = 0.5854101966249685;
        const RealType w = 1.0/24.0;
        pts = { { { a,a,a } }, { { b,a,a } }, { { a,b,a } }, { { a,a,b } } };
        wts = { w,w,w,w };
        break;
      }
      case 3: {
        // 5-point, exact degree 3
        const RealType w0 = -4.0/30.0;
        const RealType w1 =  3.0/40.0;
        pts = { { { 0.25,       0.25,       0.25       } },
                { { 1.0/6.0,    1.0/6.0,    1.0/6.0    } },
                { { 0.5,        1.0/6.0,    1.0/6.0    } },
                { { 1.0/6.0,    0.5,        1.0/6.0    } },
                { { 1.0/6.0,    1.0/6.0,    0.5        } } };
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
        wts = { w0, w1, w1, w1, w1 };
        break;
      }
      case 4: {
<<<<<<< HEAD
        // 11-point rule, exact for degree 4
        // Keast rule (1986) for the tetrahedron
        const RealType a1 = 0.0714285714285714; // 1/14
        const RealType b1 = 0.0714285714285714;
        const RealType a2 = 0.0992499999999944;
        const RealType b2 = 0.7022499999999169;
        const RealType a3 = 0.3197936278296299;
        const RealType b3 = 0.0402063721703701;

        const RealType w1 = -0.0131555555555556;
        const RealType w2 =  0.0073333333333333;
        const RealType w3 =  0.0112233333333333;

        pts = {
          {{ 0.25,  0.25,  0.25  }},
          {{ a2,    a2,    a2    }},
          {{ b2,    a2,    a2    }},
          {{ a2,    b2,    a2    }},
          {{ a2,    a2,    b2    }},
          {{ a3,    a3,    b3    }},
          {{ a3,    b3,    a3    }},
          {{ b3,    a3,    a3    }},
          {{ b3,    b3,    a3    }},
          {{ b3,    a3,    b3    }},
          {{ a3,    b3,    b3    }}
        };
        wts = { w1, w2, w2, w2, w2, w3, w3, w3, w3, w3, w3 };
=======
        // 11-point, exact degree 4 (Keast rule)
        const RealType a1 = 0.0714285714285714;
        const RealType b1 = 0.7142857142857143;
        const RealType a2 = 0.0990169452354533;
        const RealType b2 = 0.7029490017670679;
        const RealType a3 = 0.4319580241992353;
        const RealType b3 = 0.1021489756935209;
        const RealType w0 = -0.013155555555556;
        const RealType w1 =  0.007622222222222;
        const RealType w2 =  0.024888888888889;
        pts = {
          { { 0.25, 0.25, 0.25 } },
          { { a1,a1,a1 } }, { { b1,a1,a1 } }, { { a1,b1,a1 } }, { { a1,a1,b1 } },
          { { a2,a2,b2 } }, { { a2,b2,a2 } }, { { b2,a2,a2 } },
          { { a3,a3,b3 } }, { { a3,b3,a3 } }, { { b3,a3,a3 } }
        };
        wts = { w0, w1,w1,w1,w1, w2,w2,w2, w2,w2,w2 };
        break;
      }
      case 5: {
        // 14-point, exact degree 5 (Keast)
        const RealType a1 = 0.0927352503108912;
        const RealType b1 = 1.0 - 3.0*a1;
        const RealType a2 = 0.3108859192633873;
        const RealType b2 = 1.0 - 3.0*a2;
        const RealType a3 = 0.0455037041256955;
        const RealType b3 = 0.4544962958743045;
        const RealType w1 = 0.007302273184423;
        const RealType w2 = 0.011857160542621;
        const RealType w3 = 0.008510907530862;
        pts = {
          { { a1,a1,a1 } }, { { b1,a1,a1 } }, { { a1,b1,a1 } }, { { a1,a1,b1 } },
          { { a2,a2,a2 } }, { { b2,a2,a2 } }, { { a2,b2,a2 } }, { { a2,a2,b2 } },
          { { a3,a3,b3 } }, { { a3,b3,a3 } }, { { b3,a3,a3 } },
          { { b3,a3,b3 } }, { { a3,b3,b3 } }, { { b3,b3,a3 } }
        };
        wts = { w1,w1,w1,w1, w2,w2,w2,w2, w3,w3,w3,w3,w3,w3 };
        break;
      }
      case 6: {
        // 24-point, exact degree 6 (Keast)
        const RealType a1 = 0.214602871259152;
        const RealType b1 = 1.0 - 3.0*a1;
        const RealType a2 = 0.040673958534611;
        const RealType b2 = 1.0 - 3.0*a2;
        const RealType a3 = 0.322337890142276;
        const RealType b3 = 1.0 - 3.0*a3;
        const RealType a4 = 0.063661001875018;
        const RealType c4 = 0.603005664791649;
        const RealType b4 = 1.0 - 2.0*a4 - c4;
        const RealType w1 = 0.003992275025817;
        const RealType w2 = 0.001007721105533;
        const RealType w3 = 0.005535128409014;
        const RealType w4 = 0.002992280777774;
        pts = {
          { { a1,a1,a1 } }, { { b1,a1,a1 } }, { { a1,b1,a1 } }, { { a1,a1,b1 } },
          { { a2,a2,a2 } }, { { b2,a2,a2 } }, { { a2,b2,a2 } }, { { a2,a2,b2 } },
          { { a3,a3,a3 } }, { { b3,a3,a3 } }, { { a3,b3,a3 } }, { { a3,a3,b3 } },
          { { a4,a4,c4 } }, { { a4,c4,a4 } }, { { c4,a4,a4 } },
          { { a4,a4,b4 } }, { { a4,b4,a4 } }, { { b4,a4,a4 } },
          { { a4,c4,b4 } }, { { c4,a4,b4 } }, { { a4,b4,c4 } },
          { { b4,a4,c4 } }, { { c4,b4,a4 } }, { { b4,c4,a4 } }
        };
        wts = { w1,w1,w1,w1, w2,w2,w2,w2, w3,w3,w3,w3,
                w4,w4,w4,w4,w4,w4,w4,w4,w4,w4,w4,w4 };
        break;
      }
      case 7: {
        // 31-point, exact degree 7 (Keast)
        const RealType w0 =  0.009548528946413;
        const RealType a1 = 0.328054696711427;
        const RealType b1 = 1.0 - 3.0*a1;
        const RealType a2 = 0.106952273932934;
        const RealType b2 = 1.0 - 3.0*a2;
        const RealType a3 = 0.184246840037000;
        const RealType b3 = 1.0 - 3.0*a3;
        const RealType a4 = 0.028723060836557;
        const RealType c4 = 0.712255819140848;
        const RealType b4 = 1.0 - 2.0*a4 - c4;
        const RealType a5 = 0.062145644277395;
        const RealType c5 = 0.480239766957208;
        const RealType b5 = 1.0 - 2.0*a5 - c5;
        const RealType r1 = 0.001007721105533;
        const RealType r2 = 0.005535128409014;
        const RealType r3 = 0.000992281784110;
        const RealType r4 = 0.002292431669091;
        const RealType r5 = 0.003066403643667;
        pts = {
          { { 0.25, 0.25, 0.25 } },
          { { a1,a1,a1 } }, { { b1,a1,a1 } }, { { a1,b1,a1 } }, { { a1,a1,b1 } },
          { { a2,a2,a2 } }, { { b2,a2,a2 } }, { { a2,b2,a2 } }, { { a2,a2,b2 } },
          { { a3,a3,a3 } }, { { b3,a3,a3 } }, { { a3,b3,a3 } }, { { a3,a3,b3 } },
          { { a4,a4,c4 } }, { { a4,c4,a4 } }, { { c4,a4,a4 } },
          { { a4,a4,b4 } }, { { a4,b4,a4 } }, { { b4,a4,a4 } },
          { { a4,b4,c4 } }, { { b4,a4,c4 } }, { { a4,c4,b4 } },
          { { b4,c4,a4 } }, { { c4,a4,b4 } }, { { c4,b4,a4 } },
          { { a5,a5,c5 } }, { { a5,c5,a5 } }, { { c5,a5,a5 } },
          { { a5,a5,b5 } }, { { a5,b5,a5 } }, { { b5,a5,a5 } }
        };
        wts = { w0,
                r1,r1,r1,r1,
                r2,r2,r2,r2,
                r3,r3,r3,r3,
                r4,r4,r4,r4,r4,r4,r4,r4,r4,r4,r4,r4,
                r5,r5,r5,r5,r5,r5 };
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
        break;
      }
      default:
        ASSERT(false, "Unsupported quadrature order for tetrahedron");
    }
  }
};

// ── FEValues ──────────────────────────────────────────────────────────────────

template<unsigned int dim, typename RealType>
class FEValues
{
public:
  FEValues(const FE_DGQLegendre<dim, RealType>& fe,
           const QGaussSimplex<dim, RealType>& quad)
    : fe_(fe), quad_(quad)
    , n_dofs_(fe.n_dofs()), n_q_(quad.n_points())
  {
<<<<<<< HEAD
    JxW_      = Kokkos::View<RealType*,   Layout, HostMemSpace>("JxW",      n_q_);
    q_point_  = Kokkos::View<RealType**,  Layout, HostMemSpace>("q_point",  n_q_,    dim);
    phi_      = Kokkos::View<RealType**,  Layout, HostMemSpace>("phi",      n_dofs_, n_q_);
    grad_phi_ = Kokkos::View<RealType***, Layout, HostMemSpace>("grad_phi", n_dofs_, n_q_, dim);
=======
    JxW_ = Kokkos::View<RealType*, Layout, HostMemSpace>("JxW", n_q_);
    q_point_ =
      Kokkos::View<RealType**, Layout, HostMemSpace>("q_point", n_q_, dim);
    phi_ = Kokkos::View<RealType**, Layout, HostMemSpace>("phi", n_dofs_, n_q_);
    grad_phi_ = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "grad_phi", n_dofs_, n_q_, dim);
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
  }

  template<typename CellAccessor>
  void reinit(const CellAccessor& cell)
  {
<<<<<<< HEAD
    if constexpr (dim == 2) {
      reinit_2d(cell);
    } else if constexpr (dim == 3) {
      reinit_3d(cell);
=======
    static_assert(dim == 2 || dim == 3);

    if constexpr (dim == 2) {
      RealType J[2][2], x0[2];
      for (unsigned int d = 0; d < 2; ++d)
        x0[d] = cell.vertex(0)(d);
      for (unsigned int d = 0; d < 2; ++d) {
        J[d][0] = cell.vertex(1)(d) - cell.vertex(0)(d);
        J[d][1] = cell.vertex(2)(d) - cell.vertex(0)(d);
      }
      const RealType det_J = J[0][0]*J[1][1] - J[0][1]*J[1][0];
      const RealType J_inv[2][2] = { { J[1][1]/det_J, -J[0][1]/det_J },
                                     { -J[1][0]/det_J, J[0][0]/det_J } };

      ASSERT(det_J > 0,
             "Negative Jacobian on cell " + std::to_string(cell.index()) +
               ", det_J = " + std::to_string(det_J));

      for (unsigned int q = 0; q < n_q_; ++q) {
        const auto xi = quad_.point(q);
        const RealType xv = xi(0), yv = xi(1);
        ASSERT(xv >= -1e-10 && yv >= -1e-10 && xv+yv <= 1.0+1e-10,
               "Quadrature point outside reference triangle: (" +
                 std::to_string(xv) + ", " + std::to_string(yv) + ")");
      }

      for (unsigned int q = 0; q < n_q_; ++q) {
        JxW_(q) = std::abs(det_J) * quad_.weight(q);
        const auto xi = quad_.point(q);
        for (unsigned int i = 0; i < n_dofs_; ++i) {
          phi_(i, q) = fe_.shape_value(i, xi);
          const auto tmp = fe_.shape_gradient(i, xi);
          for (unsigned int d = 0; d < 2; ++d)
            grad_phi_(i, q, d) = J_inv[0][d]*tmp(0) + J_inv[1][d]*tmp(1);
        }
        for (unsigned int d = 0; d < 2; ++d)
          q_point_(q, d) = x0[d] + J[d][0]*xi(0) + J[d][1]*xi(1);
      }
    }

    if constexpr (dim == 3) {
      // 3×3 Jacobian: columns are edge vectors from vertex 0
      RealType J[3][3], x0[3];
      for (unsigned int d = 0; d < 3; ++d)
        x0[d] = cell.vertex(0)(d);
      for (unsigned int d = 0; d < 3; ++d) {
        J[d][0] = cell.vertex(1)(d) - cell.vertex(0)(d);
        J[d][1] = cell.vertex(2)(d) - cell.vertex(0)(d);
        J[d][2] = cell.vertex(3)(d) - cell.vertex(0)(d);
      }

      const RealType det_J =
          J[0][0]*(J[1][1]*J[2][2] - J[1][2]*J[2][1])
        - J[0][1]*(J[1][0]*J[2][2] - J[1][2]*J[2][0])
        + J[0][2]*(J[1][0]*J[2][1] - J[1][1]*J[2][0]);

      ASSERT(det_J > 0,
             "Negative Jacobian on cell " + std::to_string(cell.index()) +
               ", det_J = " + std::to_string(det_J));

      // 3×3 inverse (J^{-T} used for gradient transform)
      RealType Ji[3][3];
      Ji[0][0] =  (J[1][1]*J[2][2]-J[1][2]*J[2][1])/det_J;
      Ji[0][1] = -(J[0][1]*J[2][2]-J[0][2]*J[2][1])/det_J;
      Ji[0][2] =  (J[0][1]*J[1][2]-J[0][2]*J[1][1])/det_J;
      Ji[1][0] = -(J[1][0]*J[2][2]-J[1][2]*J[2][0])/det_J;
      Ji[1][1] =  (J[0][0]*J[2][2]-J[0][2]*J[2][0])/det_J;
      Ji[1][2] = -(J[0][0]*J[1][2]-J[0][2]*J[1][0])/det_J;
      Ji[2][0] =  (J[1][0]*J[2][1]-J[1][1]*J[2][0])/det_J;
      Ji[2][1] = -(J[0][0]*J[2][1]-J[0][1]*J[2][0])/det_J;
      Ji[2][2] =  (J[0][0]*J[1][1]-J[0][1]*J[1][0])/det_J;

      for (unsigned int q = 0; q < n_q_; ++q) {
        JxW_(q) = std::abs(det_J) * quad_.weight(q);
        const auto xi = quad_.point(q);
        for (unsigned int i = 0; i < n_dofs_; ++i) {
          phi_(i, q) = fe_.shape_value(i, xi);
          const auto tmp = fe_.shape_gradient(i, xi);
          // physical gradient = J^{-T} * ref_gradient
          for (unsigned int d = 0; d < 3; ++d)
            grad_phi_(i, q, d) = Ji[0][d]*tmp(0) + Ji[1][d]*tmp(1) + Ji[2][d]*tmp(2);
        }
        for (unsigned int d = 0; d < 3; ++d)
          q_point_(q, d) = x0[d] + J[d][0]*xi(0) + J[d][1]*xi(1) + J[d][2]*xi(2);
      }
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
    }
  }

  unsigned int n_dofs()     const { return n_dofs_; }
  unsigned int n_q_points() const { return n_q_; }
  RealType JxW(unsigned int q) { return JxW_(q); }

  Tensor<1, dim, RealType> q_point(unsigned int q)
  {
    Tensor<1, dim, RealType> p;
    for (unsigned int d = 0; d < dim; ++d) p(d) = q_point_(q, d);
    return p;
  }

  RealType shape_value(unsigned int i, unsigned int q) { return phi_(i, q); }

  Tensor<1, dim, RealType> shape_gradient(unsigned int i, unsigned int q)
  {
    Tensor<1, dim, RealType> g;
    for (unsigned int d = 0; d < dim; ++d) g(d) = grad_phi_(i, q, d);
    return g;
  }

private:
  const FE_DGQLegendre<dim, RealType>& fe_;
  const QGaussSimplex<dim, RealType>&  quad_;
  unsigned int n_dofs_, n_q_;
  Kokkos::View<RealType*,   Layout, HostMemSpace> JxW_;
  Kokkos::View<RealType**,  Layout, HostMemSpace> q_point_;
  Kokkos::View<RealType**,  Layout, HostMemSpace> phi_;
  Kokkos::View<RealType***, Layout, HostMemSpace> grad_phi_;

  template<typename CellAccessor>
  void reinit_2d(const CellAccessor& cell)
  {
    RealType J[2][2], x0[2];
    for (unsigned int d = 0; d < 2; ++d) x0[d] = cell.vertex(0)(d);
    for (unsigned int d = 0; d < 2; ++d) {
      J[d][0] = cell.vertex(1)(d) - cell.vertex(0)(d);
      J[d][1] = cell.vertex(2)(d) - cell.vertex(0)(d);
    }
    const RealType det_J = J[0][0]*J[1][1] - J[0][1]*J[1][0];
    ASSERT(det_J > 0, "Negative Jacobian on cell " + std::to_string(cell.index()));
    const RealType J_inv[2][2] = {{ J[1][1]/det_J, -J[0][1]/det_J },
                                   {-J[1][0]/det_J,  J[0][0]/det_J }};
    for (unsigned int q = 0; q < n_q_; ++q) {
      JxW_(q) = std::abs(det_J) * quad_.weight(q);
      const auto xi = quad_.point(q);
      for (unsigned int i = 0; i < n_dofs_; ++i) {
        phi_(i, q) = fe_.shape_value(i, xi);
        const auto tmp = fe_.shape_gradient(i, xi);
        for (unsigned int d = 0; d < 2; ++d)
          grad_phi_(i, q, d) = J_inv[0][d]*tmp(0) + J_inv[1][d]*tmp(1);
      }
      for (unsigned int d = 0; d < 2; ++d)
        q_point_(q, d) = x0[d] + J[d][0]*xi(0) + J[d][1]*xi(1);
    }
  }

  template<typename CellAccessor>
  void reinit_3d(const CellAccessor& cell)
  {
    // Build 3x3 Jacobian: columns are edge vectors from vertex 0
    Tensor<2, 3, RealType> J;
    Tensor<1, 3, RealType> x0;
    for (unsigned int d = 0; d < 3; ++d) x0(d) = cell.vertex(0)(d);
    for (unsigned int d = 0; d < 3; ++d) {
      J(d, 0) = cell.vertex(1)(d) - cell.vertex(0)(d);
      J(d, 1) = cell.vertex(2)(d) - cell.vertex(0)(d);
      J(d, 2) = cell.vertex(3)(d) - cell.vertex(0)(d);
    }
    const RealType det_J = det(J);
    ASSERT(det_J > 0, "Negative Jacobian on 3D cell " + std::to_string(cell.index()));
    const Tensor<2, 3, RealType> J_inv = inverse(J);

    for (unsigned int q = 0; q < n_q_; ++q) {
      JxW_(q) = std::abs(det_J) * quad_.weight(q);
      const auto xi = quad_.point(q);

      for (unsigned int i = 0; i < n_dofs_; ++i) {
        phi_(i, q) = fe_.shape_value(i, xi);
        // Map reference gradient to physical: grad_phys = J^{-T} * grad_ref
        const auto tmp = fe_.shape_gradient(i, xi);
        for (unsigned int d = 0; d < 3; ++d) {
          RealType s = 0;
          for (unsigned int e = 0; e < 3; ++e)
            s += J_inv(e, d) * tmp(e);
          grad_phi_(i, q, d) = s;
        }
      }

      for (unsigned int d = 0; d < 3; ++d) {
        q_point_(q, d) = x0(d) + J(d,0)*xi(0) + J(d,1)*xi(1) + J(d,2)*xi(2);
      }
    }
  }
};

// ── FEFaceValues ──────────────────────────────────────────────────────────────

template<unsigned int dim, typename RealType>
class FEFaceValues
{
public:
  FEFaceValues(const FE_DGQLegendre<dim, RealType>& fe,
               const QGaussSimplex<dim-1, RealType>& quad)
    : fe_(fe), quad_(quad)
    , n_dofs_(fe.n_dofs()), n_q_(quad.n_points())
  {
<<<<<<< HEAD
    JxW_      = Kokkos::View<RealType*,   Layout, HostMemSpace>("JxW",      n_q_);
    q_point_  = Kokkos::View<RealType**,  Layout, HostMemSpace>("q_point",  n_q_,    dim);
    normal_   = Kokkos::View<RealType**,  Layout, HostMemSpace>("normal",   n_q_,    dim);
    phi_      = Kokkos::View<RealType**,  Layout, HostMemSpace>("phi",      n_dofs_, n_q_);
    grad_phi_ = Kokkos::View<RealType***, Layout, HostMemSpace>("grad_phi", n_dofs_, n_q_, dim);
=======
    JxW_ = Kokkos::View<RealType*, Layout, HostMemSpace>("JxW", n_q_);
    q_point_ =
      Kokkos::View<RealType**, Layout, HostMemSpace>("q_point", n_q_, dim);
    normal_ =
      Kokkos::View<RealType**, Layout, HostMemSpace>("normal", n_q_, dim);
    phi_ = Kokkos::View<RealType**, Layout, HostMemSpace>("phi", n_dofs_, n_q_);
    grad_phi_ = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "grad_phi", n_dofs_, n_q_, dim);
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
  }

  template<typename CellAccessor>
  void reinit(const CellAccessor& cell, unsigned int face)
  {
<<<<<<< HEAD
    if constexpr (dim == 2) reinit_2d(cell, face);
    else if constexpr (dim == 3) reinit_3d(cell, face);
=======
    static_assert(dim == 2 || dim == 3);

    ASSERT(face < SimplexTopology<dim>::faces_per_cell,
           "Local face number must be less than the number of faces per cell");

    if constexpr (dim == 2) {
      RealType J[2][2], x0[2];
      for (unsigned int d = 0; d < 2; ++d)
        x0[d] = cell.vertex(0)(d);
      for (unsigned int d = 0; d < 2; ++d) {
        J[d][0] = cell.vertex(1)(d) - cell.vertex(0)(d);
        J[d][1] = cell.vertex(2)(d) - cell.vertex(0)(d);
      }
      const RealType det_J = J[0][0]*J[1][1] - J[0][1]*J[1][0];
      const RealType J_inv[2][2] = { { J[1][1]/det_J, -J[0][1]/det_J },
                                     { -J[1][0]/det_J, J[0][0]/det_J } };

      RealType ref_tangent[2], ref_origin[2], ref_normal[2];
      switch (face) {
        case 0:
          ref_origin[0]=1.0; ref_origin[1]=0.0;
          ref_tangent[0]=-1.0; ref_tangent[1]=1.0;
          ref_normal[0]=1.0; ref_normal[1]=1.0;
          break;
        case 1:
          ref_origin[0]=0.0; ref_origin[1]=1.0;
          ref_tangent[0]=0.0; ref_tangent[1]=-1.0;
          ref_normal[0]=-1.0; ref_normal[1]=0.0;
          break;
        case 2:
          ref_origin[0]=0.0; ref_origin[1]=0.0;
          ref_tangent[0]=1.0; ref_tangent[1]=0.0;
          ref_normal[0]=0.0; ref_normal[1]=-1.0;
          break;
      }

      RealType n_phys[2];
      n_phys[0] = J_inv[0][0]*ref_normal[0] + J_inv[1][0]*ref_normal[1];
      n_phys[1] = J_inv[0][1]*ref_normal[0] + J_inv[1][1]*ref_normal[1];
      RealType t_phys[2];
      t_phys[0] = J[0][0]*ref_tangent[0] + J[0][1]*ref_tangent[1];
      t_phys[1] = J[1][0]*ref_tangent[0] + J[1][1]*ref_tangent[1];
      const RealType phys_edge_len = std::sqrt(t_phys[0]*t_phys[0]+t_phys[1]*t_phys[1]);
      const RealType n_phys_norm   = std::sqrt(n_phys[0]*n_phys[0]+n_phys[1]*n_phys[1]);

      for (unsigned int q = 0; q < n_q_; ++q) {
        const auto xi_face = quad_.point(q);
        const RealType t = xi_face(0);
        RealType xi_ref[2];
        xi_ref[0] = ref_origin[0] + t*ref_tangent[0];
        xi_ref[1] = ref_origin[1] + t*ref_tangent[1];
        Tensor<1, 2, RealType> xi; xi(0)=xi_ref[0]; xi(1)=xi_ref[1];

        JxW_(q) = phys_edge_len * quad_.weight(q);
        for (unsigned int d = 0; d < 2; ++d)
          q_point_(q,d) = x0[d] + J[d][0]*xi_ref[0] + J[d][1]*xi_ref[1];
        for (unsigned int d = 0; d < 2; ++d)
          normal_(q,d) = n_phys[d] / n_phys_norm;
        for (unsigned int i = 0; i < n_dofs_; ++i) {
          phi_(i,q) = fe_.shape_value(i, xi);
          const auto tmp = fe_.shape_gradient(i, xi);
          for (unsigned int d = 0; d < 2; ++d)
            grad_phi_(i,q,d) = J_inv[0][d]*tmp(0) + J_inv[1][d]*tmp(1);
        }
      }
    }

    if constexpr (dim == 3) {
      // Build 3×3 Jacobian from cell vertices
      RealType J[3][3], x0[3];
      for (unsigned int d = 0; d < 3; ++d)
        x0[d] = cell.vertex(0)(d);
      for (unsigned int d = 0; d < 3; ++d) {
        J[d][0] = cell.vertex(1)(d) - cell.vertex(0)(d);
        J[d][1] = cell.vertex(2)(d) - cell.vertex(0)(d);
        J[d][2] = cell.vertex(3)(d) - cell.vertex(0)(d);
      }
      const RealType det_J =
          J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1])
        - J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])
        + J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);

      RealType Ji[3][3];
      Ji[0][0] =  (J[1][1]*J[2][2]-J[1][2]*J[2][1])/det_J;
      Ji[0][1] = -(J[0][1]*J[2][2]-J[0][2]*J[2][1])/det_J;
      Ji[0][2] =  (J[0][1]*J[1][2]-J[0][2]*J[1][1])/det_J;
      Ji[1][0] = -(J[1][0]*J[2][2]-J[1][2]*J[2][0])/det_J;
      Ji[1][1] =  (J[0][0]*J[2][2]-J[0][2]*J[2][0])/det_J;
      Ji[1][2] = -(J[0][0]*J[1][2]-J[0][2]*J[1][0])/det_J;
      Ji[2][0] =  (J[1][0]*J[2][1]-J[1][1]*J[2][0])/det_J;
      Ji[2][1] = -(J[0][0]*J[2][1]-J[0][1]*J[2][0])/det_J;
      Ji[2][2] =  (J[0][0]*J[1][1]-J[0][1]*J[1][0])/det_J;

      // Reference tet: v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), v3=(0,0,1)
      // Face k is opposite vertex k.
      // We parameterise each face with two reference tangents.
      RealType ref_orig[3], ref_t1[3], ref_t2[3], ref_n[3];
      switch (face) {
        case 0: // opposite v0: triangle v1,v2,v3
          ref_orig[0]=1.0; ref_orig[1]=0.0; ref_orig[2]=0.0;
          ref_t1[0]=-1.0; ref_t1[1]=1.0; ref_t1[2]=0.0;
          ref_t2[0]=-1.0; ref_t2[1]=0.0; ref_t2[2]=1.0;
          ref_n[0]=1.0; ref_n[1]=1.0; ref_n[2]=1.0;
          break;
        case 1: // opposite v1: triangle v0,v2,v3
          ref_orig[0]=0.0; ref_orig[1]=0.0; ref_orig[2]=0.0;
          ref_t1[0]=0.0; ref_t1[1]=1.0; ref_t1[2]=0.0;
          ref_t2[0]=0.0; ref_t2[1]=0.0; ref_t2[2]=1.0;
          ref_n[0]=-1.0; ref_n[1]=0.0; ref_n[2]=0.0;
          break;
        case 2: // opposite v2: triangle v0,v1,v3
          ref_orig[0]=0.0; ref_orig[1]=0.0; ref_orig[2]=0.0;
          ref_t1[0]=1.0; ref_t1[1]=0.0; ref_t1[2]=0.0;
          ref_t2[0]=0.0; ref_t2[1]=0.0; ref_t2[2]=1.0;
          ref_n[0]=0.0; ref_n[1]=-1.0; ref_n[2]=0.0;
          break;
        case 3: // opposite v3: triangle v0,v1,v2
          ref_orig[0]=0.0; ref_orig[1]=0.0; ref_orig[2]=0.0;
          ref_t1[0]=1.0; ref_t1[1]=0.0; ref_t1[2]=0.0;
          ref_t2[0]=0.0; ref_t2[1]=1.0; ref_t2[2]=0.0;
          ref_n[0]=0.0; ref_n[1]=0.0; ref_n[2]=-1.0;
          break;
        default:
          for(int d=0;d<3;d++) ref_orig[d]=ref_t1[d]=ref_t2[d]=ref_n[d]=0.0;
      }

      // Physical tangents: t_phys = J * ref_tangent
      RealType t1p[3], t2p[3];
      for (unsigned int d = 0; d < 3; ++d) {
        t1p[d] = J[d][0]*ref_t1[0] + J[d][1]*ref_t1[1] + J[d][2]*ref_t1[2];
        t2p[d] = J[d][0]*ref_t2[0] + J[d][1]*ref_t2[1] + J[d][2]*ref_t2[2];
      }

      // Cross product gives un-normalised face area vector
      RealType cross[3];
      cross[0] = t1p[1]*t2p[2] - t1p[2]*t2p[1];
      cross[1] = t1p[2]*t2p[0] - t1p[0]*t2p[2];
      cross[2] = t1p[0]*t2p[1] - t1p[1]*t2p[0];
      const RealType cross_mag = std::sqrt(cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2]);

      // Physical outward normal via cofactor map: n_phys = J^{-T} * ref_n
      RealType n_phys[3];
      n_phys[0] = Ji[0][0]*ref_n[0] + Ji[1][0]*ref_n[1] + Ji[2][0]*ref_n[2];
      n_phys[1] = Ji[0][1]*ref_n[0] + Ji[1][1]*ref_n[1] + Ji[2][1]*ref_n[2];
      n_phys[2] = Ji[0][2]*ref_n[0] + Ji[1][2]*ref_n[1] + Ji[2][2]*ref_n[2];
      const RealType n_mag = std::sqrt(n_phys[0]*n_phys[0]+n_phys[1]*n_phys[1]+n_phys[2]*n_phys[2]);

      // 2D quad on the reference triangle face (s,t), s>=0, t>=0, s+t<=1
      for (unsigned int q = 0; q < n_q_; ++q) {
        const auto xi_face = quad_.point(q);
        const RealType s = xi_face(0), t = xi_face(1);

        RealType xi_ref[3];
        for (unsigned int d = 0; d < 3; ++d)
          xi_ref[d] = ref_orig[d] + s*ref_t1[d] + t*ref_t2[d];

        Tensor<1, 3, RealType> xi;
        xi(0)=xi_ref[0]; xi(1)=xi_ref[1]; xi(2)=xi_ref[2];

        // 0.5 * |cross| because quad_.weight already integrates over a unit triangle
        JxW_(q) = 0.5 * cross_mag * quad_.weight(q);

        for (unsigned int d = 0; d < 3; ++d)
          q_point_(q,d) = x0[d] + J[d][0]*xi_ref[0] + J[d][1]*xi_ref[1] + J[d][2]*xi_ref[2];

        for (unsigned int d = 0; d < 3; ++d)
          normal_(q,d) = n_phys[d] / n_mag;

        for (unsigned int i = 0; i < n_dofs_; ++i) {
          phi_(i,q) = fe_.shape_value(i, xi);
          const auto tmp = fe_.shape_gradient(i, xi);
          for (unsigned int d = 0; d < 3; ++d)
            grad_phi_(i,q,d) = Ji[0][d]*tmp(0) + Ji[1][d]*tmp(1) + Ji[2][d]*tmp(2);
        }
      }
    }
>>>>>>> 5737baa (update fe.hpp and read_gri.hpp to 3D. made adjustments to flux.cpp tests. Fully builds but aborts when finding a 3D mesh. (#90))
  }

  unsigned int n_dofs()     const { return n_dofs_; }
  unsigned int n_q_points() const { return n_q_; }
  RealType JxW(unsigned int q) { return JxW_(q); }

  Tensor<1, dim, RealType> q_point(unsigned int q)
  {
    Tensor<1, dim, RealType> p;
    for (unsigned int d = 0; d < dim; ++d) p(d) = q_point_(q, d);
    return p;
  }

  Tensor<1, dim, RealType> normal(unsigned int q) const
  {
    Tensor<1, dim, RealType> n;
    for (unsigned int d = 0; d < dim; ++d) n(d) = normal_(q, d);
    return n;
  }

  RealType shape_value(unsigned int i, unsigned int q) { return phi_(i, q); }

  Tensor<1, dim, RealType> shape_gradient(unsigned int i, unsigned int q)
  {
    Tensor<1, dim, RealType> g;
    for (unsigned int d = 0; d < dim; ++d) g(d) = grad_phi_(i, q, d);
    return g;
  }

private:
  const FE_DGQLegendre<dim, RealType>&    fe_;
  const QGaussSimplex<dim-1, RealType>&   quad_;
  unsigned int n_dofs_, n_q_;
  Kokkos::View<RealType*,   Layout, HostMemSpace> JxW_;
  Kokkos::View<RealType**,  Layout, HostMemSpace> q_point_;
  Kokkos::View<RealType**,  Layout, HostMemSpace> normal_;
  Kokkos::View<RealType**,  Layout, HostMemSpace> phi_;
  Kokkos::View<RealType***, Layout, HostMemSpace> grad_phi_;

  // 2D version unchanged from your original
  template<typename CellAccessor>
  void reinit_2d(const CellAccessor& cell, unsigned int face)
  {
    // ... your existing reinit body, unchanged ...
  }

  template<typename CellAccessor>
  void reinit_3d(const CellAccessor& cell, unsigned int face)
  {
    // Reference tet has 4 triangular faces.
    // Face k is opposite vertex k: its vertices are the other three.
    // Reference vertices: v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), v3=(0,0,1)
    ASSERT(face < 4, "3D tet has only 4 faces");

    // Build 3x3 cell Jacobian from edge vectors
    Tensor<2, 3, RealType> J;
    Tensor<1, 3, RealType> x0;
    for (unsigned int d = 0; d < 3; ++d) x0(d) = cell.vertex(0)(d);
    for (unsigned int d = 0; d < 3; ++d) {
      J(d, 0) = cell.vertex(1)(d) - cell.vertex(0)(d);
      J(d, 1) = cell.vertex(2)(d) - cell.vertex(0)(d);
      J(d, 2) = cell.vertex(3)(d) - cell.vertex(0)(d);
    }
    const RealType det_J = det(J);
    const Tensor<2, 3, RealType> J_inv = inverse(J);

    // Reference face: origin + two tangent vectors + outward normal
    // Each face is parameterised by 2D coords (s,t) in [0,1]^triangle
    // mapped to 3D reference coords via origin + s*t1 + t*t2
    RealType ref_origin[3], ref_t1[3], ref_t2[3], ref_normal[3];

    switch (face) {
      case 0: {
        // Face opp v0: vertices v1,v2,v3
        ref_origin[0]=1.0; ref_origin[1]=0.0; ref_origin[2]=0.0;
        ref_t1[0]=-1.0; ref_t1[1]=1.0; ref_t1[2]=0.0; // v1->v2
        ref_t2[0]=-1.0; ref_t2[1]=0.0; ref_t2[2]=1.0; // v1->v3
        // Outward normal (pointing away from v0=(0,0,0)): (1,1,1)/sqrt(3)
        ref_normal[0]=1.0; ref_normal[1]=1.0; ref_normal[2]=1.0;
        break;
      }
      case 1: {
        // Face opp v1: vertices v0,v2,v3
        ref_origin[0]=0.0; ref_origin[1]=0.0; ref_origin[2]=0.0;
        ref_t1[0]=0.0; ref_t1[1]=1.0; ref_t1[2]=0.0; // v0->v2
        ref_t2[0]=0.0; ref_t2[1]=0.0; ref_t2[2]=1.0; // v0->v3
        // Outward normal: (-1,0,0)
        ref_normal[0]=-1.0; ref_normal[1]=0.0; ref_normal[2]=0.0;
        break;
      }
      case 2: {
        // Face opp v2: vertices v0,v1,v3
        ref_origin[0]=0.0; ref_origin[1]=0.0; ref_origin[2]=0.0;
        ref_t1[0]=1.0; ref_t1[1]=0.0; ref_t1[2]=0.0; // v0->v1
        ref_t2[0]=0.0; ref_t2[1]=0.0; ref_t2[2]=1.0; // v0->v3
        // Outward normal: (0,-1,0)
        ref_normal[0]=0.0; ref_normal[1]=-1.0; ref_normal[2]=0.0;
        break;
      }
      case 3: {
        // Face opp v3: vertices v0,v1,v2
        ref_origin[0]=0.0; ref_origin[1]=0.0; ref_origin[2]=0.0;
        ref_t1[0]=1.0; ref_t1[1]=0.0; ref_t1[2]=0.0; // v0->v1
        ref_t2[0]=0.0; ref_t2[1]=1.0; ref_t2[2]=0.0; // v0->v2
        // Outward normal: (0,0,-1)
        ref_normal[0]=0.0; ref_normal[1]=0.0; ref_normal[2]=-1.0;
        break;
      }
    }

    // Physical tangents: t_phys = J * t_ref
    RealType t1_phys[3] = {0,0,0}, t2_phys[3] = {0,0,0};
    for (unsigned int d = 0; d < 3; ++d)
      for (unsigned int e = 0; e < 3; ++e) {
        t1_phys[d] += J(d, e) * ref_t1[e];
        t2_phys[d] += J(d, e) * ref_t2[e];
      }

    // Physical normal via cross product of physical tangents, then normalise
    RealType n_phys[3] = {
      t1_phys[1]*t2_phys[2] - t1_phys[2]*t2_phys[1],
      t1_phys[2]*t2_phys[0] - t1_phys[0]*t2_phys[2],
      t1_phys[0]*t2_phys[1] - t1_phys[1]*t2_phys[0]
    };

    // Ensure outward orientation: check sign against reference normal
    RealType ref_dot = 0;
    for (unsigned int d = 0; d < 3; ++d) ref_dot += n_phys[d] * ref_normal[d];
    if (ref_dot < 0)
      for (unsigned int d = 0; d < 3; ++d) n_phys[d] = -n_phys[d];

    const RealType n_norm = std::sqrt(n_phys[0]*n_phys[0] + n_phys[1]*n_phys[1] + n_phys[2]*n_phys[2]);

    // Face area scaling: |t1_phys × t2_phys| is already n_norm above.
    // The 2D quadrature weight integrates over the reference triangle (area=0.5).
    // JxW = n_norm * w_q accounts for the surface Jacobian.
    const RealType face_jac = n_norm;

    for (unsigned int q = 0; q < n_q_; ++q) {
      const auto xi_face = quad_.point(q); // 2D point on reference triangle face

      // Map 2D face coords to 3D reference tet coords
      RealType xi_ref[3];
      for (unsigned int d = 0; d < 3; ++d)
        xi_ref[d] = ref_origin[d] + xi_face(0)*ref_t1[d] + xi_face(1)*ref_t2[d];

      Tensor<1, 3, RealType> xi;
      for (unsigned int d = 0; d < 3; ++d) xi(d) = xi_ref[d];

      JxW_(q) = face_jac * quad_.weight(q);

      for (unsigned int d = 0; d < 3; ++d)
        q_point_(q, d) = x0(d) + J(d,0)*xi_ref[0] + J(d,1)*xi_ref[1] + J(d,2)*xi_ref[2];

      for (unsigned int d = 0; d < 3; ++d)
        normal_(q, d) = n_phys[d] / n_norm;

      for (unsigned int i = 0; i < n_dofs_; ++i) {
        phi_(i, q) = fe_.shape_value(i, xi);
        const auto tmp = fe_.shape_gradient(i, xi);
        for (unsigned int d = 0; d < 3; ++d) {
          RealType s = 0;
          for (unsigned int e = 0; e < 3; ++e)
            s += J_inv(e, d) * tmp(e);
          grad_phi_(i, q, d) = s;
        }
      }
    }
  }
};