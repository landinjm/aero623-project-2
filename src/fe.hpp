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
class FEValues
{
public:
  FEValues(const FE_DGLagrangeSimplex<dim, RealType>& fe,
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

      J_inv[0][0] = J[1][1] / det_J;
      J_inv[0][1] = -J[0][1] / det_J;
      J_inv[1][0] = -J[1][0] / det_J;
      J_inv[1][1] = J[0][0] / det_J;
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
    } else {
      static_assert(dim == 2 || dim == 3,
                    "Only dim = 2 and dim = 3 are supported");
    }

    for (unsigned int q = 0; q < n_q_; ++q) {
      JxW_(q) = std::abs(det_J) * quad_.weight(q);

      const auto xi = quad_.point(q);

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

      for (unsigned int d = 0; d < dim; ++d) {
        RealType xq = x0[d];
        for (unsigned int k = 0; k < dim; ++k) {
          xq += J[d][k] * xi(k);
        }
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
  const FE_DGLagrangeSimplex<dim, RealType>& fe_;
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
  FEFaceValues(const FE_DGLagrangeSimplex<dim, RealType>& fe,
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
    ASSERT(face < SimplexTopology<dim>::faces_per_cell,
           "Local face number must be less than the number of faces per cell");

    // Sort the face vertices with some global ordering to preserve quad point
    // locations across shared faces.
    const unsigned int n_face_verts = SimplexTopology<dim>::verts_per_face;
    std::array<unsigned int, dim> global_ids;
    for (unsigned int v = 0; v < n_face_verts; ++v) {
      global_ids[v] = cell.face(face).vertex_index(v);
    }

    // Canonical = sorted ascending
    std::array<unsigned int, dim> sorted_ids = global_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());

    // Compute the permutation: where does each sorted vertex appear in local
    // order?
    std::array<unsigned int, dim> perm;
    for (unsigned int v = 0; v < dim; ++v) {
      perm[v] = std::find(global_ids.begin(), global_ids.end(), sorted_ids[v]) -
                global_ids.begin();
    }

    // Physical positions of face vertices in canonical order
    RealType fv[dim][dim]; // fv[v][coord]
    for (unsigned int v = 0; v < dim; ++v) {
      // perm[v] gives the local face-vertex index corresponding to canonical v
      for (unsigned int d = 0; d < dim; ++d) {
        fv[v][d] = cell.face(face).vertex(perm[v])(d);
      }
    }

    // Canonical tangents in physical space
    // t0 = fv[1] - fv[0],  t1 = fv[2] - fv[0]  (3D)
    RealType t_canon[dim - 1][dim];
    for (unsigned int s = 0; s < dim - 1; ++s)
      for (unsigned int d = 0; d < dim; ++d)
        t_canon[s][d] = fv[s + 1][d] - fv[0][d];

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

      J_inv[0][0] = J[1][1] / det_J;
      J_inv[0][1] = -J[0][1] / det_J;
      J_inv[1][0] = -J[1][0] / det_J;
      J_inv[1][1] = J[0][0] / det_J;
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
    } else {
      static_assert(dim == 2 || dim == 3,
                    "Only dim = 2 and dim = 3 are supported");
    }

    // Here is where things become a little different than FEValues.
    // In 2d, a face is an edge which has a 1D quadrature rule and 2D normal
    // vector. In 3d, a face is a triangle which has a 2D quadrature rule and 3D
    // normal vector. We have to account for this mapping

    RealType ref_origin[dim];
    RealType ref_tangent[dim - 1][dim];
    RealType ref_normal[dim];

    if constexpr (dim == 2) {
      switch (face) {
        case 0: {
          // Face 0 -> v1=(1,0) and v2=(0,1)
          ref_origin[0] = 1.0;
          ref_origin[1] = 0.0;
          ref_tangent[0][0] = -1.0;
          ref_tangent[0][1] = 1.0;
          ref_normal[0] = 1.0;
          ref_normal[1] = 1.0;
          break;
        }
        case 1: {
          // Face 1 -> v2=(0,1) and v0=(0,0)
          ref_origin[0] = 0.0;
          ref_origin[1] = 1.0;
          ref_tangent[0][0] = 0.0;
          ref_tangent[0][1] = -1.0;
          ref_normal[0] = -1.0;
          ref_normal[1] = 0.0;
          break;
        }
        case 2: {
          // Face 2 -> v0=(0,0) and v1=(1,0)
          ref_origin[0] = 0.0;
          ref_origin[1] = 0.0;
          ref_tangent[0][0] = 1.0;
          ref_tangent[0][1] = 0.0;
          ref_normal[0] = 0.0;
          ref_normal[1] = -1.0;
          break;
        }
      }
    } else if constexpr (dim == 3) {
      switch (face) {
        case 0: {
          // Face 0 -> v1=(1,0,0), v2=(0,1,0), and v3=(0,0,1)
          ref_origin[0] = 1.0;
          ref_origin[1] = 0.0;
          ref_origin[2] = 0.0;
          ref_tangent[0][0] = -1.0;
          ref_tangent[0][1] = 1.0;
          ref_tangent[0][2] = 0.0;
          ref_tangent[1][0] = -1.0;
          ref_tangent[1][1] = 0.0;
          ref_tangent[1][2] = 1.0;
          ref_normal[0] = 1.0;
          ref_normal[1] = 1.0;
          ref_normal[2] = 1.0;
          break;
        }
        case 1: {
          // Face 1 -> v0=(0,0,0), v2=(0,1,0), and v3=(0,0,1)
          ref_origin[0] = 0.0;
          ref_origin[1] = 0.0;
          ref_origin[2] = 0.0;
          ref_tangent[0][0] = 0.0;
          ref_tangent[0][1] = 1.0;
          ref_tangent[0][2] = 0.0;
          ref_tangent[1][0] = 0.0;
          ref_tangent[1][1] = 0.0;
          ref_tangent[1][2] = 1.0;
          ref_normal[0] = -1.0;
          ref_normal[1] = 0.0;
          ref_normal[2] = 0.0;
          break;
        }
        case 2: {
          // Face 2 -> v0=(0,0,0), v1=(1,0,0), and v3=(0,0,1)
          ref_origin[0] = 0.0;
          ref_origin[1] = 0.0;
          ref_origin[2] = 0.0;
          ref_tangent[0][0] = 1.0;
          ref_tangent[0][1] = 0.0;
          ref_tangent[0][2] = 0.0;
          ref_tangent[1][0] = 0.0;
          ref_tangent[1][1] = 0.0;
          ref_tangent[1][2] = 1.0;
          ref_normal[0] = 0.0;
          ref_normal[1] = -1.0;
          ref_normal[2] = 0.0;
          break;
        }
        case 3: {
          // Face 3 -> v0=(0,0,0), v1=(1,0,0), and v2=(0,1,0)
          ref_origin[0] = 0.0;
          ref_origin[1] = 0.0;
          ref_origin[2] = 0.0;
          ref_tangent[0][0] = 1.0;
          ref_tangent[0][1] = 0.0;
          ref_tangent[0][2] = 0.0;
          ref_tangent[1][0] = 0.0;
          ref_tangent[1][1] = 1.0;
          ref_tangent[1][2] = 0.0;
          ref_normal[0] = 0.0;
          ref_normal[1] = 0.0;
          ref_normal[2] = -1.0;
          break;
        }
      }
    }

    // Now grab the physical normal and tangent
    RealType n_phys[dim] = {};
    for (unsigned int d = 0; d < dim; ++d) {
      for (unsigned int k = 0; k < dim; ++k) {
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

      // Physical quad point via canonical face parameterization
      RealType x_phys[dim];
      for (unsigned int d = 0; d < dim; ++d) {
        x_phys[d] = fv[0][d];
        for (unsigned int s = 0; s < dim - 1; ++s)
          x_phys[d] += xi_face(s) * t_canon[s][d];
      }

      // Map x_phys back to reference cell coords via J_inv
      RealType xi_ref[dim];
      for (unsigned int d = 0; d < dim; ++d) {
        xi_ref[d] = 0.0;
        for (unsigned int k = 0; k < dim; ++k)
          xi_ref[d] += J_inv[d][k] * (x_phys[k] - x0[k]);
      }

      // Wrap in a Tensor
      Tensor<1, dim, RealType> xi;
      for (unsigned int d = 0; d < dim; ++d) {
        xi(d) = xi_ref[d];
      }

      JxW_(q) = face_jac * quad_.weight(q);

      for (unsigned int d = 0; d < dim; ++d) {
        q_point_(q, d) = x0[d];
        for (unsigned int k = 0; k < dim; ++k) {
          q_point_(q, d) += J[d][k] * xi_ref[k];
        }
      }

      for (unsigned int d = 0; d < dim; ++d) {
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
  const FE_DGLagrangeSimplex<dim, RealType>& fe_;
  const QGaussSimplex<dim - 1, RealType>& quad_;

  unsigned int n_dofs_;
  unsigned int n_q_;

  Kokkos::View<RealType*, Layout, HostMemSpace> JxW_;        // [q]
  Kokkos::View<RealType**, Layout, HostMemSpace> q_point_;   // [q, dim]
  Kokkos::View<RealType**, Layout, HostMemSpace> normal_;    // [q, dim]
  Kokkos::View<RealType**, Layout, HostMemSpace> phi_;       // [dof, q]
  Kokkos::View<RealType***, Layout, HostMemSpace> grad_phi_; // [dof, q, dim]
};
