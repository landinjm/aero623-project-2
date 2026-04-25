#include <dof_handler.hpp>
#include <fe.hpp>
#include <random>
#include <read_gri.hpp>
#include <triangulation.hpp>

#include <gtest/gtest.h>

using RealType = double;
static constexpr RealType tol = 1.0e-10;

static constexpr unsigned int max_order = 10;

template<unsigned int dim>
std::vector<Tensor<1, dim, RealType>>
random_simplex_points(unsigned int n, unsigned int seed = 42)
{
  std::mt19937 rng(seed);
  std::exponential_distribution<RealType> dist(1.0);

  std::vector<Tensor<1, dim, RealType>> pts(n);
  for (auto& pt : pts) {
    std::array<RealType, dim + 1> e;
    for (auto& ei : e) {
      ei = dist(rng);
    }
    RealType sum = 0;
    for (auto ei : e) {
      sum += ei;
    }
    for (unsigned int d = 0; d < dim; ++d) {
      pt(d) = e[d] / sum;
    }
  }
  return pts;
}

template<unsigned int dim>
Tensor<1, dim, RealType>
fd_gradient(const FE_DGLagrangeSimplex<dim, RealType>& fe,
            unsigned int i,
            const Tensor<1, dim, RealType>& pt)
{
  const RealType pt_tol = RealType(1.0e-7);
  Tensor<1, dim, RealType> grad;
  for (unsigned int d = 0; d < dim; ++d) {
    Tensor<1, dim, RealType> pt_fwd = pt;
    Tensor<1, dim, RealType> pt_bwd = pt;
    pt_fwd(d) += pt_tol;
    pt_bwd(d) -= pt_tol;
    grad(d) =
      (fe.shape_value(i, pt_fwd) - fe.shape_value(i, pt_bwd)) / (2 * pt_tol);
  }
  return grad;
}

TEST(FE_DGLagrangeSimplex, 1D_DoF_counts)
{
  for (unsigned int order = 0; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<1, RealType> fe(order);
    EXPECT_EQ(fe.n_dofs(), order + 1);
  }
}

TEST(FE_DGLagrangeSimplex, 2D_DoF_counts)
{
  for (unsigned int order = 0; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<2, RealType> fe(order);
    EXPECT_EQ(fe.n_dofs(), (order + 1) * (order + 2) / 2);
  }
}

TEST(FE_DGLagrangeSimplex, 3D_DoF_counts)
{
  for (unsigned int order = 0; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<3, RealType> fe(order);
    EXPECT_EQ(fe.n_dofs(), (order + 1) * (order + 2) * (order + 3) / 6);
  }
}

TEST(FE_DGLagrangeSimplex, 1D_KroneckerDelta)
{
  for (unsigned int order = 0; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<1, RealType> fe(order);
    for (unsigned int i = 0; i < fe.n_dofs(); ++i) {
      for (unsigned int j = 0; j < fe.n_dofs(); ++j) {
        EXPECT_NEAR(fe.shape_value(i, fe.node(j)), (i == j) ? 1.0 : 0.0, tol);
      }
    }
  }
}

TEST(FE_DGLagrangeSimplex, 2D_KroneckerDelta)
{
  for (unsigned int order = 0; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<2, RealType> fe(order);
    for (unsigned int i = 0; i < fe.n_dofs(); ++i) {
      for (unsigned int j = 0; j < fe.n_dofs(); ++j) {
        EXPECT_NEAR(fe.shape_value(i, fe.node(j)), (i == j) ? 1.0 : 0.0, tol);
      }
    }
  }
}

TEST(FE_DGLagrangeSimplex, 3D_KroneckerDelta)
{
  for (unsigned int order = 0; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<3, RealType> fe(order);
    for (unsigned int i = 0; i < fe.n_dofs(); ++i) {
      for (unsigned int j = 0; j < fe.n_dofs(); ++j) {
        EXPECT_NEAR(fe.shape_value(i, fe.node(j)), (i == j) ? 1.0 : 0.0, tol);
      }
    }
  }
}

TEST(FE_DGLagrangeSimplex, 1D_PartitionOfUnity)
{
  auto pts = random_simplex_points<1>(1000);
  for (unsigned int order = 0; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<1, RealType> fe(order);
    for (const auto& pt : pts) {
      RealType sum = 0.0;
      for (unsigned int i = 0; i < fe.n_dofs(); ++i) {
        sum += fe.shape_value(i, pt);
      }
      EXPECT_NEAR(sum, 1.0, tol);
    }
  }
}

TEST(FE_DGLagrangeSimplex, 2D_PartitionOfUnity)
{
  auto pts = random_simplex_points<2>(1000);
  for (unsigned int order = 0; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<2, RealType> fe(order);
    for (const auto& pt : pts) {
      RealType sum = 0.0;
      for (unsigned int i = 0; i < fe.n_dofs(); ++i) {
        sum += fe.shape_value(i, pt);
      }
      EXPECT_NEAR(sum, 1.0, tol);
    }
  }
}

TEST(FE_DGLagrangeSimplex, 3D_PartitionOfUnity)
{
  auto pts = random_simplex_points<3>(1000);
  for (unsigned int order = 0; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<3, RealType> fe(order);
    for (const auto& pt : pts) {
      RealType sum = 0.0;
      for (unsigned int i = 0; i < fe.n_dofs(); ++i) {
        sum += fe.shape_value(i, pt);
      }
      EXPECT_NEAR(sum, 1.0, tol);
    }
  }
}

TEST(FE_DGLagrangeSimplex, 1D_Gradient_FiniteDifference)
{
  auto pts = random_simplex_points<1>(200);
  for (unsigned int order = 1; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<1, RealType> fe(order);
    for (const auto& pt : pts) {
      for (unsigned int i = 0; i < fe.n_dofs(); ++i) {
        auto grad = fe.shape_gradient(i, pt);
        auto fd_grad = fd_gradient<1>(fe, i, pt);
        for (unsigned int d = 0; d < 1; ++d) {
          EXPECT_NEAR(grad(d), fd_grad(d), 1.0e-5);
        }
      }
    }
  }
}

TEST(FE_DGLagrangeSimplex, 2D_Gradient_FiniteDifference)
{
  auto pts = random_simplex_points<2>(200);
  for (unsigned int order = 1; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<2, RealType> fe(order);
    for (const auto& pt : pts) {
      for (unsigned int i = 0; i < fe.n_dofs(); ++i) {
        auto grad = fe.shape_gradient(i, pt);
        auto fd_grad = fd_gradient<2>(fe, i, pt);
        for (unsigned int d = 0; d < 2; ++d) {
          EXPECT_NEAR(grad(d), fd_grad(d), 1.0e-5);
        }
      }
    }
  }
}

TEST(FE_DGLagrangeSimplex, 3D_Gradient_FiniteDifference)
{
  auto pts = random_simplex_points<3>(200);
  for (unsigned int order = 1; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<3, RealType> fe(order);
    for (const auto& pt : pts) {
      for (unsigned int i = 0; i < fe.n_dofs(); ++i) {
        auto grad = fe.shape_gradient(i, pt);
        auto fd_grad = fd_gradient<3>(fe, i, pt);
        for (unsigned int d = 0; d < 3; ++d) {
          EXPECT_NEAR(grad(d), fd_grad(d), 1.0e-5);
        }
      }
    }
  }
}

TEST(FEValues, 2D)
{
  constexpr unsigned int dim = 2;
  constexpr unsigned int mesh_q = 1;

  Triangulation<dim, mesh_q> tria;
  GriReader<dim, mesh_q> gri;
  gri.read_gri("../tests/test_5.gri");
  gri.transfer_to_triangulation(tria);

  for (unsigned int order = 0; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<dim, RealType> fe(order);
    QGaussSimplex<dim, RealType> quad(order + 1);
    FEValues<dim, mesh_q, RealType> fe_values(fe, quad);

    for (const auto& cell : tria.active_cell_range()) {
      fe_values.reinit(cell);

      const unsigned int n_q = fe_values.n_q_points();
      const unsigned int n_dof = fe_values.n_dofs();

      for (unsigned int q = 0; q < n_q; ++q) {

        // Quad points are finite
        for (unsigned int d = 0; d < dim; ++d) {
          EXPECT_TRUE(std::isfinite(fe_values.q_point(q)(d)));
        }

        // Sum of shape values is 1
        RealType phi_sum = 0.0;
        for (unsigned int i = 0; i < n_dof; ++i) {
          phi_sum += fe_values.shape_value(i, q);
        }
        EXPECT_NEAR(phi_sum, 1.0, tol);

        // Gradients are finite
        for (unsigned int i = 0; i < n_dof; ++i) {
          for (unsigned int d = 0; d < dim; ++d) {
            EXPECT_TRUE(std::isfinite(fe_values.shape_gradient(i, q)(d)));
          }
        }
      }

      // Sum of shape gradients are 0 at every quad
      for (unsigned int q = 0; q < n_q; ++q) {
        for (unsigned int d = 0; d < dim; ++d) {
          RealType grad_sum = 0.0;
          for (unsigned int i = 0; i < n_dof; ++i) {
            grad_sum += fe_values.shape_gradient(i, q)(d);
          }
          EXPECT_NEAR(grad_sum, 0.0, tol);
        }
      }

      // Gradient consistency
      if (order == 1) {
        for (unsigned int i = 0; i < n_dof; ++i) {
          for (unsigned int j = 1; j <= dim; ++j) {
            RealType dot = 0.0;
            for (unsigned int d = 0; d < dim; ++d) {
              dot += fe_values.shape_gradient(i, 0)(d) *
                     (cell.vertex(j)(d) - cell.vertex(0)(d));
            }
            const RealType expected = (i == j) ? 1.0 : (i == 0 ? -1.0 : 0.0);
            EXPECT_NEAR(dot, expected, tol);
          }
        }
      }
    }
  }
}

TEST(FEValues, 3D)
{
  constexpr unsigned int dim = 3;
  constexpr unsigned int mesh_q = 1;

  Triangulation<dim, mesh_q> tria;
  GriReader<dim, mesh_q> gri;
  gri.read_gri("../tests/test3D.gri");
  gri.transfer_to_triangulation(tria);

  for (unsigned int order = 0; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<dim, RealType> fe(order);
    QGaussSimplex<dim, RealType> quad(order + 1);
    FEValues<dim, mesh_q, RealType> fe_values(fe, quad);

    for (const auto& cell : tria.active_cell_range()) {
      fe_values.reinit(cell);

      const unsigned int n_q = fe_values.n_q_points();
      const unsigned int n_dof = fe_values.n_dofs();

      for (unsigned int q = 0; q < n_q; ++q) {

        // Quad points are finite
        for (unsigned int d = 0; d < dim; ++d) {
          EXPECT_TRUE(std::isfinite(fe_values.q_point(q)(d)));
        }

        // Sum of shape values is 1
        RealType phi_sum = 0.0;
        for (unsigned int i = 0; i < n_dof; ++i) {
          phi_sum += fe_values.shape_value(i, q);
        }
        EXPECT_NEAR(phi_sum, 1.0, tol);

        // Gradients are finite
        for (unsigned int i = 0; i < n_dof; ++i) {
          for (unsigned int d = 0; d < dim; ++d) {
            EXPECT_TRUE(std::isfinite(fe_values.shape_gradient(i, q)(d)));
          }
        }
      }

      // Sum of shape gradients are 0 at every quad
      for (unsigned int q = 0; q < n_q; ++q) {
        for (unsigned int d = 0; d < dim; ++d) {
          RealType grad_sum = 0.0;
          for (unsigned int i = 0; i < n_dof; ++i) {
            grad_sum += fe_values.shape_gradient(i, q)(d);
          }
          EXPECT_NEAR(grad_sum, 0.0, tol);
        }
      }

      // Gradient consistency
      if (order == 1) {
        for (unsigned int i = 0; i < n_dof; ++i) {
          for (unsigned int j = 1; j <= dim; ++j) {
            RealType dot = 0.0;
            for (unsigned int d = 0; d < dim; ++d) {
              dot += fe_values.shape_gradient(i, 0)(d) *
                     (cell.vertex(j)(d) - cell.vertex(0)(d));
            }
            const RealType expected = (i == j) ? 1.0 : (i == 0 ? -1.0 : 0.0);
            EXPECT_NEAR(dot, expected, tol);
          }
        }
      }
    }
  }
}

TEST(FEFaceValues, 2D)
{
  constexpr unsigned int dim = 2;
  constexpr unsigned int mesh_q = 1;

  Triangulation<dim, mesh_q> tria;
  GriReader<dim, mesh_q> gri;
  gri.read_gri("../tests/test_5.gri");
  gri.transfer_to_triangulation(tria);

  for (unsigned int order = 0; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<dim, RealType> fe(order);
    QGaussSimplex<dim - 1, RealType> quad(order + 1);
    FEFaceValues<dim, mesh_q, RealType> fe_face_values(fe, quad);

    for (const auto& cell : tria.active_cell_range()) {
      for (unsigned int f = 0; f < SimplexTopology<dim, 1>::faces_per_cell; ++f) {
        fe_face_values.reinit(cell, f);

        const unsigned int n_q = fe_face_values.n_q_points();
        const unsigned int n_dof = fe_face_values.n_dofs();

        for (unsigned int q = 0; q < n_q; ++q) {

          // Quad points are finite
          for (unsigned int d = 0; d < dim; ++d) {
            EXPECT_TRUE(std::isfinite(fe_face_values.q_point(q)(d)));
          }

          // Normal is finite
          for (unsigned int d = 0; d < dim; ++d) {
            EXPECT_TRUE(std::isfinite(fe_face_values.normal(q)(d)));
          }

          // Normal is unit length
          EXPECT_NEAR(fe_face_values.normal(q).norm(), 1.0, tol);

          // Sum of shape values is 1
          RealType phi_sum = 0.0;
          for (unsigned int i = 0; i < n_dof; ++i) {
            phi_sum += fe_face_values.shape_value(i, q);
          }
          EXPECT_NEAR(phi_sum, 1.0, tol);

          // Gradients are finite
          for (unsigned int i = 0; i < n_dof; ++i) {
            for (unsigned int d = 0; d < dim; ++d) {
              EXPECT_TRUE(
                std::isfinite(fe_face_values.shape_gradient(i, q)(d)));
            }
          }
        }

        // Sum of shape gradients are 0 at every quad
        for (unsigned int q = 0; q < n_q; ++q) {
          for (unsigned int d = 0; d < dim; ++d) {
            RealType grad_sum = 0.0;
            for (unsigned int i = 0; i < n_dof; ++i) {
              grad_sum += fe_face_values.shape_gradient(i, q)(d);
            }
            EXPECT_NEAR(grad_sum, 0.0, tol);
          }
        }

        // Normal is constant across quad points on a flat face
        for (unsigned int q = 1; q < n_q; ++q) {
          for (unsigned int d = 0; d < dim; ++d) {
            EXPECT_NEAR(
              fe_face_values.normal(q)(d), fe_face_values.normal(0)(d), tol);
          }
        }

        // Quad points lie on the correct physical face
        {
          static constexpr unsigned int face_vertices[3][2] = { { 1, 2 },
                                                                { 2, 0 },
                                                                { 0, 1 } };
          const auto v0 = cell.vertex(face_vertices[f][0]);
          const auto v1 = cell.vertex(face_vertices[f][1]);

          for (unsigned int q = 0; q < n_q; ++q) {
            RealType dot = 0.0;
            for (unsigned int d = 0; d < dim; ++d) {
              dot += (fe_face_values.q_point(q)(d) - v0(d)) *
                     fe_face_values.normal(0)(d);
            }
            EXPECT_NEAR(dot, 0.0, tol);

            const RealType edge_len_sq = [&]() {
              RealType s = 0.0;
              for (unsigned int d = 0; d < dim; ++d) {
                s += (v1(d) - v0(d)) * (v1(d) - v0(d));
              }
              return s;
            }();

            RealType t = 0.0;
            for (unsigned int d = 0; d < dim; ++d) {
              t += (fe_face_values.q_point(q)(d) - v0(d)) * (v1(d) - v0(d));
            }
            t /= edge_len_sq;
            EXPECT_GE(t, -tol);
            EXPECT_LE(t, 1.0 + tol);
          }
        }
      }

      // Divergence theorem
      for (unsigned int d = 0; d < dim; ++d) {
        RealType flux = 0.0;
        for (unsigned int f = 0; f < SimplexTopology<dim, mesh_q>::faces_per_cell;
             ++f) {
          fe_face_values.reinit(cell, f);
          for (unsigned int q = 0; q < fe_face_values.n_q_points(); ++q) {
            flux += fe_face_values.normal(q)(d) * fe_face_values.JxW(q);
          }
        }
        EXPECT_NEAR(flux, 0.0, tol);
      }
    }
  }
}

TEST(FEFaceValues, 3D)
{
  constexpr unsigned int dim = 3;
  constexpr unsigned int mesh_q = 1;

  Triangulation<dim, mesh_q> tria;
  GriReader<dim, mesh_q> gri;
  gri.read_gri("../tests/test3D.gri");
  gri.transfer_to_triangulation(tria);

  for (unsigned int order = 0; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<dim, RealType> fe(order);
    QGaussSimplex<dim - 1, RealType> quad(order + 1);
    FEFaceValues<dim, mesh_q, RealType> fe_face_values(fe, quad);

    for (const auto& cell : tria.active_cell_range()) {
      for (unsigned int f = 0; f < SimplexTopology<dim, mesh_q>::faces_per_cell; ++f) {
        fe_face_values.reinit(cell, f);

        const unsigned int n_q = fe_face_values.n_q_points();
        const unsigned int n_dof = fe_face_values.n_dofs();

        for (unsigned int q = 0; q < n_q; ++q) {

          // Quad points are finite
          for (unsigned int d = 0; d < dim; ++d) {
            EXPECT_TRUE(std::isfinite(fe_face_values.q_point(q)(d)));
          }

          // Normal is finite
          for (unsigned int d = 0; d < dim; ++d) {
            EXPECT_TRUE(std::isfinite(fe_face_values.normal(q)(d)));
          }

          // Normal is unit length
          EXPECT_NEAR(fe_face_values.normal(q).norm(), 1.0, tol);

          // Sum of shape values is 1
          RealType phi_sum = 0.0;
          for (unsigned int i = 0; i < n_dof; ++i) {
            phi_sum += fe_face_values.shape_value(i, q);
          }
          EXPECT_NEAR(phi_sum, 1.0, tol);

          // Gradients are finite
          for (unsigned int i = 0; i < n_dof; ++i) {
            for (unsigned int d = 0; d < dim; ++d) {
              EXPECT_TRUE(
                std::isfinite(fe_face_values.shape_gradient(i, q)(d)));
            }
          }
        }

        // Sum of shape gradients are 0 at every quad
        for (unsigned int q = 0; q < n_q; ++q) {
          for (unsigned int d = 0; d < dim; ++d) {
            RealType grad_sum = 0.0;
            for (unsigned int i = 0; i < n_dof; ++i) {
              grad_sum += fe_face_values.shape_gradient(i, q)(d);
            }
            EXPECT_NEAR(grad_sum, 0.0, tol);
          }
        }

        // Normal is constant across quad points on a flat face
        for (unsigned int q = 1; q < n_q; ++q) {
          for (unsigned int d = 0; d < dim; ++d) {
            EXPECT_NEAR(
              fe_face_values.normal(q)(d), fe_face_values.normal(0)(d), tol);
          }
        }

        // Quad points lie on the correct physical face
        {
          static constexpr unsigned int face_vertices[4][3] = {
            { 1, 2, 3 },
            { 0, 2, 3 },
            { 0, 1, 3 },
            { 0, 1, 2 },
          };
          const auto va = cell.vertex(face_vertices[f][0]);
          const auto vb = cell.vertex(face_vertices[f][1]);
          const auto vc = cell.vertex(face_vertices[f][2]);

          for (unsigned int q = 0; q < n_q; ++q) {
            RealType dot = 0.0;
            for (unsigned int d = 0; d < dim; ++d) {
              dot += (fe_face_values.q_point(q)(d) - va(d)) *
                     fe_face_values.normal(0)(d);
            }
            EXPECT_NEAR(dot, 0.0, tol);

            RealType e1[dim], e2[dim], r[dim];
            for (unsigned int d = 0; d < dim; ++d) {
              e1[d] = vb(d) - va(d);
              e2[d] = vc(d) - va(d);
              r[d] = fe_face_values.q_point(q)(d) - va(d);
            }
            const RealType e1e1 = [&] {
              RealType s = 0;
              for (unsigned int d = 0; d < dim; ++d) {
                s += e1[d] * e1[d];
              }
              return s;
            }();
            const RealType e1e2 = [&] {
              RealType s = 0;
              for (unsigned int d = 0; d < dim; ++d) {
                s += e1[d] * e2[d];
              }
              return s;
            }();
            const RealType e2e2 = [&] {
              RealType s = 0;
              for (unsigned int d = 0; d < dim; ++d) {
                s += e2[d] * e2[d];
              }
              return s;
            }();
            const RealType re1 = [&] {
              RealType s = 0;
              for (unsigned int d = 0; d < dim; ++d) {
                s += r[d] * e1[d];
              }
              return s;
            }();
            const RealType re2 = [&] {
              RealType s = 0;
              for (unsigned int d = 0; d < dim; ++d) {
                s += r[d] * e2[d];
              }
              return s;
            }();
            const RealType det = e1e1 * e2e2 - e1e2 * e1e2;
            const RealType s = (re1 * e2e2 - re2 * e1e2) / det;
            const RealType t = (re2 * e1e1 - re1 * e1e2) / det;
            EXPECT_GE(s, -tol);
            EXPECT_GE(t, -tol);
            EXPECT_LE(s + t, 1.0 + tol);
          }
        }
      }

      // Divergence theorem
      for (unsigned int d = 0; d < dim; ++d) {
        RealType flux = 0.0;
        for (unsigned int f = 0; f < SimplexTopology<dim, mesh_q>::faces_per_cell;
             ++f) {
          fe_face_values.reinit(cell, f);
          for (unsigned int q = 0; q < fe_face_values.n_q_points(); ++q) {
            flux += fe_face_values.normal(q)(d) * fe_face_values.JxW(q);
          }
        }
        EXPECT_NEAR(flux, 0.0, tol);
      }
    }
  }
}

TEST(FEValues, 3D_Q2)
{
  constexpr unsigned int dim = 3;
  constexpr unsigned int mesh_q = 2;

  Triangulation<dim, mesh_q> tria;
  GriReader<dim, mesh_q> gri;
  gri.read_gri("../airfoils/cube.gri");
  gri.transfer_to_triangulation(tria);

  for (unsigned int order = 0; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<dim, RealType> fe(order);
    QGaussSimplex<dim, RealType> quad(order + 1);
    FEValues<dim, mesh_q, RealType> fe_values(fe, quad);

    for (const auto& cell : tria.active_cell_range()) {
      fe_values.reinit(cell);

      const unsigned int n_q   = fe_values.n_q_points();
      const unsigned int n_dof = fe_values.n_dofs();

      // ------------------------------------------------------------------
      // Same sanity checks as Q1 — these must hold regardless of mesh_q
      // ------------------------------------------------------------------
      for (unsigned int q = 0; q < n_q; ++q) {

        // Quad points are finite
        for (unsigned int d = 0; d < dim; ++d)
          EXPECT_TRUE(std::isfinite(fe_values.q_point(q)(d)));

        // Partition of unity: sum of shape values == 1
        RealType phi_sum = 0.0;
        for (unsigned int i = 0; i < n_dof; ++i)
          phi_sum += fe_values.shape_value(i, q);
        EXPECT_NEAR(phi_sum, 1.0, tol);

        // Gradients are finite
        for (unsigned int i = 0; i < n_dof; ++i)
          for (unsigned int d = 0; d < dim; ++d)
            EXPECT_TRUE(std::isfinite(fe_values.shape_gradient(i, q)(d)));
      }

      // Sum of shape gradients == 0 at every quad point
      for (unsigned int q = 0; q < n_q; ++q) {
        for (unsigned int d = 0; d < dim; ++d) {
          RealType grad_sum = 0.0;
          for (unsigned int i = 0; i < n_dof; ++i)
            grad_sum += fe_values.shape_gradient(i, q)(d);
          EXPECT_NEAR(grad_sum, 0.0, tol);
        }
      }

      // Gradient consistency — uses only corner vertices (0..dim), which
      // exist identically in Q2, so this check is still valid.
      if (order == 1) {
        for (unsigned int i = 0; i < n_dof; ++i) {
          for (unsigned int j = 1; j <= dim; ++j) {
            RealType dot = 0.0;
            for (unsigned int d = 0; d < dim; ++d)
              dot += fe_values.shape_gradient(i, 0)(d) *
                     (cell.vertex(j)(d) - cell.vertex(0)(d));
            const RealType expected = (i == j) ? 1.0 : (i == 0 ? -1.0 : 0.0);
            EXPECT_NEAR(dot, expected, tol);
          }
        }
      }

      // ------------------------------------------------------------------
      // Q2-specific: JxW sum == cell volume.
      // For a unit cube split into 24 tets the individual cell volumes
      // vary, so we compare against cell.measure() which is computed
      // directly from the geometry and is independent of FEValues.
      // ------------------------------------------------------------------
      RealType jxw_sum = 0.0;
      for (unsigned int q = 0; q < n_q; ++q)
        jxw_sum += fe_values.JxW(q);
      EXPECT_NEAR(jxw_sum, cell.measure(), tol);
    }
  }
}

TEST(FEFaceValues, 3D_Q2)
{
  constexpr unsigned int dim = 3;
  constexpr unsigned int mesh_q = 2;

  Triangulation<dim, mesh_q> tria;
  GriReader<dim, mesh_q> gri;
  gri.read_gri("../airfoils/cube.gri");
  gri.transfer_to_triangulation(tria);

  for (unsigned int order = 0; order <= max_order; ++order) {
    FE_DGLagrangeSimplex<dim, RealType> fe(order);
    QGaussSimplex<dim - 1, RealType> quad(order + 1);
    FEFaceValues<dim, mesh_q, RealType> fe_face_values(fe, quad);

    for (const auto& cell : tria.active_cell_range()) {
      for (unsigned int f = 0;
           f < SimplexTopology<dim, mesh_q>::faces_per_cell; ++f) {
        fe_face_values.reinit(cell, f);

        const unsigned int n_q   = fe_face_values.n_q_points();
        const unsigned int n_dof = fe_face_values.n_dofs();

        // ----------------------------------------------------------------
        // Same sanity checks as Q1
        // ----------------------------------------------------------------
        for (unsigned int q = 0; q < n_q; ++q) {

          // Quad points are finite
          for (unsigned int d = 0; d < dim; ++d)
            EXPECT_TRUE(std::isfinite(fe_face_values.q_point(q)(d)));

          // Normal is finite
          for (unsigned int d = 0; d < dim; ++d)
            EXPECT_TRUE(std::isfinite(fe_face_values.normal(q)(d)));

          // Normal is unit length
          EXPECT_NEAR(fe_face_values.normal(q).norm(), 1.0, tol);

          // Partition of unity
          RealType phi_sum = 0.0;
          for (unsigned int i = 0; i < n_dof; ++i)
            phi_sum += fe_face_values.shape_value(i, q);
          EXPECT_NEAR(phi_sum, 1.0, tol);

          // Gradients are finite
          for (unsigned int i = 0; i < n_dof; ++i)
            for (unsigned int d = 0; d < dim; ++d)
              EXPECT_TRUE(
                std::isfinite(fe_face_values.shape_gradient(i, q)(d)));
        }

        // Sum of shape gradients == 0 at every quad point
        for (unsigned int q = 0; q < n_q; ++q) {
          for (unsigned int d = 0; d < dim; ++d) {
            RealType grad_sum = 0.0;
            for (unsigned int i = 0; i < n_dof; ++i)
              grad_sum += fe_face_values.shape_gradient(i, q)(d);
            EXPECT_NEAR(grad_sum, 0.0, tol);
          }
        }

        // ----------------------------------------------------------------
        // NOTE: The Q1 test checks that the normal is constant across all
        // quad points on a face. This is intentionally omitted here —
        // for Q2 curved faces the normal genuinely varies per point and
        // a constant normal would indicate a bug, not correctness.
        // ----------------------------------------------------------------

        // ----------------------------------------------------------------
        // NOTE: The Q1 test checks quad points lie on the affine plane
        // spanned by the three corner vertices. This is also intentionally
        // omitted — for curved Q2 faces the quad points follow the curved
        // geometry and will not lie on that plane.
        //
        // Instead we check that quad points lie within the bounding box of
        // the cell's corner vertices, which is a necessary (though not
        // sufficient) condition for correctness and is geometry-independent.
        // ----------------------------------------------------------------
        {
          // Compute bounding box from the dim+1 corner vertices only
          RealType lo[dim], hi[dim];
          for (unsigned int d = 0; d < dim; ++d) {
            lo[d] = cell.vertex(0)(d);
            hi[d] = cell.vertex(0)(d);
          }
          // Corner nodes are always vertices 0..dim for a simplex
          for (unsigned int v = 1; v <= dim; ++v) {
            for (unsigned int d = 0; d < dim; ++d) {
              lo[d] = std::min(lo[d], cell.vertex(v)(d));
              hi[d] = std::max(hi[d], cell.vertex(v)(d));
            }
          }
          // Relax by a small factor to allow for curved protrusion
          constexpr RealType bbox_tol = 1.0e-6;
          for (unsigned int q = 0; q < n_q; ++q) {
            for (unsigned int d = 0; d < dim; ++d) {
              EXPECT_GE(fe_face_values.q_point(q)(d), lo[d] - bbox_tol);
              EXPECT_LE(fe_face_values.q_point(q)(d), hi[d] + bbox_tol);
            }
          }
        }

        // ----------------------------------------------------------------
        // Q2-specific: JxW sum == face measure.
        // cell.face(f).measure() computes the area of the planar triangle
        // from corner vertices. For a nearly-flat Q2 face this should
        // match the integrated JxW to within a small tolerance.
        // This is a weaker check than the divergence theorem below but
        // catches gross errors in face_jac scaling.
        // ----------------------------------------------------------------
        RealType jxw_sum = 0.0;
        for (unsigned int q = 0; q < n_q; ++q)
          jxw_sum += fe_face_values.JxW(q);
        EXPECT_NEAR(jxw_sum, cell.face(f).measure(), 1.0e-6);
      }

      // ----------------------------------------------------------------
      // Divergence theorem — the most important Q2 correctness check.
      // For any cell, sum over all faces of (n · e_d) * JxW must be 0
      // for each Cartesian direction d. This holds for any geometry,
      // curved or not, and exercises both the per-point normal and JxW.
      // ----------------------------------------------------------------
      for (unsigned int d = 0; d < dim; ++d) {
        RealType flux = 0.0;
        for (unsigned int f = 0;
             f < SimplexTopology<dim, mesh_q>::faces_per_cell; ++f) {
          fe_face_values.reinit(cell, f);
          for (unsigned int q = 0; q < fe_face_values.n_q_points(); ++q)
            flux += fe_face_values.normal(q)(d) * fe_face_values.JxW(q);
        }
        EXPECT_NEAR(flux, 0.0, tol);
      }
    }
  }
}
