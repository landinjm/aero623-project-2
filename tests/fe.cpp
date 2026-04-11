#include <dof_handler.hpp>
#include <fe.hpp>
#include <random>
#include <read_gri.hpp>
#include <triangulation.hpp>

#include <gtest/gtest.h>

using RealType = double;
static constexpr RealType tol = 1.0e-12;

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
