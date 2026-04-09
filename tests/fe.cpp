#include <dof_handler.hpp>
#include <fe.hpp>
#include <read_gri.hpp>
#include <triangulation.hpp>

#include <gtest/gtest.h>

using RealType = double;
static constexpr RealType tol = 1.0e-12;
static constexpr RealType fd_tol = 1.0e-5;

static RealType
exact_tet_integral(unsigned int a, unsigned int b, unsigned int c)
{
  auto factorial = [](unsigned int n) -> RealType {
    RealType f = 1.0;
    for (unsigned int i = 2; i <= n; ++i) {
      f *= i;
    }
    return f;
  };
  return factorial(a) * factorial(b) * factorial(b) / factorial(a + b + c + 3);
}

static RealType
exact_triangle_integral(unsigned int a, unsigned int b)
{
  auto factorial = [](unsigned int n) -> RealType {
    RealType f = 1.0;
    for (unsigned int i = 2; i <= n; ++i) {
      f *= i;
    }
    return f;
  };
  return factorial(a) * factorial(b) / factorial(a + b + 2);
}

static RealType
exact_line_integral(unsigned int a)
{
  return 1.0 / (a + 1.0);
}

TEST(FE_DGQLegendre, 1D_DoF_counts)
{
  for (unsigned int order = 0; order <= 3; ++order) {
    FE_DGQLegendre<1, RealType> fe(order);
    EXPECT_EQ(fe.n_dofs(), order + 1);
  }
}

TEST(FE_DGQLegendre, 2D_DoF_counts)
{
  for (unsigned int order = 0; order <= 3; ++order) {
    FE_DGQLegendre<2, RealType> fe(order);
    EXPECT_EQ(fe.n_dofs(), (order + 1) * (order + 2) / 2);
  }
}

TEST(FE_DGQLegendre, 3D_DoF_counts)
{
  for (unsigned int order = 0; order <= 3; ++order) {
    FE_DGQLegendre<3, RealType> fe(order);
    EXPECT_EQ(fe.n_dofs(), (order + 1) * (order + 2) * (order + 3) / 6);
  }
}

TEST(FE_DGQLegendre, 2D_basis_value)
{
  for (unsigned int order = 0; order <= 3; ++order) {
    FE_DGQLegendre<2, RealType> fe(order);

    const auto n_dofs = fe.n_dofs();

    switch (order) {
      case 0: {
        Tensor<1, 2, RealType> n0 = { RealType(0), RealType(0) };
        EXPECT_NEAR(fe.shape_value(0, n0), 1.0, tol);
        break;
      }
      case 1: {
        Tensor<1, 2, RealType> n0 = { RealType(0), RealType(0) };
        Tensor<1, 2, RealType> n1 = { RealType(1), RealType(0) };
        Tensor<1, 2, RealType> n2 = { RealType(0), RealType(1) };
        std::vector<Tensor<1, 2, RealType>> nodes = { n0, n1, n2 };

        for (unsigned int i = 0; i < n_dofs; ++i)
          for (unsigned int j = 0; j < n_dofs; ++j)
            EXPECT_NEAR(fe.shape_value(j, nodes[i]), (i == j) ? 1.0 : 0.0, tol);
        break;
      }
      case 2: {
        Tensor<1, 2, RealType> n0 = { RealType(0), RealType(0) };
        Tensor<1, 2, RealType> n1 = { RealType(1), RealType(0) };
        Tensor<1, 2, RealType> n2 = { RealType(0), RealType(1) };
        Tensor<1, 2, RealType> n3 = { RealType(0.5), RealType(0.5) };
        Tensor<1, 2, RealType> n4 = { RealType(0), RealType(0.5) };
        Tensor<1, 2, RealType> n5 = { RealType(0.5), RealType(0) };
        std::vector<Tensor<1, 2, RealType>> nodes = { n0, n1, n2, n3, n4, n5 };

        for (unsigned int i = 0; i < n_dofs; ++i)
          for (unsigned int j = 0; j < n_dofs; ++j)
            EXPECT_NEAR(fe.shape_value(j, nodes[i]), (i == j) ? 1.0 : 0.0, tol);
        break;
      }
      case 3: {

        Tensor<1, 2, RealType> n0 = { RealType(0), RealType(0) };
        Tensor<1, 2, RealType> n1 = { RealType(1), RealType(0) };
        Tensor<1, 2, RealType> n2 = { RealType(0), RealType(1) };
        Tensor<1, 2, RealType> n3 = { RealType(2. / 3.), RealType(1. / 3.) };
        Tensor<1, 2, RealType> n4 = { RealType(1. / 3.), RealType(2. / 3.) };
        Tensor<1, 2, RealType> n5 = { RealType(0), RealType(2. / 3.) };
        Tensor<1, 2, RealType> n6 = { RealType(0), RealType(1. / 3.) };
        Tensor<1, 2, RealType> n7 = { RealType(1. / 3.), RealType(0) };
        Tensor<1, 2, RealType> n8 = { RealType(2. / 3.), RealType(0) };       
        Tensor<1, 2, RealType> n9 = { RealType(1. / 3.), RealType(1. / 3.) };

        std::vector<Tensor<1, 2, RealType>> nodes = { n0, n1, n2, n3, n4,
                                                      n5, n6, n7, n8, n9 };

        for (unsigned int i = 0; i < n_dofs; ++i)
          for (unsigned int j = 0; j < n_dofs; ++j)
            EXPECT_NEAR(fe.shape_value(j, nodes[i]), (i == j) ? 1.0 : 0.0, tol);
        break;
      }
    }
  }
}

TEST(FE_DGQLegendre, 2D_basis_gradient)
{
  for (unsigned int order = 0; order <= 3; ++order) {
    FE_DGQLegendre<2, RealType> fe(order);

    const auto n_dofs = fe.n_dofs();

    switch (order) {
      case 0:
      case 1:
      case 2:
      case 3: {
        // finite difference check at an interior point
        const RealType h = 1e-6;
        const RealType fd_tol = 1e-5;
        Tensor<1, 2, RealType> pt = { RealType(0.2), RealType(0.3) };
        Tensor<1, 2, RealType> pt_dx = { RealType(0.2) + h, RealType(0.3) };
        Tensor<1, 2, RealType> pt_dy = { RealType(0.2), RealType(0.3) + h };

        for (unsigned int j = 0; j < n_dofs; ++j) {
          auto grad = fe.shape_gradient(j, pt);
          RealType fd_x =
            (fe.shape_value(j, pt_dx) - fe.shape_value(j, pt)) / h;
          RealType fd_y =
            (fe.shape_value(j, pt_dy) - fe.shape_value(j, pt)) / h;
          EXPECT_NEAR(grad(0), fd_x, fd_tol);
          EXPECT_NEAR(grad(1), fd_y, fd_tol);
        }
        break;
        break;
      }
    }
  }
}

TEST(FE_DGQLegendre, 3D_partition_of_unity)
{
  std::vector<std::array<RealType, 3>> pts = {
    {0.1, 0.1, 0.1}, {0.5, 0.1, 0.1}, {0.1, 0.5, 0.1},
    {0.1, 0.1, 0.5}, {0.25, 0.25, 0.25}
  };
  for (unsigned int p = 0; p <= 3; ++p) {
    FE_DGQLegendre<3, RealType> fe(p);
    for (auto& arr : pts) {
      Tensor<1, 3, RealType> pt;
      pt(0)=arr[0]; pt(1)=arr[1]; pt(2)=arr[2];
      RealType sum = 0;
      for (unsigned int i = 0; i < fe.n_dofs(); ++i)
        sum += fe.shape_value(i, pt);
      EXPECT_NEAR(sum, 1.0, tol) << "  p=" << p;
    }
  }
}

TEST(FE_DGQLegendre, 3D_basis_value)
{
  for (unsigned int p = 0; p <= 3; ++p) {
    FE_DGQLegendre<3, RealType> fe(p);
    for (unsigned int i = 0; i < fe.n_dofs(); ++i)
      for (unsigned int j = 0; j < fe.n_dofs(); ++j)
        EXPECT_NEAR(fe.shape_value(i, fe.node(j)),
                    (i == j) ? 1.0 : 0.0, tol)
            << "  p=" << p << " i=" << i << " j=" << j;
  }
}

TEST(FE_DGQLegendre, 3D_basis_gradient)
{
  const RealType h = 1e-6;
  Tensor<1, 3, RealType> pt, pt_dx, pt_dy, pt_dz;
  pt(0)=0.15;    pt(1)=0.20;    pt(2)=0.10;
  pt_dx(0)=pt(0)+h; pt_dx(1)=pt(1); pt_dx(2)=pt(2);
  pt_dy(0)=pt(0);   pt_dy(1)=pt(1)+h; pt_dy(2)=pt(2);
  pt_dz(0)=pt(0);   pt_dz(1)=pt(1);   pt_dz(2)=pt(2)+h;

  for (unsigned int p = 0; p <= 3; ++p) {
    FE_DGQLegendre<3, RealType> fe(p);
    for (unsigned int i = 0; i < fe.n_dofs(); ++i) {
      auto grad = fe.shape_gradient(i, pt);
      EXPECT_NEAR(grad(0), (fe.shape_value(i,pt_dx)-fe.shape_value(i,pt))/h, fd_tol)
          << "  p=" << p << " i=" << i << " d=0";
      EXPECT_NEAR(grad(1), (fe.shape_value(i,pt_dy)-fe.shape_value(i,pt))/h, fd_tol)
          << "  p=" << p << " i=" << i << " d=1";
      EXPECT_NEAR(grad(2), (fe.shape_value(i,pt_dz)-fe.shape_value(i,pt))/h, fd_tol)
          << "  p=" << p << " i=" << i << " d=2";
    }
  }
}

TEST(FE_DGQLegendre, 3D_gradient_sum_is_zero)
{
  Tensor<1, 3, RealType> pt;
  pt(0)=0.15; pt(1)=0.20; pt(2)=0.10;
  for (unsigned int p = 0; p <= 3; ++p) {
    FE_DGQLegendre<3, RealType> fe(p);
    Tensor<1, 3, RealType> grad_sum;
    for (unsigned int i = 0; i < fe.n_dofs(); ++i) {
      auto g = fe.shape_gradient(i, pt);
      for (unsigned int d = 0; d < 3; ++d) grad_sum(d) += g(d);
    }
    for (unsigned int d = 0; d < 3; ++d)
      EXPECT_NEAR(grad_sum(d), 0.0, tol) << "  p=" << p << " d=" << d;
  }
}

TEST(QGaussSimplex, 1D_weight_sum)
{
  for (unsigned int order = 1; order <= 4; ++order) {
    QGaussSimplex<1, RealType> quad(order);
    RealType sum = 0.0;
    for (unsigned int q = 0; q < quad.n_points(); ++q) {
      sum += quad.weight(q);
    }
    EXPECT_NEAR(sum, 1.0, tol);
  }
}

TEST(QGaussSimplex, 2D_weight_sum)
{
  for (unsigned int order = 1; order <= 4; ++order) {
    QGaussSimplex<2, RealType> quad(order);
    RealType sum = 0.0;
    for (unsigned int q = 0; q < quad.n_points(); ++q) {
      sum += quad.weight(q);
    }
    EXPECT_NEAR(sum, 0.5, tol);
  }
}

TEST(QGaussSimplex, 3D_weight_sum)
{
  for (unsigned int order = 1; order <= 4; ++order) {
    QGaussSimplex<3, RealType> quad(order);
    RealType sum = 0.0;
    for (unsigned int q = 0; q < quad.n_points(); ++q) {
      sum += quad.weight(q);
    }
    EXPECT_NEAR(sum, 1./6., tol);
  }
}

TEST(QGaussSimplex, 1D_weight_values)
{
  for (unsigned int order = 1; order <= 4; ++order) {
    QGaussSimplex<1, RealType> quad(order);
    auto pts = quad.points_host();
    auto wts = quad.weights_host();

    switch (order) {
      case 1:
        EXPECT_NEAR(pts(0, 0), 0.500000000000000, tol);
        EXPECT_NEAR(wts(0), 1.000000000000000, tol);
        break;
      case 2:
        EXPECT_NEAR(pts(0, 0), 0.211324865405187, tol);
        EXPECT_NEAR(pts(1, 0), 0.788675134594813, tol);

        EXPECT_NEAR(wts(0), 0.500000000000000, tol);
        EXPECT_NEAR(wts(1), 0.500000000000000, tol);
        break;
      case 3:
        EXPECT_NEAR(pts(0, 0), 0.112701665379258, tol);
        EXPECT_NEAR(pts(1, 0), 0.500000000000000, tol);
        EXPECT_NEAR(pts(2, 0), 0.887298334620742, tol);

        EXPECT_NEAR(wts(0), 0.277777777777778, tol);
        EXPECT_NEAR(wts(1), 0.444444444444444, tol);
        EXPECT_NEAR(wts(2), 0.277777777777778, tol);
        break;
      case 4:
        EXPECT_NEAR(pts(0, 0), 0.069431844202974, tol);
        EXPECT_NEAR(pts(1, 0), 0.330009478207572, tol);
        EXPECT_NEAR(pts(2, 0), 0.669990521792428, tol);
        EXPECT_NEAR(pts(3, 0), 0.930568155797026, tol);

        EXPECT_NEAR(wts(0), 0.173927422568727, tol);
        EXPECT_NEAR(wts(1), 0.326072577431273, tol);
        EXPECT_NEAR(wts(2), 0.326072577431273, tol);
        EXPECT_NEAR(wts(3), 0.173927422568727, tol);
        break;
    }
  }
}

TEST(QGaussSimplex, 2D_weight_values)
{
  for (unsigned int order = 1; order <= 4; ++order) {
    QGaussSimplex<2, RealType> quad(order);
    auto pts = quad.points_host();
    auto wts = quad.weights_host();

    switch (order) {
      case 1:
        EXPECT_NEAR(pts(0, 0), 0.333333333333333, tol);
        EXPECT_NEAR(pts(0, 1), 0.333333333333333, tol);

        EXPECT_NEAR(wts(0), 0.500000000000000, tol);
        break;
      case 2:
        EXPECT_NEAR(pts(0, 0), 0.166666666666667, tol);
        EXPECT_NEAR(pts(0, 1), 0.166666666666667, tol);
        EXPECT_NEAR(pts(1, 0), 0.666666666666667, tol);
        EXPECT_NEAR(pts(1, 1), 0.166666666666667, tol);
        EXPECT_NEAR(pts(2, 0), 0.166666666666667, tol);
        EXPECT_NEAR(pts(2, 1), 0.666666666666667, tol);

        EXPECT_NEAR(wts(0), 0.166666666666666, tol);
        EXPECT_NEAR(wts(1), 0.166666666666666, tol);
        EXPECT_NEAR(wts(2), 0.166666666666666, tol);
        break;
      case 3:
        EXPECT_NEAR(pts(0, 0), 0.333333333333333, tol);
        EXPECT_NEAR(pts(0, 1), 0.333333333333333, tol);
        EXPECT_NEAR(pts(1, 0), 0.200000000000000, tol);
        EXPECT_NEAR(pts(1, 1), 0.200000000000000, tol);
        EXPECT_NEAR(pts(2, 0), 0.600000000000000, tol);
        EXPECT_NEAR(pts(2, 1), 0.200000000000000, tol);
        EXPECT_NEAR(pts(3, 0), 0.200000000000000, tol);
        EXPECT_NEAR(pts(3, 1), 0.600000000000000, tol);

        EXPECT_NEAR(wts(0), -0.281250000000000, tol);
        EXPECT_NEAR(wts(1), 0.260416666666667, tol);
        EXPECT_NEAR(wts(2), 0.260416666666667, tol);
        EXPECT_NEAR(wts(3), 0.260416666666667, tol);
        break;
      case 4:
        EXPECT_NEAR(pts(0, 0), 0.445948490915965, tol);
        EXPECT_NEAR(pts(0, 1), 0.445948490915965, tol);
        EXPECT_NEAR(pts(1, 0), 0.108103018168070, tol);
        EXPECT_NEAR(pts(1, 1), 0.445948490915965, tol);
        EXPECT_NEAR(pts(2, 0), 0.445948490915965, tol);
        EXPECT_NEAR(pts(2, 1), 0.108103018168070, tol);
        EXPECT_NEAR(pts(3, 0), 0.091576213509771, tol);
        EXPECT_NEAR(pts(3, 1), 0.091576213509771, tol);
        EXPECT_NEAR(pts(4, 0), 0.816847572980459, tol);
        EXPECT_NEAR(pts(4, 1), 0.091576213509771, tol);
        EXPECT_NEAR(pts(5, 0), 0.091576213509771, tol);
        EXPECT_NEAR(pts(5, 1), 0.816847572980459, tol);

        EXPECT_NEAR(wts(0), 0.111690794839005, tol);
        EXPECT_NEAR(wts(1), 0.111690794839005, tol);
        EXPECT_NEAR(wts(2), 0.111690794839005, tol);
        EXPECT_NEAR(wts(3), 0.054975871827661, tol);
        EXPECT_NEAR(wts(4), 0.054975871827661, tol);
        EXPECT_NEAR(wts(5), 0.054975871827661, tol);
        break;
    }
  }
}

TEST(QGaussSimplex, 3D_weight_values)
{
  for (unsigned int order = 1; order <= 4; ++order) {
    QGaussSimplex<3, RealType> quad(order);
    auto pts = quad.points_host();
    auto wts = quad.weights_host();

    switch (order) {

      case 1:
        // 1-point centroid rule
        EXPECT_NEAR(pts(0,0), 0.25, tol);
        EXPECT_NEAR(pts(0,1), 0.25, tol);
        EXPECT_NEAR(pts(0,2), 0.25, tol);

        EXPECT_NEAR(wts(0), 1.0/6.0, tol);
        break;

      case 2:
        // 4-point tetrahedral rule
        EXPECT_NEAR(pts(0,0), 0.585410196624969, tol);
        EXPECT_NEAR(pts(0,1), 0.138196601125011, tol);
        EXPECT_NEAR(pts(0,2), 0.138196601125011, tol);

        EXPECT_NEAR(pts(1,0), 0.138196601125011, tol);
        EXPECT_NEAR(pts(1,1), 0.585410196624969, tol);
        EXPECT_NEAR(pts(1,2), 0.138196601125011, tol);

        EXPECT_NEAR(pts(2,0), 0.138196601125011, tol);
        EXPECT_NEAR(pts(2,1), 0.138196601125011, tol);
        EXPECT_NEAR(pts(2,2), 0.585410196624969, tol);

        EXPECT_NEAR(pts(3,0), 0.138196601125011, tol);
        EXPECT_NEAR(pts(3,1), 0.138196601125011, tol);
        EXPECT_NEAR(pts(3,2), 0.138196601125011, tol);

        for (unsigned int i=0;i<4;++i)
          EXPECT_NEAR(wts(i), 1.0/24.0, tol);

        break;

      case 3:
        // 5-point tetrahedral rule
        EXPECT_NEAR(pts(0,0), 0.25, tol);
        EXPECT_NEAR(pts(0,1), 0.25, tol);
        EXPECT_NEAR(pts(0,2), 0.25, tol);

        EXPECT_NEAR(wts(0), -2.0/15.0, tol);

        EXPECT_NEAR(pts(1,0), 0.5, tol);
        EXPECT_NEAR(pts(1,1), 1.0/6.0, tol);
        EXPECT_NEAR(pts(1,2), 1.0/6.0, tol);

        EXPECT_NEAR(pts(2,0), 1.0/6.0, tol);
        EXPECT_NEAR(pts(2,1), 0.5, tol);
        EXPECT_NEAR(pts(2,2), 1.0/6.0, tol);

        EXPECT_NEAR(pts(3,0), 1.0/6.0, tol);
        EXPECT_NEAR(pts(3,1), 1.0/6.0, tol);
        EXPECT_NEAR(pts(3,2), 0.5, tol);

        EXPECT_NEAR(pts(4,0), 1.0/6.0, tol);
        EXPECT_NEAR(pts(4,1), 1.0/6.0, tol);
        EXPECT_NEAR(pts(4,2), 1.0/6.0, tol);

        for (unsigned int i=1;i<5;++i)
          EXPECT_NEAR(wts(i), 3.0/40.0, tol);

        break;

      case 4:
        // 11-point Keast rule (common 4th-order tetra rule)

        EXPECT_NEAR(pts(0,0), 0.25, tol);
        EXPECT_NEAR(pts(0,1), 0.25, tol);
        EXPECT_NEAR(pts(0,2), 0.25, tol);

        EXPECT_NEAR(wts(0), -0.013155555555556, tol);

        EXPECT_NEAR(pts(1,0), 0.785714285714286, tol);
        EXPECT_NEAR(pts(1,1), 0.071428571428571, tol);
        EXPECT_NEAR(pts(1,2), 0.071428571428571, tol);

        EXPECT_NEAR(pts(2,0), 0.071428571428571, tol);
        EXPECT_NEAR(pts(2,1), 0.785714285714286, tol);
        EXPECT_NEAR(pts(2,2), 0.071428571428571, tol);

        EXPECT_NEAR(pts(3,0), 0.071428571428571, tol);
        EXPECT_NEAR(pts(3,1), 0.071428571428571, tol);
        EXPECT_NEAR(pts(3,2), 0.785714285714286, tol);

        EXPECT_NEAR(pts(4,0), 0.071428571428571, tol);
        EXPECT_NEAR(pts(4,1), 0.071428571428571, tol);
        EXPECT_NEAR(pts(4,2), 0.071428571428571, tol);

        for (unsigned int i=1;i<5;++i)
          EXPECT_NEAR(wts(i), 0.007622222222222, tol);

        break;
    }
  }
}

class QGaussSimplexExactness1D : public ::testing::TestWithParam<unsigned int>
{};
class QGaussSimplexExactness2D
  : public ::testing::TestWithParam<
      std::tuple<unsigned int, unsigned int, unsigned int>>
{};
class QGaussSimplexExactness3D
  : public ::testing::TestWithParam<
      std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>>
{};

TEST_P(QGaussSimplexExactness1D, IntegratesMonomialsExactly)
{
  unsigned int order = GetParam();
  QGaussSimplex<1, RealType> quad(order);
  auto pts = quad.points_host();
  auto wts = quad.weights_host();

  for (unsigned int a = 0; a <= order; ++a) {
    RealType numerical = 0.0;
    for (unsigned int q = 0; q < quad.n_points(); ++q) {
      numerical += wts(q) * std::pow(pts(q, 0), a);
    }

    EXPECT_NEAR(numerical, exact_line_integral(a), tol);
  }
}

INSTANTIATE_TEST_SUITE_P(Orders,
                         QGaussSimplexExactness1D,
                         ::testing::Values(1u, 2u, 3u, 4u));

TEST_P(QGaussSimplexExactness2D, IntegratesMonomialsExactly)
{
  auto [order, a, b] = GetParam();
  QGaussSimplex<2, RealType> quad(order);
  auto pts = quad.points_host();
  auto wts = quad.weights_host();

  RealType numerical = 0.0;
  for (unsigned int q = 0; q < quad.n_points(); ++q) {
    numerical += wts(q) * std::pow(pts(q, 0), a) * std::pow(pts(q, 1), b);
  }

  EXPECT_NEAR(numerical, exact_triangle_integral(a, b), tol);
}

static std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>
MakeTriangleMonomialCases()
{
  std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> cases;
  for (unsigned int order = 1; order <= 4; ++order) {
    for (unsigned int total = 0; total <= order; ++total) {
      for (unsigned int b = 0; b <= total; ++b) {
        cases.emplace_back(order, total - b, b);
      }
    }
  }
  return cases;
}

INSTANTIATE_TEST_SUITE_P(MonomialCases,
                         QGaussSimplexExactness2D,
                         ::testing::ValuesIn(MakeTriangleMonomialCases()));

/**TEST_P(QGaussSimplexExactness3D, IntegratesMonomialsExactly)
{
  auto [order, a, b, c] = GetParam();

  QGaussSimplex<3, RealType> quad(order);
  auto pts = quad.points_host();
  auto wts = quad.weights_host();

  RealType numerical = 0.0;

  for (unsigned int q = 0; q < quad.n_points(); ++q) {
    numerical += wts(q)
      * std::pow(pts(q,0), a)
      * std::pow(pts(q,1), b)
      * std::pow(pts(q,2), c);
  }

  EXPECT_NEAR(numerical, exact_tet_integral(a,b,c), tol);
}

static std::vector<
  std::tuple<unsigned int,unsigned int,unsigned int,unsigned int>>
MakeTetMonomialCases()
{
  std::vector<
    std::tuple<unsigned int,unsigned int,unsigned int,unsigned int>> cases;

  for (unsigned int order=1; order<=4; ++order) {
    for (unsigned int total=0; total<=order; ++total) {
      for (unsigned int b=0; b<=total; ++b) {
        for (unsigned int c=0; c<=total-b; ++c) {
          unsigned int a = total - b - c;
          cases.emplace_back(order,a,b,c);
        }
      }
    }
  }

  return cases;
}

INSTANTIATE_TEST_SUITE_P(MonomialCases,
                         QGaussSimplexExactness3D,
                         ::testing::ValuesIn(MakeTetMonomialCases()));

 */

class FEValuesTest : public ::testing::TestWithParam<unsigned int>
{};

TEST_P(FEValuesTest, basic)
{
  unsigned int problem_degree = GetParam();

  GriReader<2> gri;
  Triangulation<2> tria;
  FE_DGQLegendre<2, double> fe(problem_degree);
  QGaussSimplex<2, double> quad(problem_degree + 1);
  QGaussSimplex<1, double> face_quad(problem_degree + 1);

  gri.read_gri("../tests/test_2.gri");
  gri.transfer_to_triangulation(tria);

  DoFHandler<2, double> dof_handler(tria, fe);

  FEValues<2, double> fe_values(fe, quad);
  FEFaceValues<2, double> fe_face_values(fe, face_quad);

  double tol = 1.0e-12;
  for (auto cell : dof_handler.active_cell_range()) {
    // For the cell FEValues check that the sum of the JxW equates to the cell
    // volume and the shape values sum to unity.
    fe_values.reinit(cell);
    double area = 0;
    for (unsigned int q = 0; q < fe_values.n_q_points(); ++q) {
      double sum = 0;
      area += fe_values.JxW(q);
      for (unsigned int i = 0; i < fe_values.n_dofs(); ++i) {
        sum += fe_values.shape_value(i, q);
      }
      EXPECT_NEAR(sum, 1.0, tol);
    }
    EXPECT_NEAR(area, cell.measure(), tol);

    // For the face FEValues check that the face lengths equal the actual and
    // the normals are correct.
    for (unsigned int lf = 0; lf < 3; ++lf) {
      fe_face_values.reinit(cell, lf);
      double length = 0;
      for (unsigned int q = 0; q < fe_face_values.n_q_points(); ++q) {
        length += fe_face_values.JxW(q);

        auto n = fe_face_values.normal(q);
        EXPECT_NEAR(n.norm(), 1.0, tol);
      }
      EXPECT_NEAR(length, cell.face(lf).measure(), tol);
    }
  }
}

TEST_P(FEValuesTest, advection_residual)
{
  unsigned int problem_degree = GetParam();
  GriReader<2> gri;
  Triangulation<2> tria;
  FE_DGQLegendre<2, double> fe(problem_degree);
  QGaussSimplex<2, double> quad(problem_degree + 1);
  QGaussSimplex<1, double> face_quad(problem_degree + 1);
  gri.read_gri("../tests/test_2.gri");
  gri.transfer_to_triangulation(tria);
  DoFHandler<2, double> dof_handler(tria, fe);
  FEValues<2, double> fe_values(fe, quad);
  FEFaceValues<2, double> fe_face_values(fe, face_quad);

  struct TestCase
  {
    double a[2];
    std::function<double(double, double)> u;
    std::string name;
  };

  // Each case chosen so a·∇u = 0, meaning no source term needed
  std::vector<TestCase> cases = {
    { { 1.0, 0.0 }, [](double x, double y) { return y; }, "a=(1,0) u=y" },
    { { 0.0, 1.0 }, [](double x, double y) { return x; }, "a=(0,1) u=x" },
    { { 1.0, 1.0 }, [](double x, double y) { return x - y; }, "a=(1,1) u=x-y" },
    { { 1.0, -1.0 },
      [](double x, double y) { return x + y; },
      "a=(1,-1) u=x+y" },
  };

  std::vector<uint32_t> dof_indices;

  for (auto& tc : cases) {
    std::vector<double> residual(dof_handler.n_dofs(), 0.0);

    for (auto cell : dof_handler.active_cell_range()) {
      fe_values.reinit(cell);
      cell.get_dof_indices(dof_indices);

      // Volume term
      for (unsigned int q = 0; q < fe_values.n_q_points(); ++q) {
        auto p = fe_values.q_point(q);
        double u_h = tc.u(p(0), p(1));
        for (unsigned int i = 0; i < fe_values.n_dofs(); ++i) {
          auto grad_v = fe_values.shape_gradient(i, q);
          double a_dot_grad_v = tc.a[0] * grad_v(0) + tc.a[1] * grad_v(1);
          residual[dof_indices[i]] += a_dot_grad_v * u_h * fe_values.JxW(q);
        }
      }

      // Face terms
      for (unsigned int lf = 0; lf < 3; ++lf) {
        fe_face_values.reinit(cell, lf);
        for (unsigned int q = 0; q < fe_face_values.n_q_points(); ++q) {
          auto p = fe_face_values.q_point(q);
          auto n = fe_face_values.normal(q);
          double a_dot_n = tc.a[0] * n(0) + tc.a[1] * n(1);
          double u_h = tc.u(p(0), p(1));
          for (unsigned int i = 0; i < fe_face_values.n_dofs(); ++i)
            residual[dof_indices[i]] -= fe_face_values.shape_value(i, q) *
                                        a_dot_n * u_h * fe_face_values.JxW(q);
        }
      }
    }

    double max_res = 0.0;
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      max_res = std::max(max_res, std::abs(residual[i]));

    EXPECT_NEAR(max_res, 0.0, tol);
  }
}

INSTANTIATE_TEST_SUITE_P(Orders,
                         FEValuesTest,
                         ::testing::Values(0u, 1u, 2u, 3u));

/**
TEST_P(FEValuesTest3D, basic)
{
  unsigned int problem_degree = GetParam();

  GriReader<3> gri;
  Triangulation<3> tria;
  FE_DGQLegendre<3,double> fe(problem_degree);

  QGaussSimplex<3,double> quad(problem_degree+1);
  QGaussSimplex<2,double> face_quad(problem_degree+1);

  gri.read_gri("../tests/test_3.gri");
  gri.transfer_to_triangulation(tria);

  DoFHandler<3,double> dof_handler(tria,fe);

  FEValues<3,double> fe_values(fe,quad);
  FEFaceValues<3,double> fe_face_values(fe,face_quad);

  double tol = 1e-12;

  for (auto cell : dof_handler.active_cell_range()) {

    fe_values.reinit(cell);

    double volume = 0;
    for (unsigned int q=0;q<fe_values.n_q_points();++q) {
      double sum = 0;

      volume += fe_values.JxW(q);

      for (unsigned int i=0;i<fe_values.n_dofs();++i)
        sum += fe_values.shape_value(i,q);

      EXPECT_NEAR(sum,1.0,tol);
    }

    EXPECT_NEAR(volume, cell.measure(), tol);

    // faces
    for (unsigned int lf=0; lf<4; ++lf) {

      fe_face_values.reinit(cell,lf);

      double area = 0;

      for (unsigned int q=0;q<fe_face_values.n_q_points();++q) {

        area += fe_face_values.JxW(q);

        auto n = fe_face_values.normal(q);
        EXPECT_NEAR(n.norm(),1.0,tol);
      }

      EXPECT_NEAR(area, cell.face(lf).measure(), tol);
    }
  }
}

TEST_P(FEValuesTest3D, advection_residual)
{
  unsigned int problem_degree = GetParam();

  GriReader<3> gri;
  Triangulation<3> tria;

  FE_DGQLegendre<3,double> fe(problem_degree);
  QGaussSimplex<3,double> quad(problem_degree+1);
  QGaussSimplex<2,double> face_quad(problem_degree+1);

  gri.read_gri("../tests/test_3.gri");
  gri.transfer_to_triangulation(tria);

  DoFHandler<3,double> dof_handler(tria,fe);

  FEValues<3,double> fe_values(fe,quad);
  FEFaceValues<3,double> fe_face_values(fe,face_quad);

  struct TestCase {
    double a[3];
    std::function<double(double,double,double)> u;
  };

  std::vector<TestCase> cases = {

    {{1,0,0}, [](double x,double y,double z){ return y; }},
    {{0,1,0}, [](double x,double y,double z){ return z; }},
    {{0,0,1}, [](double x,double y,double z){ return x; }},
    {{1,1,1}, [](double x,double y,double z){ return x-y; }},
    {{1,1,0}, [](double x,double y,double z){ return x-y; }}
  };

  std::vector<uint32_t> dof_indices;

  for (auto& tc : cases) {

    std::vector<double> residual(dof_handler.n_dofs(),0.0);

    for (auto cell : dof_handler.active_cell_range()) {

      fe_values.reinit(cell);
      cell.get_dof_indices(dof_indices);

      // volume
      for (unsigned int q=0;q<fe_values.n_q_points();++q) {

        auto p = fe_values.q_point(q);
        double u_h = tc.u(p(0),p(1),p(2));

        for (unsigned int i=0;i<fe_values.n_dofs();++i) {

          auto grad_v = fe_values.shape_gradient(i,q);

          double a_dot =
            tc.a[0]*grad_v(0) +
            tc.a[1]*grad_v(1) +
            tc.a[2]*grad_v(2);

          residual[dof_indices[i]] +=
            a_dot * u_h * fe_values.JxW(q);
        }
      }

      // faces
      for (unsigned int lf=0; lf<4; ++lf) {

        fe_face_values.reinit(cell,lf);

        for (unsigned int q=0;q<fe_face_values.n_q_points();++q) {

          auto p = fe_face_values.q_point(q);
          auto n = fe_face_values.normal(q);

          double a_dot_n =
            tc.a[0]*n(0) +
            tc.a[1]*n(1) +
            tc.a[2]*n(2);

          double u_h = tc.u(p(0),p(1),p(2));

          for (unsigned int i=0;i<fe_face_values.n_dofs();++i) {

            residual[dof_indices[i]] -=
              fe_face_values.shape_value(i,q)
              * a_dot_n * u_h * fe_face_values.JxW(q);
          }
        }
      }
    }

    double max_res = 0;
    for (auto r : residual)
      max_res = std::max(max_res,std::abs(r));

    EXPECT_NEAR(max_res,0.0,tol);
  }
}

INSTANTIATE_TEST_SUITE_P(
  Orders,
  FEValuesTest3D,
  ::testing::Values(0u,1u,2u,3u));
  */