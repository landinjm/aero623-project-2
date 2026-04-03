#include <dof_handler.hpp>
#include <fe.hpp>
#include <read_gri.hpp>
#include <triangulation.hpp>

#include <gtest/gtest.h>

using RealType = double;
static constexpr RealType tol = 1.0e-12;

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

// TEST(FE_DGQLegendre, 1D_DoF_counts)
// {
//   for (unsigned int order = 0; order <= 3; ++order) {
//     FE_DGQLegendre<1, RealType> fe(order);
//     EXPECT_EQ(fe.n_dofs(), order + 1);
//   }
// }

TEST(FE_DGQLegendre, 2D_DoF_counts)
{
  for (unsigned int order = 0; order <= 3; ++order) {
    FE_DGQLegendre<2, RealType> fe(order);
    EXPECT_EQ(fe.n_dofs(), (order + 1) * (order + 2) / 2);
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
        Tensor<1, 2, RealType> n1 = { RealType(0.5), RealType(0) };
        Tensor<1, 2, RealType> n2 = { RealType(1), RealType(0) };
        Tensor<1, 2, RealType> n3 = { RealType(0), RealType(0.5) };
        Tensor<1, 2, RealType> n4 = { RealType(0.5), RealType(0.5) };
        Tensor<1, 2, RealType> n5 = { RealType(0), RealType(1) };
        std::vector<Tensor<1, 2, RealType>> nodes = { n0, n1, n2, n3, n4, n5 };

        for (unsigned int i = 0; i < n_dofs; ++i)
          for (unsigned int j = 0; j < n_dofs; ++j)
            EXPECT_NEAR(fe.shape_value(j, nodes[i]), (i == j) ? 1.0 : 0.0, tol);
        break;
      }
      case 3: {
        Tensor<1, 2, RealType> n0 = { RealType(0), RealType(0) };
        Tensor<1, 2, RealType> n1 = { RealType(1. / 3.), RealType(0) };
        Tensor<1, 2, RealType> n2 = { RealType(2. / 3.), RealType(0) };
        Tensor<1, 2, RealType> n3 = { RealType(1), RealType(0) };
        Tensor<1, 2, RealType> n4 = { RealType(0), RealType(1. / 3.) };
        Tensor<1, 2, RealType> n5 = { RealType(1. / 3.), RealType(1. / 3.) };
        Tensor<1, 2, RealType> n6 = { RealType(2. / 3.), RealType(1. / 3.) };
        Tensor<1, 2, RealType> n7 = { RealType(0), RealType(2. / 3.) };
        Tensor<1, 2, RealType> n8 = { RealType(1. / 3.), RealType(2. / 3.) };
        Tensor<1, 2, RealType> n9 = { RealType(0), RealType(1) };
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

class QGaussSimplexExactness1D : public ::testing::TestWithParam<unsigned int>
{};
class QGaussSimplexExactness2D
  : public ::testing::TestWithParam<
      std::tuple<unsigned int, unsigned int, unsigned int>>
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