// #include <quad.hpp>

// #include <gtest/gtest.h>

// using RealType = double;
// static constexpr RealType tol = 1.0e-12;

// static constexpr unsigned int max_order = 10;

// /**
//  * Quadrature rule should sum to 1 for the reference line.
//  */
// TEST(QGaussSimplex, 1D_weight_sum)
// {
//   for (unsigned int order = 1; order <= max_order; ++order) {
//     QGaussSimplex<1, RealType> quad(order);
//     RealType sum = 0.0;
//     for (unsigned int q = 0; q < quad.n_points(); ++q) {
//       sum += quad.weight(q);
//     }
//     EXPECT_NEAR(sum, 1.0, tol);
//   }
// }

// /**
//  * Quadrature rule should sum to 1/2 for the reference triangle.
//  */
// TEST(QGaussSimplex, 2D_weight_sum)
// {
//   for (unsigned int order = 1; order <= max_order; ++order) {
//     QGaussSimplex<2, RealType> quad(order);
//     RealType sum = 0.0;
//     for (unsigned int q = 0; q < quad.n_points(); ++q) {
//       sum += quad.weight(q);
//     }
//     EXPECT_NEAR(sum, 0.5, tol);
//   }
// }

// /**
//  * Quadrature rule should sum to 1/6 for the reference tetrahedron.
//  */
// TEST(QGaussSimplex, 3D_weight_sum)
// {
//   for (unsigned int order = 1; order <= max_order; ++order) {
//     QGaussSimplex<3, RealType> quad(order);
//     RealType sum = 0.0;
//     for (unsigned int q = 0; q < quad.n_points(); ++q) {
//       sum += quad.weight(q);
//     }
//     EXPECT_NEAR(sum, 1.0 / 6.0, tol);
//   }
// }

// static RealType
// exact_line_integral(unsigned int a)
// {
//   return RealType(1) / RealType(a + 1);
// }

// class QGaussSimplexExactness1D : public ::testing::TestWithParam<unsigned int>
// {};

// TEST_P(QGaussSimplexExactness1D, IntegratesMonomialsExactly)
// {
//   unsigned int order = GetParam();
//   QGaussSimplex<1, RealType> quad(order);
//   for (unsigned int a = 0; a <= order; ++a) {
//     RealType numerical = 0.0;
//     for (unsigned int q = 0; q < quad.n_points(); ++q) {
//       numerical += quad.weight(q) * std::pow(quad.point(q)(0), a);
//     }
//     EXPECT_NEAR(numerical, exact_line_integral(a), tol);
//   }
// }

// INSTANTIATE_TEST_SUITE_P(Orders,
//                          QGaussSimplexExactness1D,
//                          ::testing::Values(1u, 2u, 3u, 4u));

// static RealType
// exact_triangle_integral(unsigned int a, unsigned int b)
// {
//   auto fact = [](unsigned int n) -> RealType {
//     RealType r = 1;
//     for (unsigned int i = 2; i <= n; ++i)
//       r *= i;
//     return r;
//   };
//   return fact(a) * fact(b) / fact(a + b + 2);
// }

// class QGaussSimplexExactness2D
//   : public ::testing::TestWithParam<
//       std::tuple<unsigned int, unsigned int, unsigned int>>
// {};

// TEST_P(QGaussSimplexExactness2D, IntegratesMonomialsExactly)
// {
//   auto [order, a, b] = GetParam();
//   QGaussSimplex<2, RealType> quad(order);
//   RealType numerical = 0.0;
//   for (unsigned int q = 0; q < quad.n_points(); ++q) {
//     numerical += quad.weight(q) * std::pow(quad.point(q)(0), a) *
//                  std::pow(quad.point(q)(1), b);
//   }
//   EXPECT_NEAR(numerical, exact_triangle_integral(a, b), tol);
// }

// static std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>
// MakeTriangleMonomialCases()
// {
//   std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> cases;
//   for (unsigned int order = 1; order <= 4; ++order) {
//     for (unsigned int total = 0; total <= order; ++total) {
//       for (unsigned int b = 0; b <= total; ++b) {
//         cases.emplace_back(order, total - b, b);
//       }
//     }
//   }
//   return cases;
// }

// INSTANTIATE_TEST_SUITE_P(MonomialCases,
//                          QGaussSimplexExactness2D,
//                          ::testing::ValuesIn(MakeTriangleMonomialCases()));

// static RealType
// exact_tet_integral(unsigned int a, unsigned int b, unsigned int c)
// {
//   auto fact = [](unsigned int n) -> RealType {
//     RealType r = 1;
//     for (unsigned int i = 2; i <= n; ++i)
//       r *= i;
//     return r;
//   };
//   return fact(a) * fact(b) * fact(c) / fact(a + b + c + 3);
// }

// class QGaussSimplexExactness3D
//   : public ::testing::TestWithParam<
//       std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>>
// {};

// TEST_P(QGaussSimplexExactness3D, IntegratesMonomialsExactly)
// {
//   auto [order, a, b, c] = GetParam();
//   QGaussSimplex<3, RealType> quad(order);
//   RealType numerical = 0.0;
//   for (unsigned int q = 0; q < quad.n_points(); ++q) {
//     numerical += quad.weight(q) * std::pow(quad.point(q)(0), a) *
//                  std::pow(quad.point(q)(1), b) * std::pow(quad.point(q)(2), c);
//   }
//   EXPECT_NEAR(numerical, exact_tet_integral(a, b, c), tol);
// }

// static std::vector<
//   std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>>
// MakeTetMonomialCases()
// {
//   std::vector<
//     std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>>
//     cases;
//   for (unsigned int order = 1; order <= 4; ++order) {
//     for (unsigned int total = 0; total <= order; ++total) {
//       for (unsigned int b = 0; b <= total; ++b) {
//         for (unsigned int c = 0; c <= total - b; ++c) {
//           cases.emplace_back(order, total - b - c, b, c);
//         }
//       }
//     }
//   }
//   return cases;
// }

// INSTANTIATE_TEST_SUITE_P(MonomialCases,
//                          QGaussSimplexExactness3D,
//                          ::testing::ValuesIn(MakeTetMonomialCases()));
