#include <fe.hpp>
#include <gtest/gtest.h>

using RealType = double;
static constexpr RealType tol = 1.0e-12;

TEST(QGaussSimplex, 1D_weight_sum)
{
  for (unsigned int order = 1; order <= 3; ++order) {
    QGaussSimplex<1, RealType> quad(order);
    auto wts = quad.weights_host();
    RealType sum = 0.0;
    for (unsigned int q = 0; q < quad.n_points(); ++q) {
      sum += wts(q);
    }
    EXPECT_NEAR(sum, 1.0, tol);
  }
}

TEST(QGaussSimplex, 2D_weight_sum)
{
  for (unsigned int order = 1; order <= 3; ++order) {
    QGaussSimplex<2, RealType> quad(order);
    auto wts = quad.weights_host();
    RealType sum = 0.0;
    for (unsigned int q = 0; q < quad.n_points(); ++q) {
      sum += wts(q);
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
