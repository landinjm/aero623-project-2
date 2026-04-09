#include <tensor.hpp>
#include <gtest/gtest.h>
#include <cmath>

// ── Fixture ───────────────────────────────────────────────────────────────────

class TensorTest3D : public ::testing::Test
{
protected:
  static constexpr unsigned int dim = 3;

  using Vec = Tensor<1, dim, double>;
  using Mat = Tensor<2, dim, double>;

  Vec a, b, zero_vec;
  Mat A, B, I, zero_mat;

  void SetUp() override
  {
    a = Vec{1.0, 2.0, 3.0};
    b = Vec{4.0, 5.0, 6.0};

    A = Mat{1.0, 2.0, 3.0,
            0.0, 4.0, 5.0,
            1.0, 0.0, 6.0};

    B = Mat{7.0, 8.0, 9.0,
            2.0, 3.0, 1.0,
            5.0, 6.0, 4.0};

    I = Mat{1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0};

    zero_vec = Vec{0.0, 0.0, 0.0};
    zero_mat = Mat{0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0};
  }

  void expect_vec_near(const Vec& v, const Vec& expected, double tol = 1e-10)
  {
    for (unsigned int i = 0; i < dim; ++i)
      EXPECT_NEAR(v(i), expected(i), tol) << "  component " << i;
  }

  void expect_mat_near(const Mat& M, const Mat& expected, double tol = 1e-10)
  {
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        EXPECT_NEAR(M(i, j), expected(i, j), tol)
            << "  component (" << i << "," << j << ")";
  }
};

// ── Construction ──────────────────────────────────────────────────────────────

TEST_F(TensorTest3D, DefaultConstructorZeroInitialises)
{
  Vec v;
  expect_vec_near(v, zero_vec);

  Mat M;
  expect_mat_near(M, zero_mat);
}

TEST_F(TensorTest3D, ScalarConstructorFillsAllComponents)
{
  Vec v(3.14);
  for (unsigned int i = 0; i < dim; ++i)
    EXPECT_DOUBLE_EQ(v(i), 3.14);

  Mat M(2.0);
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      EXPECT_DOUBLE_EQ(M(i, j), 2.0);
}

TEST_F(TensorTest3D, InitializerListConstructor)
{
  Vec v{7.0, 8.0, 9.0};
  EXPECT_DOUBLE_EQ(v(0), 7.0);
  EXPECT_DOUBLE_EQ(v(1), 8.0);
  EXPECT_DOUBLE_EQ(v(2), 9.0);
}

TEST_F(TensorTest3D, ClearZeroesAllComponents)
{
  a.clear();
  expect_vec_near(a, zero_vec);

  A.clear();
  expect_mat_near(A, zero_mat);
}

// ── Accessors ─────────────────────────────────────────────────────────────────

TEST_F(TensorTest3D, VecBracketAndParenAreEquivalent)
{
  for (unsigned int i = 0; i < dim; ++i)
    EXPECT_DOUBLE_EQ(a[i], a(i));
}

TEST_F(TensorTest3D, MatParenIndexing)
{
  EXPECT_DOUBLE_EQ(A(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(A(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(A(1, 1), 4.0);
  EXPECT_DOUBLE_EQ(A(2, 2), 6.0);
}

TEST_F(TensorTest3D, DataPointerRoundtrip)
{
  double* ptr = a.data();
  for (unsigned int i = 0; i < dim; ++i)
    EXPECT_DOUBLE_EQ(ptr[i], a(i));

  const Vec& ca = a;
  const double* cptr = ca.data();
  for (unsigned int i = 0; i < dim; ++i)
    EXPECT_DOUBLE_EQ(cptr[i], a(i));
}

// ── Arithmetic: vectors ───────────────────────────────────────────────────────

TEST_F(TensorTest3D, VecAddition)
{
  Vec result = a + b;
  expect_vec_near(result, Vec{5.0, 7.0, 9.0});
}

TEST_F(TensorTest3D, VecSubtraction)
{
  Vec result = a - b;
  expect_vec_near(result, Vec{-3.0, -3.0, -3.0});
}

TEST_F(TensorTest3D, VecScalarMultiplication)
{
  expect_vec_near(a * 2.0, Vec{2.0, 4.0, 6.0});
}

TEST_F(TensorTest3D, VecScalarDivision)
{
  // Avoid operator/= hitting ASSERT_DEBUG — divide by non-zero
  Vec v = a;
  Vec result = v / 2.0;
  expect_vec_near(result, Vec{0.5, 1.0, 1.5});
}

TEST_F(TensorTest3D, VecUnaryNegation)
{
  expect_vec_near(-a, Vec{-1.0, -2.0, -3.0});
}

TEST_F(TensorTest3D, VecCompoundAssignment)
{
  Vec v = a;
  v += b;
  expect_vec_near(v, Vec{5.0, 7.0, 9.0});

  v = a;
  v -= b;
  expect_vec_near(v, Vec{-3.0, -3.0, -3.0});

  v = a;
  v *= 3.0;
  expect_vec_near(v, Vec{3.0, 6.0, 9.0});

  v = a;
  v /= 2.0;
  expect_vec_near(v, Vec{0.5, 1.0, 1.5});
}

// ── Arithmetic: matrices ──────────────────────────────────────────────────────

TEST_F(TensorTest3D, MatAdditionAndSubtraction)
{
  Mat sum  = A + B;
  Mat diff = A - B;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j) {
      EXPECT_DOUBLE_EQ(sum(i, j),  A(i, j) + B(i, j));
      EXPECT_DOUBLE_EQ(diff(i, j), A(i, j) - B(i, j));
    }
}

TEST_F(TensorTest3D, MatScalarMultiplication)
{
  Mat result = A * 2.0;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      EXPECT_DOUBLE_EQ(result(i, j), A(i, j) * 2.0);
}

// ── Norms ─────────────────────────────────────────────────────────────────────

TEST_F(TensorTest3D, VecNormSquare)
{
  // {1,2,3} → 1+4+9 = 14
  EXPECT_DOUBLE_EQ(a.norm_square(), 14.0);
}

TEST_F(TensorTest3D, VecNorm)
{
  EXPECT_NEAR(a.norm(), std::sqrt(14.0), 1e-10);
}

TEST_F(TensorTest3D, ZeroVecNorm)
{
  EXPECT_DOUBLE_EQ(zero_vec.norm(), 0.0);
}

// ── Free functions ────────────────────────────────────────────────────────────

TEST_F(TensorTest3D, Dot)
{
  // Explicit template args sidestep int/unsigned deduction mismatch
  EXPECT_DOUBLE_EQ((dot<dim, double>(a, b)), 32.0);
}

TEST_F(TensorTest3D, DotSelfEqualsNormSquare)
{
  EXPECT_DOUBLE_EQ((dot<dim, double>(a, a)), a.norm_square());
}

TEST_F(TensorTest3D, DotIsCommutative)
{
  EXPECT_DOUBLE_EQ((dot<dim, double>(a, b)), (dot<dim, double>(b, a)));
}

TEST_F(TensorTest3D, MatVecMultiply)
{
  // Explicit template instantiation avoids int/unsigned deduction failure
  Vec Ia = operator*<dim, double>(I, a);
  expect_vec_near(Ia, a);

  Vec e0{1.0, 0.0, 0.0};
  Vec col0 = operator*<dim, double>(A, e0);
  EXPECT_DOUBLE_EQ(col0(0), A(0, 0));
  EXPECT_DOUBLE_EQ(col0(1), A(1, 0));
  EXPECT_DOUBLE_EQ(col0(2), A(2, 0));
}

TEST_F(TensorTest3D, Outer)
{
  Mat O = outer<dim, double>(a, b);
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      EXPECT_DOUBLE_EQ(O(i, j), a(i) * b(j));
}

TEST_F(TensorTest3D, TransposeInvolution)
{
  expect_mat_near(transpose<dim, double>(transpose<dim, double>(A)), A);
}

TEST_F(TensorTest3D, TransposeIdentityIsIdentity)
{
  expect_mat_near(transpose<dim, double>(I), I);
}

TEST_F(TensorTest3D, TransposeSwapsOffDiagonal)
{
  Mat At = transpose<dim, double>(A);
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      EXPECT_DOUBLE_EQ(At(i, j), A(j, i));
}

TEST_F(TensorTest3D, Trace)
{
  // A diagonal: 1+4+6 = 11
  EXPECT_DOUBLE_EQ((trace<dim, double>(A)), 11.0);
  EXPECT_DOUBLE_EQ((trace<dim, double>(I)), 3.0);
}

TEST_F(TensorTest3D, DoubleContract)
{
  EXPECT_DOUBLE_EQ((double_contract<dim, double>(I, I)), 3.0);

  double expected = 0.0;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      expected += A(i, j) * A(i, j);
  EXPECT_DOUBLE_EQ((double_contract<dim, double>(A, A)), expected);
}

// ── dim=3 specialisations ─────────────────────────────────────────────────────

TEST_F(TensorTest3D, CrossProductAnticommutative)
{
  Vec axb = cross(a, b);
  Vec bxa = cross(b, a);
  expect_vec_near(axb, -bxa);
}

TEST_F(TensorTest3D, CrossProductOrthogonalToBothInputs)
{
  Vec c = cross(a, b);
  EXPECT_NEAR((dot<dim, double>(c, a)), 0.0, 1e-10);
  EXPECT_NEAR((dot<dim, double>(c, b)), 0.0, 1e-10);
}

TEST_F(TensorTest3D, CrossProductKnownValues)
{
  Vec e0{1.0, 0.0, 0.0};
  Vec e1{0.0, 1.0, 0.0};
  Vec e2{0.0, 0.0, 1.0};
  expect_vec_near(cross(e0, e1),  e2);
  expect_vec_near(cross(e1, e2),  e0);
  expect_vec_near(cross(e2, e0),  e1);
}

TEST_F(TensorTest3D, CrossProductSelfIsZero)
{
  expect_vec_near(cross(a, a), zero_vec);
}

TEST_F(TensorTest3D, CrossMagnitudeEqualsAreaOfParallelogram)
{
  Vec c = cross(a, b);
  double ab = dot<dim, double>(a, b);
  double expected = a.norm_square() * b.norm_square() - ab * ab;
  EXPECT_NEAR(c.norm_square(), expected, 1e-10);
}

TEST_F(TensorTest3D, DetIdentityIsOne)
{
  EXPECT_DOUBLE_EQ(det(I), 1.0);
}

TEST_F(TensorTest3D, DetKnownValue)
{
  // {{1,2,3},{0,4,5},{1,0,6}} → 1*(24-0) - 2*(0-5) + 3*(0-4) = 22
  EXPECT_DOUBLE_EQ(det(A), 22.0);
}

TEST_F(TensorTest3D, DetScalesWithRow)
{
  Mat A2 = A;
  for (unsigned int j = 0; j < dim; ++j)
    A2(0, j) *= 2.0;
  EXPECT_NEAR(det(A2), 2.0 * det(A), 1e-10);
}

TEST_F(TensorTest3D, DetTransposeEquals)
{
  EXPECT_NEAR(det(transpose<dim, double>(A)), det(A), 1e-10);
}

TEST_F(TensorTest3D, DetSingularMatrixIsZero)
{
  Mat S{1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0};
  EXPECT_NEAR(det(S), 0.0, 1e-10);
}

TEST_F(TensorTest3D, InverseOfIdentityIsIdentity)
{
  expect_mat_near(inverse(I), I);
}

TEST_F(TensorTest3D, InverseTimesOriginalIsIdentity)
{
  Mat Ainv = inverse(A);
  Mat prod;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      for (unsigned int k = 0; k < dim; ++k)
        prod(i, j) += A(i, k) * Ainv(k, j);
  expect_mat_near(prod, I, 1e-10);
}

TEST_F(TensorTest3D, OriginalTimesInverseIsIdentity)
{
  Mat Ainv = inverse(A);
  Mat prod;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      for (unsigned int k = 0; k < dim; ++k)
        prod(i, j) += Ainv(i, k) * A(k, j);
  expect_mat_near(prod, I, 1e-10);
}

TEST_F(TensorTest3D, InverseInvolution)
{
  expect_mat_near(inverse(inverse(A)), A, 1e-10);
}

TEST_F(TensorTest3D, DetOfInverseIsReciprocal)
{
  EXPECT_NEAR(det(inverse(A)), 1.0 / det(A), 1e-10);
}

// ── Tetrahedral mesh use case ─────────────────────────────────────────────────

TEST_F(TensorTest3D, TetJacobianVolumeAndMapping)
{
  // Unit tet: v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), v3=(0,0,1)
  Vec e1{1.0, 0.0, 0.0};
  Vec e2{0.0, 1.0, 0.0};
  Vec e3{0.0, 0.0, 1.0};

  Mat J;
  for (unsigned int i = 0; i < dim; ++i) {
    J(i, 0) = e1(i);
    J(i, 1) = e2(i);
    J(i, 2) = e3(i);
  }

  // Volume = |det(J)| / 6
  EXPECT_NEAR(std::abs(det(J)) / 6.0, 1.0 / 6.0, 1e-10);

  // Forward mapping: J == I here so x_phys == xi
  Vec xi{0.25, 0.25, 0.25};
  Vec x_phys = operator*<dim, double>(J, xi);
  expect_vec_near(x_phys, xi);

  // Inverse mapping recovers reference coords
  Vec xi_recovered = operator*<dim, double>(inverse(J), x_phys);
  expect_vec_near(xi_recovered, xi);
}

TEST_F(TensorTest3D, TetFaceNormalViaCross)
{
  Vec e1{1.0, 0.0, 0.0};
  Vec e2{0.0, 1.0, 0.0};
  Vec normal = cross(e1, e2);
  normal = normal / normal.norm();
  expect_vec_near(normal, Vec{0.0, 0.0, 1.0});
}