#include <fe.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <tuple>

using RealType = double;
static constexpr RealType tol    = 1.0e-10;
static constexpr RealType fd_tol = 1.0e-5;

// ── helpers ───────────────────────────────────────────────────────────────────

static RealType exact_tet_integral(unsigned int a, unsigned int b, unsigned int c)
{
  auto factorial = [](unsigned int n) -> RealType {
    RealType f = 1.0;
    for (unsigned int i = 2; i <= n; ++i) f *= i;
    return f;
  };
  return factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3);
}

struct UnitTet {
  Tensor<1, 3, RealType> vertex(unsigned int i) const {
    Tensor<1, 3, RealType> v;
    if      (i == 0) { v(0)=0; v(1)=0; v(2)=0; }
    else if (i == 1) { v(0)=1; v(1)=0; v(2)=0; }
    else if (i == 2) { v(0)=0; v(1)=1; v(2)=0; }
    else             { v(0)=0; v(1)=0; v(2)=1; }
    return v;
  }
  unsigned int index() const { return 0; }
};

struct ScaledTet {
  Tensor<1, 3, RealType> vertex(unsigned int i) const {
    Tensor<1, 3, RealType> v;
    if      (i == 0) { v(0)=1; v(1)=1; v(2)=1; }
    else if (i == 1) { v(0)=3; v(1)=1; v(2)=1; }
    else if (i == 2) { v(0)=1; v(1)=3; v(2)=1; }
    else             { v(0)=1; v(1)=1; v(2)=3; }
    return v;
  }
  unsigned int index() const { return 0; }
};

// ── FE_DGQLegendre 3D ─────────────────────────────────────────────────────────

TEST(FE_DGQLegendre3D, DoFCounts)
{
  for (unsigned int p = 0; p <= 3; ++p) {
    FE_DGQLegendre<3, RealType> fe(p);
    EXPECT_EQ(fe.n_dofs(), (p+1)*(p+2)*(p+3)/6) << "  p=" << p;
  }
}

TEST(FE_DGQLegendre3D, PartitionOfUnity)
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

TEST(FE_DGQLegendre3D, KroneckerDelta)
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

TEST(FE_DGQLegendre3D, GradientFiniteDifference)
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

TEST(FE_DGQLegendre3D, GradientSumIsZero)
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

// ── QGaussSimplex 3D ──────────────────────────────────────────────────────────

TEST(QGaussSimplex3D, WeightSumEqualsVolume)
{
  for (unsigned int order = 1; order <= 4; ++order) {
    QGaussSimplex<3, RealType> quad(order);
    RealType sum = 0;
    for (unsigned int q = 0; q < quad.n_points(); ++q)
      sum += quad.weight(q);
    EXPECT_NEAR(sum, 1.0/6.0, tol) << "  order=" << order;
  }
}

TEST(QGaussSimplex3D, PointsInsideReferenceTet)
{
  for (unsigned int order = 1; order <= 4; ++order) {
    QGaussSimplex<3, RealType> quad(order);
    for (unsigned int q = 0; q < quad.n_points(); ++q) {
      auto p = quad.point(q);
      EXPECT_GE(p(0), -tol) << "  x<0 order=" << order;
      EXPECT_GE(p(1), -tol) << "  y<0 order=" << order;
      EXPECT_GE(p(2), -tol) << "  z<0 order=" << order;
      EXPECT_LE(p(0)+p(1)+p(2), 1.0+tol) << "  x+y+z>1 order=" << order;
    }
  }
}

// ── Parameterised exactness test ──────────────────────────────────────────────
// The >> parse error came from nesting std::tuple inside TestWithParam without
// a space. The fix is a typedef for the param type.

typedef std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>
  TetMonomialParam;

class QGaussSimplexExactness3D
  : public ::testing::TestWithParam<TetMonomialParam>
{};

TEST_P(QGaussSimplexExactness3D, IntegratesMonomialsExactly)
{
  unsigned int order, a, b, c;
  std::tie(order, a, b, c) = GetParam();

  QGaussSimplex<3, RealType> quad(order);
  RealType numerical = 0;
  for (unsigned int q = 0; q < quad.n_points(); ++q) {
    auto p = quad.point(q);
    numerical += quad.weight(q)
               * std::pow(p(0), a)
               * std::pow(p(1), b)
               * std::pow(p(2), c);
  }
  EXPECT_NEAR(numerical, exact_tet_integral(a, b, c), tol)
      << "  order=" << order << " a=" << a << " b=" << b << " c=" << c;
}

static std::vector<TetMonomialParam> MakeTetMonomialCases()
{
  std::vector<TetMonomialParam> cases;
  for (unsigned int order = 1; order <= 4; ++order)
    for (unsigned int total = 0; total <= order; ++total)
      for (unsigned int aa = 0; aa <= total; ++aa)
        for (unsigned int bb = 0; bb <= total - aa; ++bb)
          cases.emplace_back(order, aa, bb, total - aa - bb);
  return cases;
}

INSTANTIATE_TEST_SUITE_P(MonomialCases,
                         QGaussSimplexExactness3D,
                         ::testing::ValuesIn(MakeTetMonomialCases()));

// ── FEValues 3D ───────────────────────────────────────────────────────────────

class FEValues3DTest : public ::testing::TestWithParam<unsigned int> {};

TEST_P(FEValues3DTest, JxWSumsToVolume)
{
  unsigned int p = GetParam();
  FE_DGQLegendre<3, RealType> fe(p);
  QGaussSimplex<3, RealType>  quad(p + 1);
  FEValues<3, RealType> fev(fe, quad);

  UnitTet cell;
  fev.reinit(cell);

  RealType vol = 0;
  for (unsigned int q = 0; q < fev.n_q_points(); ++q)
    vol += fev.JxW(q);
  EXPECT_NEAR(vol, 1.0/6.0, tol) << "  p=" << p;
}

TEST_P(FEValues3DTest, ShapeValuesSumToOne)
{
  unsigned int p = GetParam();
  FE_DGQLegendre<3, RealType> fe(p);
  QGaussSimplex<3, RealType>  quad(p + 1);
  FEValues<3, RealType> fev(fe, quad);

  UnitTet cell;
  fev.reinit(cell);

  for (unsigned int q = 0; q < fev.n_q_points(); ++q) {
    RealType sum = 0;
    for (unsigned int i = 0; i < fev.n_dofs(); ++i)
      sum += fev.shape_value(i, q);
    EXPECT_NEAR(sum, 1.0, tol) << "  p=" << p << " q=" << q;
  }
}

TEST_P(FEValues3DTest, ScaledTetVolume)
{
  unsigned int p = GetParam();
  FE_DGQLegendre<3, RealType> fe(p);
  QGaussSimplex<3, RealType>  quad(p + 1);
  FEValues<3, RealType> fev(fe, quad);

  // Edges are all length 2 -> det(J)=8 -> volume = 8/6
  ScaledTet cell;
  fev.reinit(cell);

  RealType vol = 0;
  for (unsigned int q = 0; q < fev.n_q_points(); ++q)
    vol += fev.JxW(q);
  EXPECT_NEAR(vol, 8.0/6.0, tol) << "  p=" << p;
}

TEST_P(FEValues3DTest, QPointsMapCorrectly)
{
  unsigned int p = GetParam();
  FE_DGQLegendre<3, RealType> fe(p);
  QGaussSimplex<3, RealType>  quad(p + 1);
  FEValues<3, RealType> fev(fe, quad);

  // For the unit tet J=I so physical coords == reference coords
  UnitTet cell;
  fev.reinit(cell);

  for (unsigned int q = 0; q < fev.n_q_points(); ++q) {
    auto x  = fev.q_point(q);
    auto xi = quad.point(q);
    for (unsigned int d = 0; d < 3; ++d)
      EXPECT_NEAR(x(d), xi(d), tol) << "  p=" << p << " q=" << q << " d=" << d;
  }
}

// ── FEFaceValues 3D ───────────────────────────────────────────────────────────

class FEFaceValues3DTest : public ::testing::TestWithParam<unsigned int> {};

TEST_P(FEFaceValues3DTest, FaceAreasUnitTet)
{
  unsigned int p = GetParam();
  FE_DGQLegendre<3, RealType> fe(p);
  QGaussSimplex<2, RealType>  fquad(p + 1);
  FEFaceValues<3, RealType>   fefv(fe, fquad);

  UnitTet cell;

  // Three axis-aligned right-triangle faces: area = 0.5
  // Slanted face (opp v0): area = sqrt(3)/2
  const RealType axis_area  = 0.5;
  const RealType slant_area = std::sqrt(3.0) / 2.0;

  for (unsigned int f = 0; f < 4; ++f) {
    fefv.reinit(cell, f);
    RealType area = 0;
    for (unsigned int q = 0; q < fefv.n_q_points(); ++q)
      area += fefv.JxW(q);
    RealType expected = (f == 0) ? slant_area : axis_area;
    EXPECT_NEAR(area, expected, tol) << "  p=" << p << " face=" << f;
  }
}

TEST_P(FEFaceValues3DTest, NormalsAreUnit)
{
  unsigned int p = GetParam();
  FE_DGQLegendre<3, RealType> fe(p);
  QGaussSimplex<2, RealType>  fquad(p + 1);
  FEFaceValues<3, RealType>   fefv(fe, fquad);

  UnitTet cell;
  for (unsigned int f = 0; f < 4; ++f) {
    fefv.reinit(cell, f);
    for (unsigned int q = 0; q < fefv.n_q_points(); ++q)
      EXPECT_NEAR(fefv.normal(q).norm(), 1.0, tol)
          << "  p=" << p << " face=" << f << " q=" << q;
  }
}

TEST_P(FEFaceValues3DTest, NormalsPointOutward)
{
  unsigned int p = GetParam();
  FE_DGQLegendre<3, RealType> fe(p);
  QGaussSimplex<2, RealType>  fquad(p + 1);
  FEFaceValues<3, RealType>   fefv(fe, fquad);

  UnitTet cell;

  // Outward direction for each face (does not need to be unit)
  Tensor<1, 3, RealType> outward[4];
  outward[0](0)= 1; outward[0](1)= 1; outward[0](2)= 1; // face opp v0
  outward[1](0)=-1; outward[1](1)= 0; outward[1](2)= 0; // face opp v1
  outward[2](0)= 0; outward[2](1)=-1; outward[2](2)= 0; // face opp v2
  outward[3](0)= 0; outward[3](1)= 0; outward[3](2)=-1; // face opp v3

  for (unsigned int f = 0; f < 4; ++f) {
    fefv.reinit(cell, f);
    for (unsigned int q = 0; q < fefv.n_q_points(); ++q) {
      auto n = fefv.normal(q);
      RealType d = dot<3, RealType>(n, outward[f]);
      EXPECT_GT(d, 0.0) << "  p=" << p << " face=" << f << " q=" << q;
    }
  }
}

TEST_P(FEFaceValues3DTest, NormalsDotSumToZero)
{
  // For a closed surface, sum of (n_f * area_f) over all faces == 0
  unsigned int p = GetParam();
  FE_DGQLegendre<3, RealType> fe(p);
  QGaussSimplex<2, RealType>  fquad(p + 1);
  FEFaceValues<3, RealType>   fefv(fe, fquad);

  UnitTet cell;
  Tensor<1, 3, RealType> flux_sum;

  for (unsigned int f = 0; f < 4; ++f) {
    fefv.reinit(cell, f);
    for (unsigned int q = 0; q < fefv.n_q_points(); ++q) {
      auto n = fefv.normal(q);
      for (unsigned int d = 0; d < 3; ++d)
        flux_sum(d) += n(d) * fefv.JxW(q);
    }
  }

  for (unsigned int d = 0; d < 3; ++d)
    EXPECT_NEAR(flux_sum(d), 0.0, tol) << "  d=" << d;
}

INSTANTIATE_TEST_SUITE_P(Orders, FEValues3DTest,
                         ::testing::Values(0u, 1u, 2u, 3u));
INSTANTIATE_TEST_SUITE_P(Orders, FEFaceValues3DTest,
                         ::testing::Values(0u, 1u, 2u, 3u));