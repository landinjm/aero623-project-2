#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <iostream>
#include <tensor.hpp>

/**
 * @brief Gauss quadrature on the reference simplex.
 *
 * This code was directly converted from
 * https://github.com/eschnett/SimplexQuad.jl
 */
template<unsigned int dim, typename RealType>
class QGaussSimplex
{
public:
  static constexpr unsigned int n_q_points(unsigned int p)
  {
    unsigned int result = 1;
    for (unsigned int d = 0; d < dim; ++d) {
      result *= p;
    }
    return result;
  }

  explicit QGaussSimplex(unsigned int order)
    : order_(order)
    , points_("quad simplex points", n_q_points(order), dim)
    , weights_("quad simplex weights", n_q_points(order))
  {
    ASSERT(order_ >= 1, "The order must be greater than or equal to 1.");
    compute();
  };

  unsigned int order() const { return order_; }
  unsigned int n_points() const { return n_q_points(order_); }

  Tensor<1, dim, RealType> point(unsigned int q) const
  {
    Tensor<1, dim, RealType> p;
    for (unsigned int d = 0; d < dim; ++d) {
      p(d) = points_(q, d);
    }
    return p;
  }

  RealType weight(unsigned int q) const { return weights_(q); };

private:
  unsigned int order_;

  Kokkos::View<RealType**, Layout, HostMemSpace> points_; // [n_q_points, dim]
  Kokkos::View<RealType*, Layout, HostMemSpace> weights_; // [n_q_points]

  using Vec = Eigen::Matrix<RealType, Eigen::Dynamic, 1>;
  using Mat =
    Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  void compute()
  {
    const int N = static_cast<int>(order_);
    const int n = static_cast<int>(dim); // spatial dim
    const int m = n + 1;                 // number of vertices

    // Standard unit simplex
    Mat vert = Mat::Zero(m, n);
    for (int i = 0; i < n; ++i) {
      vert(i, i) = RealType(1);
    }

    // Special case for 1D
    if (n == 1) {
      compute_1d(N, vert);
      return;
    }

    compute_nd(N, n, m, vert);
  }

  /**
   * 1-D simplex is a line interval
   */
  void compute_1d(int N, const Mat& vert)
  {
    Vec q, w;
    rquad(N, 0, q, w);

    for (int i = 0; i < N; ++i) {
      points_(i, 0) = q[i];
      weights_(i) = w[i];
    }
  }

  /**
   * N-D simplex is triangle, tetrahedra, etc...
   */
  void compute_nd(int N, int n, int m, const Mat& vert)
  {
    std::vector<Vec> qs(n), ws(n);
    for (int k = 0; k < n; ++k) {
      rquad(N, n - k - 1, qs[k], ws[k]);
    }

    std::vector<int> idx(n, 0);

    for (int p = 0; p < n_points(); ++p) {
      RealType weight = RealType(1);
      RealType prod = RealType(1);

      Vec bary(m);
      for (int d = 0; d < n; ++d) {
        RealType t = qs[d][idx[d]];
        bary[d] = prod * (1.0 - t);
        prod *= t;
        weight *= ws[d][idx[d]];
      }
      bary[n] = prod;

      for (int j = 0; j < n; ++j) {
        RealType val = 0.0;
        for (int i = 0; i < m; ++i) {
          val += bary[i] * vert(i, j);
        }
        points_(p, j) = val;
      }
      weights_(p) = weight;

      for (int d = n - 1; d >= 0; --d) {
        if (++idx[d] < N)
          break;
        idx[d] = 0;
      }
    }
  }

  /**
   * 1-D Gauss-Jacobi quadrature rule on [0, 1]
   *
   * Builds the Jacobi tridiagonal matrix, diagonalises it, and then maps the
   * eigenvalues from [-1, 1] to [0, 1].
   */
  static void rquad(int N, int k, Vec& q, Vec& w)
  {
    RealType k1 = RealType(k + 1);
    RealType k2 = RealType(k + 2);

    Vec n(N);
    for (int i = 0; i < N; ++i) {
      n[i] = i + 1;
    }

    Vec nnk(N);
    for (int i = 0; i < N; ++i) {
      nnk[i] = 2 * n[i] + k;
    }

    Vec A(N + 1);
    A[0] = RealType(k) / k2;
    for (int i = 0; i < N; ++i) {
      A[i + 1] = RealType(k * k) / (nnk[i] * (nnk[i] + RealType(2)));
    }

    RealType ab0_col2 = std::pow(RealType(2), k1) / k1;
    RealType B1 = RealType(4) * k1 / (k2 * k2 * RealType(k + 3));

    Vec B(N - 1);
    for (int i = 1; i < N; ++i) {
      RealType nk = n[i] + k;
      RealType nnk2 = nnk[i] * nnk[i];
      B[i - 1] = RealType(4) * (n[i] * nk) * (n[i] * nk) / (nnk2 * nnk2 - nnk2);
    }

    Vec diag(N);
    for (int i = 0; i < N; ++i) {
      diag[i] = A[i];
    }

    Vec s(N - 1);
    if (N > 1) {
      s[0] = std::sqrt(B1);
      for (int i = 1; i < N - 1; ++i) {
        s[i] = std::sqrt(B[i - 1]);
      }
    }

    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(N, N);
    for (int i = 0; i < N; ++i) {
      T(i, i) = diag[i];
    }
    for (int i = 0; i < N - 1; ++i) {
      T(i, i + 1) = s[i];
    }
    for (int i = 0; i < N - 1; ++i) {
      T(i + 1, i) = s[i];
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(T);
    Vec X = solver.eigenvalues();
    auto V = solver.eigenvectors();

    q.resize(N);
    for (int i = 0; i < N; ++i) {
      q[i] = (X[i] + 1.0) / 2.0;
    }

    RealType weight_scale = std::pow(0.5, k1) * ab0_col2;
    w.resize(N);
    for (int i = 0; i < N; ++i) {
      RealType v0i = V(0, i);
      w[i] = weight_scale * v0i * v0i;
    }
  }
};
