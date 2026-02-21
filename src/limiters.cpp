#include "limiters.hpp"
#include <algorithm>
#include <array>
#include <cmath>

// Small 2x2 solver for symmetric normal equations
static inline bool solve2x2(double a11, double a12, double a22, double b1,
                            double b2, double &gx, double &gy) {
  const double det = a11 * a22 - a12 * a12;
  if (std::abs(det) < 1e-14)
    return false;

  const double invDet = 1.0 / det;
  gx = (b1 * a22 - b2 * a12) * invDet;
  gy = (-b1 * a12 + b2 * a11) * invDet;
  return true;
}

// Return periodic translation vector (dx,dy) that maps elem_id -> neighbor
// across that periodic face. If your Triangulation stores it elsewhere, adapt
// this function accordingly.
static inline std::pair<double, double>
periodic_shift_for_face(const Triangulation<2, double> &tri, int f,
                        int elem_id) {
  // In many AE623 codes, periodic faces are *translational* with known shift.
  // If your Triangulation already applied periodicity by duplicating geometry
  // (common), then the shift is (0,0) and this is fine.
  //
  // If you DO have a stored shift per periodic face, you must use it here.
  //
  // Default safe behavior: no shift.
  (void)tri;
  (void)f;
  (void)elem_id;
  return {0.0, 0.0};
}

void ComputeGradients(const Triangulation<2, double> &tri,
                      ElementData<2, double> &elem, int elem_id,
                      std::array<Tensor<1, 2, double>, 4> &L0) {
  const auto &interior = tri.get_interior_faces();
  const auto &boundary = tri.get_boundary_faces();
  const auto &periodic = tri.get_periodic_faces();

  // zero output
  for (int eq = 0; eq < 4; ++eq) {
    L0[eq][0] = 0.0;
    L0[eq][1] = 0.0;
  }

  // Cell centroid
  const double xi = elem.centroid_x[elem_id];
  const double yi = elem.centroid_y[elem_id];

  // Build LS normal-equation matrix A^T A (2x2) and RHS for each equation
  double a11 = 0.0, a12 = 0.0, a22 = 0.0;
  std::array<double, 4> b1 = {0.0, 0.0, 0.0, 0.0};
  std::array<double, 4> b2 = {0.0, 0.0, 0.0, 0.0};

  auto ui =
      Tensor<1, 4, double>{elem.density[elem_id], elem.momentum_x[elem_id],
                           elem.momentum_y[elem_id], elem.energy[elem_id]};

  // Helper: add one neighbor sample (xj,yj, uj)
  auto add_sample = [&](double xj, double yj, const Tensor<1, 4, double> &uj,
                        double w) {
    const double dx = xj - xi;
    const double dy = yj - yi;

    // ignore degenerate samples
    const double r2 = dx * dx + dy * dy;
    if (r2 < 1e-30)
      return;

    // weight (you can use w=1.0; or w = 1/r2; both work. Keep simple/robust.)
    const double ww = w;

    a11 += ww * dx * dx;
    a12 += ww * dx * dy;
    a22 += ww * dy * dy;

    for (int eq = 0; eq < 4; ++eq) {
      const double du = uj[eq] - ui[eq];
      b1[eq] += ww * dx * du;
      b2[eq] += ww * dy * du;
    }
  };

  // =========================
  // Interior neighbors
  // =========================
  for (int f = 0; f < (int)interior.size(); ++f) {
    const int L = (int)interior.elem_l[f];
    const int R = (int)interior.elem_r[f];

    int nbr = -1;
    if (elem_id == L)
      nbr = R;
    else if (elem_id == R)
      nbr = L;
    else
      continue;

    const double xj = elem.centroid_x[nbr];
    const double yj = elem.centroid_y[nbr];

    Tensor<1, 4, double> uj = {elem.density[nbr], elem.momentum_x[nbr],
                               elem.momentum_y[nbr], elem.energy[nbr]};

    add_sample(xj, yj, uj, 1.0);
  }

  // =========================
  // Periodic neighbors
  // =========================
  for (int f = 0; f < (int)periodic.size(); ++f) {
    const int L = (int)periodic.elem_l[f];
    const int R = (int)periodic.elem_r[f];

    int nbr = -1;
    if (elem_id == L)
      nbr = R;
    else if (elem_id == R)
      nbr = L;
    else
      continue;

    // neighbor centroid, plus periodic translation if needed
    double xj = elem.centroid_x[nbr];
    double yj = elem.centroid_y[nbr];

    auto [sx, sy] = periodic_shift_for_face(tri, f, elem_id);
    xj += sx;
    yj += sy;

    Tensor<1, 4, double> uj = {elem.density[nbr], elem.momentum_x[nbr],
                               elem.momentum_y[nbr], elem.energy[nbr]};

    add_sample(xj, yj, uj, 1.0);
  }

  // =========================
  // Boundary: do NOTHING for LS
  // =========================
  // Key point: for LS gradients, you typically do NOT add a fake boundary
  // "sample" equal to the interior state. That contributes du=0 and adds no RHS
  // information, but it *can* distort the matrix if you instead used Uhat or
  // something else.
  //
  // For your current Green–Gauss method, boundaries matter.
  // For LS, stencil is based on neighboring cell values.
  (void)boundary;

  // Solve for each conserved variable gradient
  for (int eq = 0; eq < 4; ++eq) {
    double gx = 0.0, gy = 0.0;
    const bool ok = solve2x2(a11, a12, a22, b1[eq], b2[eq], gx, gy);

    if (!ok) {
      // Fallback: zero gradient (or you can fallback to Green–Gauss here)
      gx = 0.0;
      gy = 0.0;
    }

    L0[eq][0] = gx;
    L0[eq][1] = gy;
  }
}

void Limiter_BJ(std::array<Tensor<1, 2, double>, 4> &L0,
                Tensor<1, 4, double> &u0, Tensor<1, 4, double> &umin,
                Tensor<1, 4, double> &umax,
                std::array<Tensor<1, 2, double>, 3> &r) {

  // define the tensor product between the gradient and the vector to the cell
  // nodes
  Tensor<1, 4, double> Lr1 = {r[0][0] * L0[0][0] + r[0][1] * L0[0][1],
                              r[0][0] * L0[1][0] + r[0][1] * L0[1][1],
                              r[0][0] * L0[2][0] + r[0][1] * L0[2][1],
                              r[0][0] * L0[3][0] + r[0][1] * L0[3][1]};

  Tensor<1, 4, double> Lr2 = {r[1][0] * L0[0][0] + r[1][1] * L0[0][1],
                              r[1][0] * L0[1][0] + r[1][1] * L0[1][1],
                              r[1][0] * L0[2][0] + r[1][1] * L0[2][1],
                              r[1][0] * L0[3][0] + r[1][1] * L0[3][1]};

  Tensor<1, 4, double> Lr3 = {r[2][0] * L0[0][0] + r[2][1] * L0[0][1],
                              r[2][0] * L0[1][0] + r[2][1] * L0[1][1],
                              r[2][0] * L0[2][0] + r[2][1] * L0[2][1],
                              r[2][0] * L0[3][0] + r[2][1] * L0[3][1]};

  // get adjacent states
  Tensor<1, 4, double> u1 = u0 + Lr1;
  Tensor<1, 4, double> u2 = u0 + Lr2;
  Tensor<1, 4, double> u3 = u0 + Lr3;
  std::array<Tensor<1, 4, double>, 3> ui = {u1, u2, u3};

  // find the scalar limiter
  Tensor<1, 4, double> alpha = {1, 1, 1, 1};
  // loop through the nodes of an element
  for (int i = 0; i < 3; i++) {
    double alpha_cmp;
    const auto &uiN = ui[i];

    // loop through the states
    for (int j = 0; j < 4; j++) {

      // compute the required scalar limiter for each node
      if (uiN[j] - u0[j] > 0.0) {
        alpha_cmp = std::min(1.0, (umax[j] - u0[j]) / (uiN[j] - u0[j]));
      } else if (uiN[j] - u0[j] < 0.0) {
        alpha_cmp = std::min(1.0, (umin[j] - u0[j]) / (uiN[j] - u0[j]));
      } else {
        alpha_cmp = 1.0;
      } // end if

      // set scalar limiter to the minimum of the adjacent nodes
      alpha[j] = std::min(alpha[j], alpha_cmp);
    } // end state for
  } // end node for

  // return the limited gradient
  // loop through the states
  for (int j = 0; j < 4; j++) {
    // loop through the gradients
    L0[j][0] = alpha[j] * L0[j][0];
    L0[j][1] = alpha[j] * L0[j][1];
  }
} // end Limiter_BJ

Tensor<1, 4, double> neighbormin(int elem_id,
                                 const ElementData<2, double> &elem,
                                 const Triangulation<2, double> &tri) {
  const auto &interior = tri.get_interior_faces();
  const auto &periodic = tri.get_periodic_faces();

  // Initialize min with this cell's state
  Tensor<1, 4, double> umin = {elem.density[elem_id], elem.momentum_x[elem_id],
                               elem.momentum_y[elem_id], elem.energy[elem_id]};

  // -------- Interior Faces --------
  for (int f = 0; f < interior.size(); ++f) {
    int L = interior.elem_l[f];
    int R = interior.elem_r[f];

    int neighbor = -1;

    if (elem_id == L)
      neighbor = R;
    else if (elem_id == R)
      neighbor = L;
    else
      continue;

    Tensor<1, 4, double> uN = {
        elem.density[neighbor], elem.momentum_x[neighbor],
        elem.momentum_y[neighbor], elem.energy[neighbor]};

    for (int eq = 0; eq < 4; ++eq)
      umin[eq] = std::min(umin[eq], uN[eq]);
  }

  // -------- Periodic Faces --------
  for (int f = 0; f < periodic.size(); ++f) {
    int L = periodic.elem_l[f];
    int R = periodic.elem_r[f];

    int neighbor = -1;

    if (elem_id == L)
      neighbor = R;
    else if (elem_id == R)
      neighbor = L;
    else
      continue;

    Tensor<1, 4, double> uN = {
        elem.density[neighbor], elem.momentum_x[neighbor],
        elem.momentum_y[neighbor], elem.energy[neighbor]};

    for (int eq = 0; eq < 4; ++eq)
      umin[eq] = std::min(umin[eq], uN[eq]);
  }

  return umin;
}

Tensor<1, 4, double> neighbormax(int elem_id,
                                 const ElementData<2, double> &elem,
                                 const Triangulation<2, double> &tri) {
  const auto &interior = tri.get_interior_faces();
  const auto &periodic = tri.get_periodic_faces();

  // Initialize max with this cell's state
  Tensor<1, 4, double> umax = {elem.density[elem_id], elem.momentum_x[elem_id],
                               elem.momentum_y[elem_id], elem.energy[elem_id]};

  // -------- Interior Faces --------
  for (int f = 0; f < interior.size(); ++f) {
    int L = interior.elem_l[f];
    int R = interior.elem_r[f];

    int neighbor = -1;

    if (elem_id == L)
      neighbor = R;
    else if (elem_id == R)
      neighbor = L;
    else
      continue;

    Tensor<1, 4, double> uN = {
        elem.density[neighbor], elem.momentum_x[neighbor],
        elem.momentum_y[neighbor], elem.energy[neighbor]};

    for (int eq = 0; eq < 4; ++eq)
      umax[eq] = std::max(umax[eq], uN[eq]);
  }

  // -------- Periodic Faces --------
  for (int f = 0; f < periodic.size(); ++f) {
    int L = periodic.elem_l[f];
    int R = periodic.elem_r[f];

    int neighbor = -1;

    if (elem_id == L)
      neighbor = R;
    else if (elem_id == R)
      neighbor = L;
    else
      continue;

    Tensor<1, 4, double> uN = {
        elem.density[neighbor], elem.momentum_x[neighbor],
        elem.momentum_y[neighbor], elem.energy[neighbor]};

    for (int eq = 0; eq < 4; ++eq)
      umax[eq] = std::max(umax[eq], uN[eq]);
  }

  return umax;
}

std::array<Tensor<1, 2, double>, 3>
ComputeVertexVectors(int elem_id, const MeshData &data,
                     const ElementData<2, double> &elem) {
  int v0 = data.node_1[elem_id];
  int v1 = data.node_2[elem_id];
  int v2 = data.node_3[elem_id];

  double cx = elem.centroid_x[elem_id];
  double cy = elem.centroid_y[elem_id];

  Tensor<1, 2, double> r0 = {data.x[v0] - cx, data.y[v0] - cy};

  Tensor<1, 2, double> r1 = {data.x[v1] - cx, data.y[v1] - cy};

  Tensor<1, 2, double> r2 = {data.x[v2] - cx, data.y[v2] - cy};

  return {r0, r1, r2};
} // end ComputeVertexVectors
