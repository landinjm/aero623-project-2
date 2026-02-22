#include <algorithm>
#include <cassert>
#include <cmath>
#include <flux.hpp>
#include <parameters.hpp>
#include <utility>

std::pair<Tensor<1, 4, double>, double>
flux_roe(Tensor<1, 4, double> UL,
         Tensor<1, 4, double> UR,
         Tensor<1, 2, double> n)
{
  double gamma = Parameters<double>::gamma;
  double nx = n[0];
  double ny = n[1];

  // 1. Primitive variables and Roe Averages
  double rhoL = UL[0];
  double uL = UL[1] / rhoL;
  double vL = UL[2] / rhoL;
  double pL = (gamma - 1.0) * (UL[3] - 0.5 * rhoL * (uL * uL + vL * vL));
  double HL = (UL[3] + pL) / rhoL;

  double rhoR = UR[0];
  double uR = UR[1] / rhoR;
  double vR = UR[2] / rhoR;
  double pR = (gamma - 1.0) * (UR[3] - 0.5 * rhoR * (uR * uR + vR * vR));
  double HR = (UR[3] + pR) / rhoR;

  double sL = std::sqrt(rhoL);
  double sR = std::sqrt(rhoR);
  double sLR = sL + sR;

  double u_roe = (sL * uL + sR * uR) / sLR;
  double v_roe = (sL * vL + sR * vR) / sLR;
  double H_roe = (sL * HL + sR * HR) / sLR;
  double V2 = u_roe * u_roe + v_roe * v_roe;
  double c_roe = std::sqrt((gamma - 1.0) * (H_roe - 0.5 * V2));
  double vn_roe = u_roe * nx + v_roe * ny;

  // 2. Max Wave Speed for CFL
  double smag =
    std::max(std::abs(uL * nx + vL * ny) + std::sqrt(gamma * pL / rhoL),
             std::abs(uR * nx + vR * ny) + std::sqrt(gamma * pR / rhoR));

  // 3. Eigenvalues and Entropy Fix
  double lambda[4] = { vn_roe - c_roe, vn_roe, vn_roe, vn_roe + c_roe };
  double eps = 0.1 * c_roe;

  for (int i = 0; i < 4; ++i) {
    // Only apply the Harten fix to the non-linear acoustic waves (0 and 3)
    if (i == 0 || i == 3) {
      if (std::abs(lambda[i]) < eps)
        lambda[i] = 0.5 * (lambda[i] * lambda[i] / eps + eps);
      else
        lambda[i] = std::abs(lambda[i]);
    } else {
      // Linear waves (entropy and shear) usually don't need the expansion shock
      // fix
      lambda[i] = std::abs(lambda[i]);
    }
  }

  // 4. Dissipation
  Tensor<1, 4, double> diss;
  double rho_roe = std::sqrt(rhoL * rhoR);

  double drho = UR[0] - UL[0];
  double du = uR - uL;
  double dv = vR - vL;
  double dp = pR - pL;
  double d_un = du * nx + dv * ny;
  double d_ut = du * (-ny) + dv * nx;

  // Wave strengths
  double a1 = 0.5 * (dp - rho_roe * c_roe * d_un) / (c_roe * c_roe);
  double a4 = 0.5 * (dp + rho_roe * c_roe * d_un) / (c_roe * c_roe);
  double a2 = drho - dp / (c_roe * c_roe);
  double a3 = rho_roe * d_ut;

  // Mass flux dissipation: diss[0] = |L1|*a1 + |L2|*a2 + |L4|*a4
  // If u=0, then a1 = a4.
  // Then diss[0] = |vn-c|*a1 + |vn|*a2 + |vn+c|*a1
  // Without the entropy fix on lambda[1], |vn| is 0, and diss[0] becomes
  // 2*|c|*a1. Since a1 = 0.5 * dp/c^2, diss[0] = dp/c. In your test, dp=0, so
  // diss[0] should be 0.

  diss[0] = lambda[0] * a1 + lambda[1] * a2 + lambda[3] * a4;

  diss[1] = lambda[0] * a1 * (u_roe - c_roe * nx) + lambda[1] * a2 * u_roe +
            lambda[2] * a3 * (-ny) + lambda[3] * a4 * (u_roe + c_roe * nx);

  diss[2] = lambda[0] * a1 * (v_roe - c_roe * ny) + lambda[1] * a2 * v_roe +
            lambda[2] * a3 * nx + lambda[3] * a4 * (v_roe + c_roe * ny);

  diss[3] = lambda[0] * a1 * (H_roe - c_roe * vn_roe) +
            lambda[1] * a2 * 0.5 * V2 +
            lambda[2] * a3 * (u_roe * (-ny) + v_roe * nx) +
            lambda[3] * a4 * (H_roe + c_roe * vn_roe);

  // 5. Final Flux
  Tensor<1, 4, double> FL = euler_flux(UL, n);
  Tensor<1, 4, double> FR = euler_flux(UR, n);
  Tensor<1, 4, double> F = 0.5 * (FL + FR) - 0.5 * diss;

  return { F, smag };
}

std::pair<Tensor<1, 4, double>, double>
flux_hlle(Tensor<1, 4, double> UL,
          Tensor<1, 4, double> UR,
          Tensor<1, 2, double> n)
{
  double gamma = Parameters<double>::gamma;

  // unpack the state
  double rhoL = UL[0];
  double uL = UL[1] / rhoL;
  double vL = UL[2] / rhoL;
  double unL = uL * n[0] + vL * n[1];
  double qL = std::sqrt(std::pow(UL[1], 2) + std::pow(UL[2], 2)) / rhoL;
  double pL = (gamma - 1) * (UL[3] - 0.5 * rhoL * std::pow(qL, 2));
  assert(pL >= 0 && rhoL >= 0);
  double cL = std::sqrt(gamma * pL / rhoL);

  double rhoR = UR[0];
  double uR = UR[1] / rhoR;
  double vR = UR[2] / rhoR;
  double unR = uR * n[0] + vR * n[1];
  double qR = std::sqrt(std::pow(UR[1], 2) + std::pow(UR[2], 2)) / rhoR;
  double pR = (gamma - 1) * (UR[3] - 0.5 * rhoR * std::pow(qR, 2));
  assert(pR >= 0 && rhoR >= 0);
  double cR = std::sqrt(gamma * pR / rhoR);

  // calculate wave speeds
  double sLmin = std::min(0.0, unL - cL);
  double sRmin = std::min(0.0, unR - cR);
  double sLmax = std::max(0.0, unL + cL);
  double sRmax = std::max(0.0, unR + cR);

  double sLmag = std::abs(unL) + cL;
  double sRmag = std::abs(unR) + cR;

  double smag = std::max(sLmag, sRmag);
  double sLRmin = std::min(sLmin, sRmin);
  double sLRmax = std::max(sLmax, sRmax);

  // calculate he left and right fluxes
  Tensor<1, 4, double> FL = euler_flux(UL, n);
  Tensor<1, 4, double> FR = euler_flux(UR, n);
  ;
  Tensor<1, 4, double> F;

  F = 0.5 * (FL + FR) -
      0.5 * (sLRmax + sLRmin) / (sLRmax - sLRmin) * (FR - FL) +
      sLRmax * sLRmin / (sLRmax - sLRmin) * (UR - UL);

  // return the numerical flux and wave speed
  std::pair<Tensor<1, 4, double>, double> exports = { F, smag };
  return exports;
}
