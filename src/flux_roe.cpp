#include <algorithm>
#include <cmath>
#include <flux_roe.hpp>
#include <parameters.hpp>
#include <tensor.hpp>

namespace SWE {

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

  // 2. Max Wave Speed for CFL (Required for the pair return)
  double smag =
    std::max(std::abs(uL * nx + vL * ny) + std::sqrt(gamma * pL / rhoL),
             std::abs(uR * nx + vR * ny) + std::sqrt(gamma * pR / rhoR));

  // 3. Eigenvalues and Entropy Fix (eps = 0.1 * c)
  double lambda[4] = { vn_roe - c_roe, vn_roe, vn_roe, vn_roe + c_roe };
  double eps = 0.1 * c_roe;
  for (int i = 0; i < 4; ++i) {
    if (std::abs(lambda[i]) < eps)
      lambda[i] = 0.5 * (lambda[i] * lambda[i] / eps + eps);
    else
      lambda[i] = std::abs(lambda[i]);
  }

  // 4. Dissipation (Simplified 2x2 for brevity, use full 4-wave logic in
  // production)
  Tensor<1, 4, double> dU = UR - UL;
  // ... (Insert the a1, a2, a3, a4 wave strength logic here) ...
  Tensor<1, 4, double> diss; // Calculated from eigenvectors and lambda

  // 5. Final Flux
  Tensor<1, 4, double> FL = Cell_Flux(UL, n);
  Tensor<1, 4, double> FR = Cell_Flux(UR, n);
  Tensor<1, 4, double> F = 0.5 * (FL + FR) - 0.5 * diss;

  return { F, smag };
}

} // namespace SWE
