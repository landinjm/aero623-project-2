#include "flux.hpp"
#include <cmath>
#include <algorithm>

namespace SWE {

std::vector<double> flux_roe(const std::vector<double>& uL, 
                             const std::vector<double>& uR, 
                             double nx, double ny, 
                             double gamma) {
    
    // 1. Primitive variables and pressure
    auto get_prim = [gamma](const std::vector<double>& u) {
        double rho = u[0];
        double vel_u = u[1] / rho;
        double vel_v = u[2] / rho;
        double p = (gamma - 1.0) * (u[3] - 0.5 * rho * (vel_u * vel_u + vel_v * vel_v));
        double H = (u[3] + p) / rho; 
        return std::vector<double>{rho, vel_u, vel_v, p, H};
    };

    std::vector<double> pL = get_prim(uL);
    std::vector<double> pR = get_prim(uR);

    // 2. Physical fluxes in normal direction
    auto get_normal_flux = [nx, ny](const std::vector<double>& u, const std::vector<double>& p) {
        double vn = p[1] * nx + p[2] * ny; 
        return std::vector<double>{
            u[0] * vn,
            u[1] * vn + p[3] * nx,
            u[2] * vn + p[3] * ny,
            (u[3] + p[3]) * vn
        };
    };

    std::vector<double> FL = get_normal_flux(uL, pL);
    std::vector<double> FR = get_normal_flux(uR, pR);

    // 3. Roe averages
    double sL = std::sqrt(pL[0]);
    double sR = std::sqrt(pR[0]);
    double sLR = sL + sR;

    double u_roe = (sL * pL[1] + sR * pR[1]) / sLR;
    double v_roe = (sL * pL[2] + sR * pR[2]) / sLR;
    double H_roe = (sL * pL[4] + sR * pR[4]) / sLR;
    
    double V2 = u_roe * u_roe + v_roe * v_roe;
    double c_roe = std::sqrt((gamma - 1.0) * (H_roe - 0.5 * V2));
    double vn_roe = u_roe * nx + v_roe * ny;

    // 4. Eigenvalues and entropy fix (Required: eps = 0.1 * c) [cite: 20]
    double lambda[4] = {vn_roe - c_roe, vn_roe, vn_roe, vn_roe + c_roe};
    double eps = 0.1 * c_roe; 

    for(int i = 0; i < 4; ++i) {
        if (std::abs(lambda[i]) < eps)
            lambda[i] = 0.5 * (lambda[i] * lambda[i] / eps + eps);
        else
            lambda[i] = std::abs(lambda[i]);
    }

    // 5. Dissipation term
    double dU[4] = {uR[0]-uL[0], uR[1]-uL[1], uR[2]-uL[2], uR[3]-uL[3]};
    double g1 = gamma - 1.0;
    double a2 = g1 * (V2 * dU[0] - 2.0 * (u_roe * dU[1] + v_roe * dU[2]) + 2.0 * dU[3]) / (2.0 * c_roe * c_roe);
    double a1 = ( (c_roe + vn_roe) * dU[0] - (dU[1] * nx + dU[2] * ny) - c_roe * a2 ) / (2.0 * c_roe);
    double a4 = dU[0] - a1 - a2;
    double a3 = ( (dU[1] * (-ny) + dU[2] * nx) - (u_roe * (-ny) + v_roe * nx) * dU[0] ) / c_roe;

    double w[4] = {a1 * lambda[0], a2 * lambda[1], a3 * lambda[2], a4 * lambda[3]};
    double diss[4];
    diss[0] = w[0] + w[1] + w[3];
    diss[1] = w[0]*(u_roe - c_roe*nx) + w[1]*u_roe + w[2]*(-ny*c_roe) + w[3]*(u_roe + c_roe*nx);
    diss[2] = w[0]*(v_roe - c_roe*ny) + w[1]*v_roe + w[2]*(nx*c_roe)  + w[3]*(v_roe + c_roe*ny);
    diss[3] = w[0]*(H_roe - c_roe*vn_roe) + w[1]*0.5*V2 + w[2]*c_roe*(nx*v_roe - ny*u_roe) + w[3]*(H_roe + c_roe*vn_roe);

    return {
        0.5 * (FL[0] + FR[0] - diss[0]),
        0.5 * (FL[1] + FR[1] - diss[1]),
        0.5 * (FL[2] + FR[2] - diss[2]),
        0.5 * (FL[3] + FR[3] - diss[3])
    };
}

} // namespace SWE