#pragma once

#include <Kokkos_Core.hpp>
#include "fe.hpp"
#include "flux.hpp"
#include "tensor.hpp"

template <unsigned int dim, typename RealType>
class PositivityLimiter {
public:
    // eps should be slightly larger than the floor in conservative_to_primitive
    static constexpr RealType eps = 1e-10; 

    KOKKOS_INLINE_FUNCTION
    static void apply(
        const unsigned int n_dofs_per_cell,
        const unsigned int n_q_points,
        const Kokkos::View<RealType**, Kokkos::LayoutStride, Kokkos::HostSpace> basis_values,
        RealType* rho,
        RealType* rho_u,
        RealType* rho_v,
        RealType* rho_w,
        RealType* rho_E) 
    {
        RealType rho_avg = 0, rho_E_avg = 0;
        RealType rhou_avg = 0, rhov_avg = 0, rhow_avg = 0;

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
            rho_avg  += rho[i];
            rhou_avg += rho_u[i];
            rhov_avg += rho_v[i];
            rhow_avg += rho_w[i];
            rho_E_avg += rho_E[i];
        }
        rho_avg   /= n_dofs_per_cell;
        rhou_avg  /= n_dofs_per_cell;
        rhov_avg  /= n_dofs_per_cell;
        rhow_avg  /= n_dofs_per_cell;
        rho_E_avg /= n_dofs_per_cell;

        RealType theta = 1.0;

        // 2. Check Density at all quadrature points
        for (unsigned int q = 0; q < n_q_points; ++q) {
            RealType rho_q = 0;
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
                rho_q += rho[i] * basis_values(i, q);
            }

            if (rho_q < eps) {
                RealType ratio = (rho_avg - eps) / (rho_avg - rho_q + 1e-16);
                theta = Kokkos::min(theta, Kokkos::max(RealType(0), ratio));
            }
        }

        // 3. Apply theta to density and re-check pressure
        // Note: Pressure limiting is a quadratic inequality check. 
        // For p=1, checking quadrature points is usually sufficient.
        for (unsigned int q = 0; q < n_q_points; ++q) {
            RealType r_q = 0, re_q = 0, ru_q = 0, rv_q = 0, rw_q = 0;
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
                RealType phi = basis_values(i, q);
                r_q  += (rho_avg   + theta * (rho[i]   - rho_avg))   * phi;
                ru_q += (rhou_avg  + theta * (rho_u[i] - rhou_avg))  * phi;
                rv_q += (rhov_avg  + theta * (rho_v[i] - rhov_avg))  * phi;
                rw_q += (rhow_avg  + theta * (rho_w[i] - rhow_avg))  * phi;
                re_q += (rho_E_avg + theta * (rho_E[i] - rho_E_avg)) * phi;
            }
            Tensor<1, dim, RealType> v_q;
            v_q[0] = ru_q / r_q;
            v_q[1] = rv_q / r_q;
            v_q[2] = rw_q / r_q;
            RealType p_q = Flux<dim, RealType>::pressure(r_q, v_q, re_q);
        }

        // 4. Update the actual DoFs
        // 3. Iteratively reduce theta until pressure is non-negative at all q points
        for (int iter = 0; iter < 50; ++iter) {
            bool pressure_ok = true;
            for (unsigned int q = 0; q < n_q_points; ++q) {
                RealType r_q = 0, re_q = 0, ru_q = 0, rv_q = 0, rw_q = 0;
                for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
                    RealType phi = basis_values(i, q);
                    r_q  += (rho_avg   + theta * (rho[i]   - rho_avg))   * phi;
                    ru_q += (rhou_avg  + theta * (rho_u[i] - rhou_avg))  * phi;
                    rv_q += (rhov_avg  + theta * (rho_v[i] - rhov_avg))  * phi;
                    rw_q += (rhow_avg  + theta * (rho_w[i] - rhow_avg))  * phi;
                    re_q += (rho_E_avg + theta * (rho_E[i] - rho_E_avg)) * phi;
                }
                Tensor<1, dim, RealType> v_q;
                v_q[0] = ru_q / r_q;
                v_q[1] = rv_q / r_q;
                v_q[2] = rw_q / r_q;
                RealType p_q = Flux<dim, RealType>::pressure(r_q, v_q, re_q);

                if (p_q < eps) {
                    pressure_ok = false;
                    break;
                }
            }
            if (pressure_ok) break;
            theta *= 0.5;
        }
    }
};