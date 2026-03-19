#pragma once

#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <dof_handler.hpp>
#include <fe.hpp>
#include <iomanip>
#include <iostream>

template<unsigned int dim, typename RealType>
class MassMatrix
{
public:
  MassMatrix(FEValues<dim, RealType>& fe_values)
    : fe_values_(fe_values) {};

  void assemble(const DoFHandler<dim, RealType>& dof_handler)
  {
    const auto n_dofs_per_cell = fe_values_.n_dofs();
    const auto n_q_points = fe_values_.n_q_points();
    const auto n_total_dofs = dof_handler.n_dofs();
    ASSERT(n_dofs_per_cell == dof_handler.n_dofs_per_cell(),
           "Mismatch in DoFs per cell");

    // Allocate view
    // TODO: We technically only need to create one block, which is lower
    // overhead
    mass_matrix_host_ = Kokkos::View<RealType**, Layout, HostMemSpace>(
      "mass_matrix_host_", n_total_dofs, n_total_dofs);

    // Initialize to zero
    Kokkos::deep_copy(mass_matrix_host_, 0.0);

    // Fill in mass matrix
    std::vector<uint32_t> dof_indices;
    for (const auto& cell : dof_handler.active_cell_range()) {
      fe_values_.reinit(cell);

      // Grab the global DoF indices
      cell.get_dof_indices(dof_indices);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
            mass_matrix_host_(dof_indices[i], dof_indices[i]) +=
              fe_values_.shape_value(i, q) * fe_values_.shape_value(j, q) *
              fe_values_.JxW(q);
          }
        }
      }
    }
  }

  /**
   * @brief Grab the inverse of the mass matrix. For orthogonal basis, this is
   * simply a diagonal.
   */
  void apply_inverse(Kokkos::View<RealType*, Layout, HostMemSpace> diag) const
  {
    const unsigned int n = mass_matrix_host_.extent(0);
    for (unsigned int i = 0; i < n; ++i) {
      diag(i) /= mass_matrix_host_(i, i);
    }
  }

  void print(std::ostream& os = std::cout) const
  {
    const unsigned int n = mass_matrix_host_.extent(0);
    for (unsigned int i = 0; i < n; ++i) {
      for (unsigned int j = 0; j < n; ++j)
        os << std::setw(12) << std::scientific << std::setprecision(4)
           << mass_matrix_host_(i, j);
      os << "\n";
    }
  }

  void spy(std::ostream& os = std::cout) const
  {
    const unsigned int n = mass_matrix_host_.extent(0);
    for (unsigned int i = 0; i < n; ++i) {
      for (unsigned int j = 0; j < n; ++j)
        os << (std::abs(mass_matrix_host_(i, j)) > 0.0 ? 'X' : '.');
      os << "\n";
    }
  }

private:
  FEValues<dim, RealType>& fe_values_;

  Kokkos::View<RealType**, Layout, HostMemSpace> mass_matrix_host_;
};
