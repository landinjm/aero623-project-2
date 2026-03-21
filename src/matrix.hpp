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
    const auto ndpc = fe_values_.n_dofs();
    const auto n_q = fe_values_.n_q_points();
    const auto n_cells = dof_handler.n_cells();

    ASSERT(ndpc == dof_handler.n_dofs_per_cell(), "Mismatch in DoFs per cell");

    // Store as [n_cells, ndpc, ndpc] blocks — block diagonal structure
    auto mm_h = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "mm_h", n_cells, ndpc, ndpc);
    Kokkos::deep_copy(mm_h, RealType(0));

    for (const auto& cell : dof_handler.active_cell_range()) {
      fe_values_.reinit(cell);
      const auto k = cell.index();

      for (unsigned int q = 0; q < n_q; ++q)
        for (unsigned int i = 0; i < ndpc; ++i)
          for (unsigned int j = 0; j < ndpc; ++j)
            mm_h(k, i, j) += fe_values_.shape_value(i, q) *
                             fe_values_.shape_value(j, q) * fe_values_.JxW(q);
    }

    // Copy blocks to device
    mm_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "mm", n_cells, ndpc, ndpc);
    Kokkos::deep_copy(mm_, mm_h);

    // Allocate inverse on device
    invm_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "invm", n_cells, ndpc, ndpc);
  }

  void invert()
  {
    const int ndpc = static_cast<int>(mm_.extent(1));
    const int n_cells = static_cast<int>(mm_.extent(0));

    auto M = mm_;
    auto Minv = invm_;

    Kokkos::parallel_for(
      "mass_matrix_invert", n_cells, KOKKOS_LAMBDA(int k) {
        // Copy block into local arrays (lives in registers/local memory)
        RealType aug[10][20]; // max_dofs=10, augmented [M|I]

        for (int i = 0; i < ndpc; ++i) {
          for (int j = 0; j < ndpc; ++j) {
            aug[i][j] = M(k, i, j);
            aug[i][ndpc + j] = (i == j) ? RealType(1) : RealType(0);
          }
        }

        // Gauss-Jordan elimination with partial pivoting
        for (int col = 0; col < ndpc; ++col) {
          // Find pivot
          int pivot = col;
          RealType max_val = Kokkos::abs(aug[col][col]);
          for (int row = col + 1; row < ndpc; ++row) {
            RealType val = Kokkos::abs(aug[row][col]);
            if (val > max_val) {
              max_val = val;
              pivot = row;
            }
          }

          // Swap rows
          if (pivot != col) {
            for (int j = 0; j < 2 * ndpc; ++j) {
              RealType tmp = aug[col][j];
              aug[col][j] = aug[pivot][j];
              aug[pivot][j] = tmp;
            }
          }

          // Scale pivot row
          const RealType scale = RealType(1) / aug[col][col];
          for (int j = 0; j < 2 * ndpc; ++j)
            aug[col][j] *= scale;

          // Eliminate column
          for (int row = 0; row < ndpc; ++row) {
            if (row == col)
              continue;
            const RealType factor = aug[row][col];
            for (int j = 0; j < 2 * ndpc; ++j)
              aug[row][j] -= factor * aug[col][j];
          }
        }

        // Extract inverse from right half of augmented matrix
        for (int i = 0; i < ndpc; ++i)
          for (int j = 0; j < ndpc; ++j)
            Minv(k, i, j) = aug[i][ndpc + j];
      });
    Kokkos::fence();
  }

  void check_inverse(std::ostream& os = std::cout) const
  {
    const unsigned int n_cells = mm_.extent(0);
    const unsigned int ndpc = mm_.extent(1);

    auto mm_h = Kokkos::create_mirror_view(mm_);
    auto invm_h = Kokkos::create_mirror_view(invm_);
    Kokkos::deep_copy(mm_h, mm_);
    Kokkos::deep_copy(invm_h, invm_);

    RealType max_err = 0;
    unsigned int worst_cell = 0;

    for (unsigned int k = 0; k < n_cells; ++k) {
      for (unsigned int i = 0; i < ndpc; ++i) {
        for (unsigned int j = 0; j < ndpc; ++j) {
          // Compute (M * Minv)[i,j]
          RealType val = 0;
          for (unsigned int l = 0; l < ndpc; ++l)
            val += mm_h(k, i, l) * invm_h(k, l, j);

          // Should be 1 on diagonal, 0 off diagonal
          const RealType expected = (i == j) ? RealType(1) : RealType(0);
          const RealType err = Kokkos::abs(val - expected);
          if (err > max_err) {
            max_err = err;
            worst_cell = k;
          }
        }
      }
    }

    os << "Mass matrix inverse check:\n"
       << "  max |M * Minv - I| = " << max_err << " (worst cell=" << worst_cell
       << ")\n";

    if (max_err < 1e-10)
      os << "  PASSED\n";
    else
      os << "  FAILED\n";
  }

  Kokkos::View<RealType***, Layout, DeviceMemSpace> device_inverse() const
  {
    return invm_;
  }

  void print(std::ostream& os = std::cout) const
  {
    const unsigned int n_cells = mm_.extent(0);
    const unsigned int ndpc = mm_.extent(1);
    auto mm_h = Kokkos::create_mirror_view(mm_);
    Kokkos::deep_copy(mm_h, mm_);
    for (unsigned int k = 0; k < n_cells; ++k) {
      os << "Cell " << k << ":\n";
      for (unsigned int i = 0; i < ndpc; ++i) {
        for (unsigned int j = 0; j < ndpc; ++j)
          os << std::setw(12) << std::scientific << std::setprecision(4)
             << mm_h(k, i, j);
        os << "\n";
      }
    }
  }

  void spy(std::ostream& os = std::cout) const
  {
    const unsigned int n_cells = mm_.extent(0);
    const unsigned int ndpc = mm_.extent(1);
    auto mm_h = Kokkos::create_mirror_view(mm_);
    Kokkos::deep_copy(mm_h, mm_);
    for (unsigned int k = 0; k < n_cells; ++k) {
      os << "Cell " << k << ":\n";
      for (unsigned int i = 0; i < ndpc; ++i) {
        for (unsigned int j = 0; j < ndpc; ++j)
          os << (std::abs(mm_h(k, i, j)) > 0.0 ? 'X' : '.');
        os << "\n";
      }
    }
  }

private:
  FEValues<dim, RealType>& fe_values_;

  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    mm_; // [n_cells, n_dofs_per_cell, n_dofs_per_cell]

  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    invm_; // [n_cells, n_dofs_per_cell, n_dofs_per_cell]
};
