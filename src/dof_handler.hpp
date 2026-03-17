#pragma once

#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <fe.hpp>
#include <triangulation.hpp>

/**
 * @brief Collection of all degrees of freedom across the mesh.
 *
 * Since we have a DG scheme, each cell owns its own dofs.
 */
template<unsigned int dim, typename RealType>
class DoFHandler
{
public:
  DoFHandler(const Triangulation<dim>& tria, const FE_DGP<dim, RealType>& fe)
    : tria_(tria)
    , fe_(fe)
    , n_cells_(tria.n_cells())
    , n_dofs_per_cell_(fe.n_dofs())
    , n_dofs_total_(tria.n_cells() * fe.n_dofs())
  {
    ASSERT(tria.n_cells() > 0, "Triangulation must have at least one cell");
  }

  unsigned int n_cells() const { return n_cells_; }
  unsigned int n_dofs_per_cell() const { return n_dofs_per_cell_; }
  unsigned int n_dofs() const { return n_dofs_total_; }

  const FE_DGP<dim, RealType>& fe() const { return fe_; }
  const Triangulation<dim>& tria() const { return tria_; }

private:
  const Triangulation<dim>& tria_;
  const FE_DGP<dim, RealType>& fe_;

  unsigned int n_cells_;
  unsigned int n_dofs_per_cell_;
  unsigned int n_dofs_total_;
};
