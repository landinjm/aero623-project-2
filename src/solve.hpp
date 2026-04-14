#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <config.hpp>
#include <dof_handler.hpp>
#include <fe.hpp>
#include <quad.hpp>
#include <triangulation.hpp>

template<unsigned int dim, unsigned int degree, typename RealType>
class LocalOperator
{
  KOKKOS_FUNCTION LocalOperator() {};

  static constexpr unsigned int n_q_points =
    QGaussSimplex<dim, RealType>::n_q_points(degree + 1);

  static constexpr unsigned int n_local_dofs =
    FE_DGLagrangeSimplex<dim, RealType>::n_dofs_per_cell(degree);
};
