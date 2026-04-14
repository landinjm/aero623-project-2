#include <dof_handler.hpp>
#include <fe.hpp>
#include <read_gri.hpp>
#include <triangulation.hpp>

#include <gtest/gtest.h>

using RealType = double;
static constexpr RealType tol = 1.0e-12;
