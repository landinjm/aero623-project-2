#pragma once

#include "tensor.hpp"
#include "triangulation.hpp"

//Requires:
//Modifies:
//Effects: Computes the gradient (no limiter) between a cell and its adjacent cells
Tensor<1,2,double> Gradient();

//Requires:
//Modifies:
//Effects: Computes a limited gradient based on Barth Jespenen
Tensor<1,2,double> Limiter_BJ();