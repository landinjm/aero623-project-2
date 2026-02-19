#pragma once

#include "tensor.hpp"
#include "triangulation.hpp"

//Requires:
//Modifies:
//Effects: Computes the gradient (no limiter) between a cell and its adjacent cells
//  Written by Chat-GPT
Tensor<2,4,double> ComputeGradients(const Triangulation<2,double>& tri,
                           ElementData<2,double>& elem);

//Requires: L0 is the cell gradient [1x2] for all 4 states
//          u0 = cell state
//          r = vectors to the cell corners --> computed by ComputeVertexVectors
//Modifies:
//Effects: Computes the scalar limiter gradient via Barth Jespenen
Tensor<2,4,double> 
Limiter_BJ(Tensor<2,4,double>& L0,
           Tensor<1,4,double>& u0,
           std::array<Tensor<1,2,double>,3>& r);

//Requires:
//Modifies:
//Effects: Computes the vertex vectors of a specific node
//  Written by Chat-GPT
std::array<Tensor<1,2,double>,3>
ComputeVertexVectors(unsigned int elem_id,
                     const MeshData& data,
                     const ElementData<2,double>& elem);