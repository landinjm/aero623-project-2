#pragma once

#include "tensor.hpp"
#include "triangulation.hpp"

//Requires:
//Modifies:
//Effects: Computes the gradient (no limiter) between a cell and its adjacent cells
//  Written by Chat-GPT
template<typename RealType>
void ComputeGradients(const Triangulation<2,RealType>& tri,
                      ElementData<2,RealType>& elem,
                      std::vector<Tensor<1,2,RealType>>& grad_rho);

//Requires: L0 is the cell gradient [1x2] for all 4 states
//          u0 = cell state
//          r = vectors to the cell corners --> computed by ComputeVertexVectors
//Modifies:
//Effects: Computes the scalar limiter gradient via Barth Jespenen
Limiter_BJ(const Tensor<1,4,Tensor<1,2,double>>& L0,
           const Tensor<1,4,double>& u0,
           const std::array<Tensor<1,2,double>,3>& r);

//Requires:
//Modifies:
//Effects: Computes the vertex vectors of a specific node
//  Written by Chat-GPT
template<typename RealType>
std::array<Tensor<1,2,RealType>,3>
ComputeVertexVectors(unsigned int elem_id,
                     const MeshData& data,
                     const ElementData<2,RealType>& elem);