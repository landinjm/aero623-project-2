#pragma once

#include "tensor.hpp"
#include "triangulation.hpp"

//Requires:
//Modifies: L0
//Effects: Computes the gradient (no limiter) between a cell and its adjacent cells
void ComputeGradients(const Triangulation<2,double>& tri,
                            ElementData<2,double>& elem,
                            int elem_id,
                            std::array<Tensor<1,2,double>,4>& L0);

//Requires: L0 is the cell gradient [1x2] for all 4 states
//          u0 = cell state
//          r = vectors to the cell corners --> computed by ComputeVertexVectors
//          umin = minimum state of all neighboring states
//          umax = maximum state of all neighboring states
//Modifies: L0
//Effects: Computes the scalar limiter gradient via Barth Jespenen
void 
Limiter_BJ(std::array<Tensor<1,2,double>,4>& L0,
           Tensor<1,4,double>& u0,
           Tensor<1,4,double>& umin,
           Tensor<1,4,double>& umax,
           std::array<Tensor<1,2,double>,3>& r);

//Requires
//Modifies
//Effects: Computes the minimum states between a cell and its neighbors
Tensor<1,4,double>
neighbormin(int elem_id,
            const ElementData<2,double>& elem,
            const Triangulation<2,double>& tri);

//Requires
//Modifies
//Effects: Computes the minimum states between a cell and its neighbors
Tensor<1,4,double>
neighbormax(int elem_id,
            const ElementData<2,double>& elem,
            const Triangulation<2,double>& tri);

//Requires:
//Modifies:
//Effects: Computes the vertex vectors of a specific node
std::array<Tensor<1,2,double>,3>
ComputeVertexVectors(unsigned int elem_id,
                     const MeshData& data,
                     const ElementData<2,double>& elem);

//  Code written by Chat-GPT: edited by TJ