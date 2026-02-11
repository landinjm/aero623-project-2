// flux.hpp
#ifndef FLUX_HPP
#define FLUX_HPP

#include "tensor.hpp"

//Requires: UL and UR are 4x1 Tensor doubles representing the state [rho, rho*u, rho*v, rho*E]. 
//          n = 2x1 tensor double showing the normal vector
//Modifies:
//Effects: computes the numerical flux and the max propagation speed using the HLLE Riemman solver
std::pair<Tensor<1,4,double>, double>
Flux_HLLE(Tensor<1,4,double> UL,
          Tensor<1,4,double> UR,
          Tensor<1,2,double> n);

//Requires: U is a 4x1 Tensor doubles representing the state [rho, rho*u, rho*v, rho*E]. 
//          n = 2x1 tensor double showing the normal vector
//Modifies:
//Effects: computes the cell flux for a given cell state
Tensor<1,4,double> Cell_Flux(Tensor<1,4,double> U, Tensor<1,2,double> n);

#endif //flux.hpp