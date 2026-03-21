#pragma once

template<unsigned int dim, typename RealType>
class Mapping
{
    // //FUNCTIONALITIES for curved elements

    // //Get global coordiante from reference coordinate
    // //Compute the Normal
    // //Compute the Jacobian

    // public:

    // void normal() {

    // }

    // void Jacobian() {

    // }

    // void ref_cord(RealType& invJ, RealType& xn) {
    //     //initialize the Newton Raphson Convergance
    //     //TODO - set the initial guess
    //     double tol = 1e-10;
    //     double dmax = 1.0;
    //     unsigned int iNewton = 0;
    //     bool converged = 0;

    //     //loop through Newton Raphson until converged
    //     while(!converged) {

    //         //map point to global coords

    //         //calcualate the residual

    //         //check if the residual is below the tolerance

    //         //calculate the residual lineraization = mapping Jacobian

    //         //determine state update: dxref = -invJ*R

    //         //limit update

    //         //apply state update

    //         //increment the iteration number
    //         ++iNewton;
    //         ASSERT(iNewton < 100, "Too many Newton Raphson iterations");
    //     }
    // }

    // private:

};
