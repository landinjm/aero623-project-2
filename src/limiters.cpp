#include "limiters.cpp"
#include <algorithm>

Tensor<1,2,double> Gradient() {
    //Gradient = Sum(edge-state * normal * length) / Area of Cell
}//end Gradient

double Limiter_BJ() {
    //get adjacent states
    double u0 = //TODO
    double u1 = //TODO
    double u2 = //TODO
    double u3 = //TODO

    //find umin and umax based on adjacent states
    double umin = std::min(u0, u1, u2, u3);
    double umax = std::max(u0, u1, u2, u3);

    //compute the current local gradient
    Tensor<1,2,double> L0 = Gradient(); //TODO

    //compute states at adjacent nodes

    //compute the required scalar limiter for each node
    if (uiN - u0 > 0) { //TODO
        //TODO
    } else if (uiN - u0 < 0) { //TODO
        //TODO
    } else {
        //TODO
    }

    //set scalar limiter to the minimum of the adjacent nodes

    //return the limited gradient
}//end Limiter_BJ
