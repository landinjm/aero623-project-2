#include "limiters.cpp"
#include <algorithm>

Tensor<1,2,double> Gradient(double l[], double A, Tensor<1,2,double> n[]) {
    //Gradient = Sum(edge-state * normal * length) / Area of Cell
    Tensor<1,2,double> sum = {0,0};
    for (int i = 0; i < 3; i++) {
        sum = sum + l[i] * n[i] * ;//TODO * u_hat --> average between the left and right averaged states
    }//end for
    sum = sum / A;
    return sum;
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
    
    double alpha = 0;
    for (i = 0; i < 3; i++) {
        double alpha_cmp;
        //compute the required scalar limiter for each node
        if (uiN - u0 > 0) { //TODO
            //TODO
        } else if (uiN - u0 < 0) { //TODO
            //TODO
        } else {
            //TODO
        }
        //set scalar limiter to the minimum of the adjacent nodes
        alpha = std::min(alpha, alpha_cmp);
    }

    //return the limited gradient
    return alpha * L0;
}//end Limiter_BJ
