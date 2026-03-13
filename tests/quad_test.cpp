#include <iostream>
#include <iomanip>
#include <cmath>
#include "quadrature.hpp"

// Test function: f(x,y) = sin(x+y)
double f_test(double x, double y) { return std::sin(x + y); }

int main() {
    // 1. Test 1D Quadrature: Integral of sin(x) on [0, 1]
    // Analytical: -cos(1) + cos(0) = 0.45969769413
    double analytical1D = 1.0 - std::cos(1.0);
    auto q1d = Quadrature::get1D(5); // Order 5
    double num1D = 0;
    for (auto& p : q1d) num1D += p.w * std::sin(p.x);
    
    std::cout << "--- 1D Unit Test (sin(x) on [0,1]) ---" << std::endl;
    std::cout << "Analytical: " << std::fixed << std::setprecision(12) << analytical1D << std::endl;
    std::cout << "Numerical:  " << num1D << std::endl;
    std::cout << "Error:      " << std::abs(num1D - analytical1D) << "\n" << std::endl;

    // 2. Test 2D Quadrature: Integral of sin(x+y) on unit triangle
    // Analytical: sin(1) - cos(1) = 0.30116867893
    double analytical2D = std::sin(1.0) - std::cos(1.0);
    auto q2d = Quadrature::get2D(8); // High order for accuracy
    
    // Mapping for a physical triangle (matching fint.m logic)
    // Vertices: (0,0), (1,0), (0,1)
    double x1 = 0, y1 = 0, x2 = 1, y2 = 0, x3 = 0, y3 = 1;
    double detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    
    double num2D = 0;
    for (auto& p : q2d) {
        double x_phys = x1 + (x2 - x1) * p.x + (x3 - x1) * p.y;
        double y_phys = y1 + (y2 - y1) * p.x + (y3 - y1) * p.y;
        num2D += p.w * f_test(x_phys, y_phys);
    }
    num2D *= std::abs(detJ); // detJ is constant for linear triangles

    std::cout << "--- 2D Unit Test (sin(x+y) on Unit Triangle) ---" << std::endl;
    std::cout << "Analytical: " << analytical2D << std::endl;
    std::cout << "Numerical:  " << num2D << std::endl;
    std::cout << "Error:      " << std::abs(num2D - analytical2D) << std::endl;

    return 0;
}