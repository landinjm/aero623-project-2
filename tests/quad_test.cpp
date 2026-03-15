#include <iostream>
#include <cmath>
#include <iomanip>
#include "../src/quadrature.hpp"

int main() {
    try {
        // Integrate sin(x+y) over reference triangle (0,0)-(1,0)-(0,1)
        // Analytical result is sin(1) - cos(1)
        auto points = Quadrature::get2D(8);
        double integral = 0.0;
        for (const auto& p : points) {
            integral += p.w * std::sin(p.x + p.y);
        }

        double exact = std::sin(1.0) - std::cos(1.0);
        std::cout << std::fixed << std::setprecision(12);
        std::cout << "--- 2D Unit Test ---" << std::endl;
        std::cout << "Numerical: " << integral << std::endl;
        std::cout << "Exact:     " << exact << std::endl;
        std::cout << "Error:     " << std::abs(integral - exact) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
    }
    return 0;
}