#include <iostream>
#include <cmath>
#include <iomanip>
#include "../src/quadrature.hpp"

// Sine test function (Non-polynomial)
double func_sine(double x, double y) {
    return std::sin(x + y);
}

// Polynomial test function
// f(x,y) = x^2 * y + y^2
// Integral over reference triangle (0,0)-(1,0)-(0,1):
// \int_0^1 \int_0^{1-x} (x^2 y + y^2) dy dx = 1/40 + 1/12 = 13/120 = 0.108333333333
double func_poly_2d(double x, double y) {
    return x * x * y + y * y;
}

// 1D Polynomial test function
// f(x) = x^3
// Integral over [0,1]: \int_0^1 x^3 dx = 1/4 = 0.25
double func_poly_1d(double x) {
    return x * x * x;
}

int main() {
    std::cout << std::fixed << std::setprecision(12);
    const double tol = 1e-13;

    try {
        // --- TEST 1: 2D POLYNOMIAL (EXACTNESS) ---
        // Order 8 Dunavant should integrate a degree 3 polynomial exactly.
        int order2d = 8;
        auto pts2d = Quadrature::get2D(order2d);
        double val_poly2d = 0.0;
        for (const auto& p : pts2d) val_poly2d += p.w * func_poly_2d(p.x, p.y);

        double exact_poly2d = 13.0 / 120.0;
        std::cout << "--- 2D Polynomial Test (x^2*y + y^2) ---" << std::endl;
        std::cout << "Result:    " << val_poly2d << std::endl;
        std::cout << "Exact:     " << exact_poly2d << std::endl;
        std::cout << "Error:     " << std::abs(val_poly2d - exact_poly2d) << std::endl;
        if (std::abs(val_poly2d - exact_poly2d) < tol) std::cout << "PASS (Exact)" << std::endl;
        std::cout << std::endl;

        // --- TEST 2: 1D POLYNOMIAL (EXACTNESS) ---
        // Order 5 Gauss-Legendre (3 points) integrates up to degree 2n-1 = 5 exactly.
        int order1d = 5;
        auto pts1d = Quadrature::get1D(order1d);
        double val_poly1d = 0.0;
        for (const auto& p : pts1d) val_poly1d += p.w * func_poly_1d(p.x);

        double exact_poly1d = 0.25;
        std::cout << "--- 1D Polynomial Test (x^3) ---" << std::endl;
        std::cout << "Result:    " << val_poly1d << std::endl;
        std::cout << "Exact:     " << exact_poly1d << std::endl;
        std::cout << "Error:     " << std::abs(val_poly1d - exact_poly1d) << std::endl;
        if (std::abs(val_poly1d - exact_poly1d) < tol) std::cout << "PASS (Exact)" << std::endl;
        std::cout << std::endl;

        // --- TEST 3: 2D SINE (CONVERGENCE) ---
        double val_sine = 0.0;
        for (const auto& p : pts2d) val_sine += p.w * func_sine(p.x, p.y);

        double exact_sine = std::sin(1.0) - std::cos(1.0);
        std::cout << "--- 2D Sine Test (sin(x+y)) ---" << std::endl;
        std::cout << "Result:    " << val_sine << std::endl;
        std::cout << "Exact:     " << exact_sine << std::endl;
        std::cout << "Error:     " << std::abs(val_sine - exact_sine) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}