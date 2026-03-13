#ifndef QUADRATURE_HPP
#define QUADRATURE_HPP

#include <vector>
#include <cmath>

struct QuadPoint1D { double x, w; };
struct QuadPoint2D { double x, y, w; };

class Quadrature {
public:
    // 1D Gauss-Legendre (Surface Integrals). Points on [0, 1].
    static std::vector<QuadPoint1D> get1D(int order) {
        if (order <= 1) return {{0.5, 1.0}};
        if (order <= 3) return {{0.211324865405187, 0.5}, {0.788675134594813, 0.5}};
        if (order <= 5) return {{0.112701665379258, 0.277777777777778}, {0.5, 0.444444444444444}, {0.887298334620742, 0.277777777777778}};
        // Higher orders from quad1d.c for curved edges (order 9 shown)
        return {{0.046910077030668, 0.118463442528095}, {0.230765344947158, 0.239314335249683}, {0.5, 0.284444444444444}, {0.769234655052841, 0.239314335249683}, {0.953089922969332, 0.118463442528095}};
    }

    // 2D Dunavant (Volume Integrals). Reference Triangle: (0,0)-(1,0)-(0,1).
    static std::vector<QuadPoint2D> get2D(int order) {
        if (order <= 1) return {{0.333333333333333, 0.333333333333333, 0.5}};
        if (order <= 2) return {{0.666666666666667, 0.166666666666667, 0.166666666666667}, {0.166666666666667, 0.166666666666667, 0.166666666666667}, {0.166666666666667, 0.666666666666667, 0.166666666666667}};
        // Order 8 (16 points) - Recommended for p=3 with curved elements
        return {
            {0.333333333333333, 0.333333333333333, 0.072108514126988}, {0.459292588292723, 0.459292588292723, 0.047071333312021},
            {0.081414823414554, 0.459292588292723, 0.047071333312021}, {0.459292588292723, 0.081414823414554, 0.047071333312021},
            {0.054415842243082, 0.054415842243082, 0.016229721510944}, {0.891168315513837, 0.054415842243082, 0.016229721510944},
            {0.054415842243082, 0.891168315513837, 0.016229721510944}, {0.170569307751760, 0.170569307751760, 0.035154024471904},
            {0.658861384496480, 0.170569307751760, 0.035154024471904}, {0.170569307751760, 0.658861384496480, 0.035154024471904},
            {0.007018677314915, 0.496490661342542, 0.012548237778551}, {0.496490661342542, 0.007018677314915, 0.012548237778551},
            {0.496490661342542, 0.496490661342542, 0.012548237778551}, {0.210334710104841, 0.394832644947579, 0.026341392284955},
            {0.394832644947579, 0.210334710104841, 0.026341392284955}, {0.394832644947579, 0.394832644947579, 0.026341392284955}
        };
    }
};

// Orthonormal Dubiner Basis for Triangles
class Basis {
public:
    static double evaluate(int i, int j, double xi, double eta) {
        // Linearized indexing for p=0,1,2,3: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)...
        // Implementation uses coordinate mapping to apply Jacobi polynomials
        double a = (eta < 1.0) ? (2.0 * xi / (1.0 - eta) - 1.0) : -1.0;
        double b = 2.0 * eta - 1.0;
        return std::sqrt(2.0) * jacobiP(i, 0, 0, a) * std::pow(1.0 - eta, i) * jacobiP(j, 2 * i + 1, 0, b);
    }

private:
    static double jacobiP(int n, int alpha, int beta, double x) {
        if (n == 0) return 1.0;
        if (n == 1) return 0.5 * (alpha - beta + (alpha + beta + 2.0) * x);
        // Recurrence for higher orders (p=2, 3)
        double p0 = 1.0, p1 = 0.5 * (alpha - beta + (alpha + beta + 2.0) * x), p2;
        for (int k = 1; k < n; ++k) {
            double a1 = 2.0 * (k + 1) * (k + alpha + beta + 1) * (2 * k + alpha + beta);
            double a2 = (2 * k + alpha + beta + 1) * (alpha * alpha - beta * beta);
            double a3 = (2 * k + alpha + beta) * (2 * k + alpha + beta + 1) * (2 * k + alpha + beta + 2);
            double a4 = 2.0 * (k + alpha) * (k + beta) * (2 * k + alpha + beta + 2);
            p2 = ((a2 + a3 * x) * p1 - a4 * p0) / a1;
            p0 = p1; p1 = p2;
        }
        return p1;
    }
};

#endif