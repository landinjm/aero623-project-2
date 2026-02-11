#include <iostream>
#include <vector>
#include <iomanip>
#include "../src/flux.hpp"

void print_flux(const std::string& label, const std::vector<double>& f) {
    std::cout << label << ": [";
    for(size_t i=0; i<f.size(); ++i) std::cout << (i==0?"":", ") << f[i];
    std::cout << "]\n";
}

int main() {
    double gamma = 1.4;
    double nx = 1.0, ny = 0.0; // Test across a vertical interface

    std::cout << "--- ROE FLUX VERIFICATION TESTS ---\n" << std::endl;

    // TEST 1: CONSISTENCY
    // If uL == uR, the numerical flux must equal the physical flux
    // Let rho=1, u=1, v=0, p=1 -> E = p/(g-1) + 0.5*rho*u^2 = 2.5 + 0.5 = 3.0
    std::vector<double> u_const = {1.0, 1.0, 0.0, 3.0};
    auto F_consist = SWE::flux_roe(u_const, u_const, nx, ny, gamma);
    print_flux("1. Consistency Test (F_num vs F_phys)", F_consist);
    // Expected: [1.0, 2.0, 0.0, 4.0] (rho*u, rho*u^2+p, rho*u*v, (rho*E+p)*u)

    // TEST 2: SYMMETRY
    // Reversing states and normal vector should yield the same magnitude
    std::vector<double> uL = {1.0, 0.0, 0.0, 2.5}; // stationary
    std::vector<double> uR = {0.125, 0.0, 0.0, 0.25}; // Sod low pressure
    auto F1 = SWE::flux_roe(uL, uR, nx, ny, gamma);
    auto F2 = SWE::flux_roe(uR, uL, -nx, -ny, gamma);
    print_flux("2a. Symmetry L->R", F1);
    print_flux("2b. Symmetry R->L (Negated)", {-F2[0], -F2[1], -F2[2], -F2[3]});

    // TEST 3: ENTROPY FIX (STAGNATION POINT)
    // Testing logic for vn_roe close to zero
    std::vector<double> u_stag = {1.0, 0.01, 0.0, 2.50005}; 
    auto F_stag = SWE::flux_roe(u_stag, u_stag, nx, ny, gamma);
    print_flux("3. Stagnation Test", F_stag);

    return 0;
}