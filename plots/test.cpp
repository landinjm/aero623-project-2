#include "record.hpp"
#include "forcecoeffs.hpp"
#include <string>
#include <vector>
#include <iostream>

int main() {
    std::string grid = "testgrid";
    std::string order = "order";
    std::string type = "steady";
    double err = 2.01;
    Recorder rec;
    rec.cleanFile(grid,order,type,"error");
    rec.recordHist(err,grid,order,type,"error");
    rec.recordHist(err,grid,order,type,"error");
    rec.recordHist(err,grid,order,type,"error");
    rec.recordData(err,err,err,err,grid,order,type,"error");
    CalcForceCoeffs calc;
    double density = 1.0;
    double momentum_x = 0.5;
    double momentum_y = 0.5;
    double energy = 2.5;
    std::vector<double> coeffs = calc.calcForceCoeffs(density, momentum_x, momentum_y, energy, 1.0, -1.0);
    std::cout << "Force coefficients: cx = " << coeffs[0] << ", cy = " << coeffs[1] << std::endl;

    return 0;
}

