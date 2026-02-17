#include "record.hpp"
#include <string>

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
    rec.recordData(err,grid,order,type,"error");

    return 0;
}

