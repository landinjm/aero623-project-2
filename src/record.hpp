#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <tensor.hpp>
#include <triangulation.hpp>
#include <vector>

template<unsigned int dim, typename RealType>
class Recorder
{
public:
    void cleanFile(std::string grid, std::string order, std::string type, std::string value){
        // Create/clear and open a text file
        std::ofstream file((grid+"."+order+"."+type+"."+value+".txt"));

        // Close the file
        file.close();
    }

    void recordHist(std::vector<double> data, std::string grid, std::string order, std::string type, std::string value){
        // Open and append a text file
        std::ofstream file((grid+"."+order+"."+type+"."+value+".txt"), std::ios::app);

        // Write to the file
        for (const auto& d : data) {
            file << d << "\n";
        }
        file << std::endl;

        // Close the file
        file.close();
    }

    void recordData(ElementData<dim, RealType>& element_state,
                std::string grid, std::string order, std::string type){
        // Grab values
        const auto rho = element_state[0];
        const auto momentum_x = element_state[1];
        const auto momentum_y = element_state[2];
        const auto energy = element_state[3];
        
        // Open and append a text file
        std::ofstream file((grid+"."+order+"."+type+".data.txt"), std::ios::app);

        // Write to the file
        for (unsigned int i = 0; i < element_state.size(); ++i) {
            file << rho[i] << " " << momentum_x[i] << " " << momentum_y[i] << " " << energy[i] << "\n";
        }
        file << std::endl;

        // Close the file
        file.close();
    }

    void recordSimulation(ElementData<dim, RealType>& element_state,
            std::vector<double> residual, std::vector<double> c_x, std::vector<double> c_y,
            std::string grid, std::string order, std::string type){
        cleanFile(grid, order, type, "data");
        cleanFile(grid, order, type, "residual");
        cleanFile(grid, order, type, "c_x");
        cleanFile(grid, order, type, "c_y");
        recordData(element_state, grid, order, type);
        recordHist(residual, grid, order, type, "residual");
        recordHist(c_x, grid, order, type, "c_x");
        recordHist(c_y, grid, order, type, "c_y");
    }
};

