#pragma once

#include <iostream>
#include <fstream>
#include <string>
//#include <tensor.hpp>

class Recorder
{
public:
    void recordHist(double data, std::string grid, std::string order, std::string type, std::string value){
        // Open and append a text file
        std::ofstream file((grid+"."+order+"."+type+"."+value+".txt"), std::ios::app);

        // Write to the file
        file << data << std::endl;

        // Close the file
        file.close();
    }

    void cleanFile(std::string grid, std::string order, std::string type, std::string value){
        // Create/clear and open a text file
        std::ofstream file((grid+"."+order+"."+type+"."+value+".txt"));

        // Close the file
        file.close();
    }

    // make this use tensor as well
    void recordData(double density, double momentum_x, double momentum_y, double energy,
                std::string grid, std::string order, std::string type, std::string value){
        /*// Grab values
        const auto rho = interior_state[0];
        const auto momentum_x = interior_state[1];
        const auto momentum_y = interior_state[2];
        const auto energy = interior_state[3];*/
        
        // Open and append a text file
        std::ofstream file((grid+"."+order+"."+type+"."+value+".txt"), std::ios::app);

        // Write to the file
        // for loop over each element
        file << density << " " << momentum_x << " " << momentum_y << " " << energy << std::endl;

        // Close the file
        file.close();
    }
};

