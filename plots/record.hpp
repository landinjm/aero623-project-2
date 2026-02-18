#pragma once

#include <iostream>
#include <fstream>
#include <string>

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
    void recordData(double density, double momentum_x, double momentum_y, double energy,
            std::string grid, std::string order, std::string type, std::string value){
        // Open and append a text file
        std::ofstream file((grid+"."+order+"."+type+"."+value+".txt"), std::ios::app);

        // Write to the file
        file << density << " " << momentum_x << " " << momentum_y << " " << energy << std::endl;

        // Close the file
        file.close();
    }
};

