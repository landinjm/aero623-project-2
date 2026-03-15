#ifndef QUADRATURE_HPP
#define QUADRATURE_HPP

#include "quad1d_data.hpp"
#include "quad2d_data.hpp"
#include <vector>
#include <stdexcept>

struct Point2D { double x, y, w; };

class Quadrature {
public:
    // For surface integrals along edges
    static std::vector<Point2D> get1D(int order) {
        auto rules = QuadData1D::getRules();
        if (rules.find(order) == rules.end()) throw std::runtime_error("1D Order not found in database.");
        
        std::vector<Point2D> pts;
        auto& r = rules[order];
        for(size_t i=0; i<r.x.size(); ++i) pts.push_back({r.x[i], 0.0, r.w[i]});
        return pts;
    }

    // For volume integrals within elements
    static std::vector<Point2D> get2D(int order) {
        auto rules = QuadData2D::getRules();
        if (rules.find(order) == rules.end()) throw std::runtime_error("2D Order not found in database.");
        
        std::vector<Point2D> pts;
        auto& r = rules[order];
        for(size_t i=0; i<r.x.size(); ++i) pts.push_back({r.x[i], r.y[i], r.w[i]});
        return pts;
    }
};
#endif