#include <gtest/gtest.h>
#include <iostream>
#include "flux_roe.hpp"
#include "tensor.hpp"

TEST(RoeTest, Consistency) {
    Tensor<1,4,double> U = {1.0, 1.0, 0.0, 3.0};
    Tensor<1,2,double> n = {1.0, 0.0};
    std::pair<Tensor<1,4,double>, double> fluxRoe = SWE::flux_roe(U, U, n);
    Tensor<1,4,double> fluxCell = SWE::Cell_Flux(U, n);
    EXPECT_TRUE(fluxRoe.first == fluxCell);
}

TEST(RoeTest, Conservation) {
    Tensor<1,4,double> UL = {1.0, 3.0, 4.0, 17.5};
    Tensor<1,4,double> UR = {1.0, 1.0, 2.0, 10.0};
    Tensor<1,2,double> n = {0.7071, 0.7071};
    std::pair<Tensor<1,4,double>, double> fluxRoe_1 = SWE::flux_roe(UL, UR, n);
    std::pair<Tensor<1,4,double>, double> fluxRoe_2 = SWE::flux_roe(UR, UL, -1.0 * n);
    EXPECT_TRUE(fluxRoe_1.first == -1.0 * fluxRoe_2.first);
    EXPECT_NEAR(fluxRoe_1.second, fluxRoe_2.second, 1e-10);
}

TEST(RoeTest, EntropyFixStagnation) {
    Tensor<1,4,double> U = {1.0, 0.001, 0.0, 2.5000005}; 
    Tensor<1,2,double> n = {1.0, 0.0};
    std::pair<Tensor<1,4,double>, double> fluxRoe = SWE::flux_roe(U, U, n);
    EXPECT_FALSE(std::isnan(fluxRoe.first[0]));
    EXPECT_GT(fluxRoe.second, 0.0);
}